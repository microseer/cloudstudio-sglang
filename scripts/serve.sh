#!/usr/bin/env bash
# =============================================================================
# CloudStudio × MiniMax-H3 Ref2VA 应用模板 - SGLang 服务一键启动
# 被 .vscode/preview.yml 的 run 调用，也可在终端直接执行。
#
# 自动识别两种权重形态：
#
# 【量化模式 prequant】（本模板默认，面向 A10 / RTX 3090 / 4090 等 24GB 单卡）
#   检测到 /workspace/models/Comfy-MiniMax-H3 下的 Comfy 量化权重时启用，
#   对齐官方消费级卡配方（32GB 主机 / 22GiB cap 实测基准，480P）：
#     --component-weights-paths.{transformer,text_encoder,video_vae,audio_vae}
#                                 指向 4 个量化权重（预量化文件自描述，禁止再
#                                 加 --quantization）
#     --performance-mode memory
#     --layerwise-offload-components dit,text_encoder,vae
#                                 （VAE 也逐层 offload，搭配 video_vae=36 部分
#                                  常驻——不带常驻的纯 offload 会让解码 167 个
#                                  tile 各重复流式传输约 9GB）
#     --layerwise-resident-layers video_vae=36
#     --dit-offload-prefetch-size 1 --dit-layerwise-resident-layers 6
#     --attention-backend fa  --enable-torch-compile false
#   官方实测：6 层 DiT 常驻 + 32GB 主机 ≈ 8.5s/步、解码约 9.6s。
#   OOM 降档顺序（官方注释原文，先快后稳）：768P 下先把 DiT 常驻降到 0，
#   解码仍冲突再把 video_vae 降到 24；这两个旋钮按此顺序用显存换余量。
#   ConvRot INT8 在 CC 8.6/8.9 自动走 comfy-kitchen（须先安装）。
#
# 【在线量化模式 online】已下载官方 BF16 全量权重（H3_WEIGHTS=online）且
#   单卡运行时启用：加载时在线量化 --quantization convrot_int8 + 同样的
#   layerwise offload（官方 1×4090 实测路径，PSNR≈24.8dB，峰值显存约 18GB）。
#
# 【全精度模式 bf16】多卡 + 官方 BF16 全量权重：沿用 4 卡 Ulysses4 / speed。
#
# 参考：
#   https://docs.sglang.io/cookbook/diffusion/MiniMax/MiniMax-H3
#   https://docs.sglang.io/docs/sglang-diffusion/quantization
#
# 用法：
#   bash /workspace/scripts/serve.sh                    # 自动检测，自动选配方
#   bash /workspace/scripts/serve.sh /path/to/base-repo # 指定官方配置树根目录
#   bash /workspace/scripts/serve.sh --enforce-eager    # 其余参数透传 sglang serve
#
# 可用环境变量：
#   SGLANG_HOST=0.0.0.0           监听地址
#   SGLANG_PORT=30011             服务端口（Ref2VA 官方约定 30011，FL2VA 为 30010）
#   SGLANG_PROFILE=auto           auto|prequant|online|bf16（强制指定配方）
#   SGLANG_NUM_GPUS=1             GPU 数（量化配方默认 1；全精度默认检测到的卡数）
#   SGLANG_ULYSSES_DEGREE=        Ulysses 度（默认与 GPU 数相同）
#   SGLANG_TP_SIZE=               可选 tensor parallel size
#   SGLANG_DIT_RESIDENT_LAYERS=6  量化模式常驻 DiT 层数（官方 32GB 主机基准值；
#                                 OOM 时先降 0，速度换显存）
#   SGLANG_VIDEO_VAE_RESIDENT=36  video_vae 部分常驻层数（解码 OOM 时降到 24）
#   SGLANG_OFFLOAD_PREFETCH=1     DiT layerwise 预取层数
#   SGLANG_WARMUP_RESOLUTION=     可选，按实际出片分辨率预热（如 1024x576），
#                                 避免默认 1344x768x124 帧预热拖慢启动
#   SGLANG_WARMUP_FRAMES=         可选，预热帧数（与上一项搭配使用）
#   SGLANG_ATTENTION_BACKEND=fa   注意力后端（fa 精确；sol_attn/sage_attn 更快但有损）
#   COMFY_MODEL_DIR=...           量化权重目录（默认 /workspace/models/Comfy-MiniMax-H3）
#   SGLANG_EXTRA_ARGS='...'       额外启动参数（字符串形式）
#   SGLANG_MODEL=...              强制指定官方配置树根路径 / 模型 ID
# =============================================================================
set -euo pipefail

log()  { printf '\033[1;32m[serve]\033[0m %s\n' "$*"; }
warn() { printf '\033[1;33m[serve]\033[0m %s\n' "$*" >&2; }
die()  { printf '\033[1;31m[serve]\033[0m %s\n' "$*" >&2; exit 1; }

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
VENV="${SGLANG_VENV:-$PROJECT_DIR/.venv}"

HOST="${SGLANG_HOST:-0.0.0.0}"
PORT="${SGLANG_PORT:-30011}"
MODEL_ID="MiniMax/MiniMax-H3"
LOCAL_MODEL="$PROJECT_DIR/models/MiniMax-H3"
COMFY_MODEL_DIR="${COMFY_MODEL_DIR:-$PROJECT_DIR/models/Comfy-MiniMax-H3}"

# -----------------------------------------------------------------------------
# 1. 环境检查
# -----------------------------------------------------------------------------
[ -x "$VENV/bin/sglang" ] || die "尚未安装 SGLang（未找到 $VENV/bin/sglang）。
       请先在终端执行：bash $SCRIPT_DIR/setup.sh"

if ! command -v nvidia-smi >/dev/null 2>&1; then
  warn "未检测到 nvidia-smi：当前可能是 CPU 算力规格。"
  die "MiniMax-H3 Ref2VA 必须在 GPU 算力下运行，请先在 CloudStudio 切换 GPU 算力规格（A10 24GB 即可）。"
fi

GPU_COUNT="$(nvidia-smi -L 2>/dev/null | wc -l | tr -d ' ')"
[ "$GPU_COUNT" -ge 1 ] || die "未检测到可用 GPU（nvidia-smi -L 为空）。"

for bin in ffmpeg ffprobe; do
  command -v "$bin" >/dev/null 2>&1 || die "未找到 $bin。请执行：bash $SCRIPT_DIR/install-cuda.sh（会通过 apt 安装 ffmpeg）"
done

# -----------------------------------------------------------------------------
# 2. 确定模型路径 / 透传参数
# -----------------------------------------------------------------------------
PASSTHROUGH=()
MODEL=""
for arg in "$@"; do
  if [ -z "$MODEL" ] && [[ "$arg" != --* ]]; then
    MODEL="$arg"
  else
    PASSTHROUGH+=("$arg")
  fi
done
MODEL="${MODEL:-${SGLANG_MODEL:-}}"

HAS_SLIM_TREE=0; [ -f "$LOCAL_MODEL/model_index.json" ] && HAS_SLIM_TREE=1
HAS_FULL_WEIGHTS=0
if compgen -G "$LOCAL_MODEL/Ref2VA/transformer/model-*.safetensors" >/dev/null 2>&1; then
  HAS_FULL_WEIGHTS=1
fi

# 量化权重候选（文件名与 download-model.sh 的选择保持一致，可用环境变量覆盖）
DIT_WEIGHTS="${SGLANG_DIT_WEIGHTS:-}"
TE_WEIGHTS="${SGLANG_TE_WEIGHTS:-}"
VIDEO_VAE_WEIGHTS="${SGLANG_VIDEO_VAE_WEIGHTS:-$COMFY_MODEL_DIR/vae/minimax_h3_video_vae_fp16.safetensors}"
AUDIO_VAE_WEIGHTS="${SGLANG_AUDIO_VAE_WEIGHTS:-$COMFY_MODEL_DIR/vae/minimax_h3_audio_vae_fp32.safetensors}"
if [ -z "$DIT_WEIGHTS" ]; then
  for f in \
    diffusion_models/minimax_h3_ref2va_pruned_int8_convrot.safetensors \
    diffusion_models/minimax_h3_ref2va_int8_convrot.safetensors \
    diffusion_models/minimax_h3_ref2va_pruned_w6a8.safetensors; do
    [ -f "$COMFY_MODEL_DIR/$f" ] && { DIT_WEIGHTS="$COMFY_MODEL_DIR/$f"; break; }
  done
fi
if [ -z "$TE_WEIGHTS" ]; then
  for f in \
    text_encoders/qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors \
    text_encoders/qwen3vl_32b_minimax_h3_int8_convrot.safetensors; do
    [ -f "$COMFY_MODEL_DIR/$f" ] && { TE_WEIGHTS="$COMFY_MODEL_DIR/$f"; break; }
  done
fi
HAS_QUANT_WEIGHTS=0
if [ -n "$DIT_WEIGHTS" ] && [ -n "$TE_WEIGHTS" ] \
   && [ -f "$VIDEO_VAE_WEIGHTS" ] && [ -f "$AUDIO_VAE_WEIGHTS" ]; then
  HAS_QUANT_WEIGHTS=1
fi

# -----------------------------------------------------------------------------
# 3. 选择配方：auto = 量化文件优先 > 单卡全量(在线量化) > 多卡全精度
# -----------------------------------------------------------------------------
PROFILE="${SGLANG_PROFILE:-auto}"
if [ "$PROFILE" = "auto" ]; then
  if [ "$HAS_QUANT_WEIGHTS" = 1 ]; then
    PROFILE="prequant"
  elif [ "$HAS_FULL_WEIGHTS" = 1 ] && [ "$GPU_COUNT" -ge 4 ]; then
    PROFILE="bf16"
  elif [ "$HAS_FULL_WEIGHTS" = 1 ]; then
    PROFILE="online"
  else
    die "未找到模型权重。请先执行：bash $SCRIPT_DIR/download-model.sh
       （默认下载量化版约 43GB 到 $LOCAL_MODEL 与 $COMFY_MODEL_DIR）"
  fi
fi

case "$PROFILE" in
  prequant)
    [ "$HAS_SLIM_TREE" = 1 ] || die "量化模式需要官方精简配置树 $LOCAL_MODEL/model_index.json，
       请先执行：bash $SCRIPT_DIR/download-model.sh"
    [ "$HAS_QUANT_WEIGHTS" = 1 ] || die "量化权重不完整。请重新执行：bash $SCRIPT_DIR/download-model.sh
       DiT=$DIT_WEIGHTS
       TE =$TE_WEIGHTS
       VAE=$VIDEO_VAE_WEIGHTS
       AudioVAE=$AUDIO_VAE_WEIGHTS"
    [ -n "$MODEL" ] || MODEL="$LOCAL_MODEL"
    ;;
  online)
    [ "$HAS_FULL_WEIGHTS" = 1 ] || die "在线量化模式需要官方 BF16 全量权重，请执行：
       H3_WEIGHTS=online bash $SCRIPT_DIR/download-model.sh"
    [ -n "$MODEL" ] || MODEL="$LOCAL_MODEL"
    ;;
  bf16)
    [ "$HAS_FULL_WEIGHTS" = 1 ] || {
      [ -n "$MODEL" ] || {
        MODEL="$MODEL_ID"
        export SGLANG_USE_MODELSCOPE="${SGLANG_USE_MODELSCOPE:-true}"
        warn "本地无全量权重，将使用 ModelScope 在线拉取（约 144GB），建议先 download-model.sh。"
      }
    }
    [ -n "$MODEL" ] || MODEL="$LOCAL_MODEL"
    ;;
  *) die "未知 SGLANG_PROFILE=$PROFILE（可选 auto|prequant|online|bf16）" ;;
esac
export MODELSCOPE_CACHE="${MODELSCOPE_CACHE:-$PROJECT_DIR/.cache/modelscope}"

# -----------------------------------------------------------------------------
# 4. 拓扑与配方参数
# -----------------------------------------------------------------------------
NUM_GPUS="${SGLANG_NUM_GPUS:-}"
if [ -z "$NUM_GPUS" ]; then
  if [ "$PROFILE" = "bf16" ]; then NUM_GPUS="$GPU_COUNT"; else NUM_GPUS=1; fi
fi
[ "$GPU_COUNT" -ge "$NUM_GPUS" ] || die "请求 ${NUM_GPUS} 张 GPU，但当前仅检测到 ${GPU_COUNT} 张。"
ULYSSES_DEGREE="${SGLANG_ULYSSES_DEGREE:-$NUM_GPUS}"

TOPO_ARGS=(--num-gpus "$NUM_GPUS" --ulysses-degree "$ULYSSES_DEGREE")
if [ -n "${SGLANG_TP_SIZE:-}" ]; then
  TOPO_ARGS+=(--tensor-parallel-size "$SGLANG_TP_SIZE")
fi

RECIPE_ARGS=()
case "$PROFILE" in
  prequant|online)
    # 减少显存碎片（官方注释明确要求；解码贴近显存上限时无此项会被碎片压垮）
    export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
    RESIDENT="${SGLANG_DIT_RESIDENT_LAYERS:-6}"
    VAE_RESIDENT="${SGLANG_VIDEO_VAE_RESIDENT:-36}"
    PREFETCH="${SGLANG_OFFLOAD_PREFETCH:-1}"
    ATTN="${SGLANG_ATTENTION_BACKEND:-fa}"
    # VAE 加入 offload 必须搭配 video_vae 部分常驻（官方消费级卡配方），
    # 否则 167 个解码 tile 每个都要重流约 9GB
    RECIPE_ARGS=(
      --attention-backend "$ATTN"
      --performance-mode memory
      --layerwise-offload-components dit,text_encoder,vae
      --layerwise-resident-layers "video_vae=$VAE_RESIDENT"
      --dit-offload-prefetch-size "$PREFETCH"
      --dit-layerwise-resident-layers "$RESIDENT"
      --enable-torch-compile false
    )
    if [ -n "${SGLANG_WARMUP_RESOLUTION:-}" ]; then
      RECIPE_ARGS+=(--warmup-resolutions "$SGLANG_WARMUP_RESOLUTION")
    fi
    if [ -n "${SGLANG_WARMUP_FRAMES:-}" ]; then
      RECIPE_ARGS+=(--warmup-num-frames "$SGLANG_WARMUP_FRAMES")
    fi
    if [ "$PROFILE" = prequant ]; then
      # 预量化文件自描述：禁止再加 --quantization（SGLang 量化文档明确要求）
      RECIPE_ARGS+=(
        --component-weights-paths.transformer "$DIT_WEIGHTS"
        --component-weights-paths.text_encoder "$TE_WEIGHTS"
        --component-weights-paths.video_vae "$VIDEO_VAE_WEIGHTS"
        --component-weights-paths.audio_vae "$AUDIO_VAE_WEIGHTS"
      )
    else
      # 官方 BF16 全量 + 加载时在线量化（ConvRot W8A8，无需校准数据）
      RECIPE_ARGS+=(--quantization convrot_int8)
    fi
    ;;
  bf16)
    RECIPE_ARGS=(--performance-mode speed)
    ;;
esac

# shellcheck disable=SC2206  # SGLANG_EXTRA_ARGS 有意按空格拆分
EXTRA=( ${SGLANG_EXTRA_ARGS:-} )

# -----------------------------------------------------------------------------
# 5. 量化依赖与主机内存提示
# -----------------------------------------------------------------------------
if [ "$PROFILE" != "bf16" ]; then
  if ! "$VENV/bin/python" -c 'import importlib.metadata,sys
sys.exit(0 if importlib.metadata.distribution("comfy-kitchen") else 1)' 2>/dev/null; then
    die "量化模式需要 comfy-kitchen（A10/3090/4090 的 ConvRot INT8 后端）。
       请执行：bash $SCRIPT_DIR/install-sglang.sh（会自动 pip install comfy-kitchen）"
  fi
  MEM_TOTAL_GB="$(awk '/MemTotal/ {printf "%d", $2/1024/1024}' /proc/meminfo 2>/dev/null || echo 0)"
  if [ "$MEM_TOTAL_GB" -gt 0 ] && [ "$MEM_TOTAL_GB" -lt 60 ]; then
    warn "主机内存约 ${MEM_TOTAL_GB}GB（<64GB）：权重将主要从 NVMe 逐层流式加载，"
    warn "请确认系统盘为 NVMe SSD（SATA SSD 不可用），并预期更慢的生成速度。"
  fi
fi

# shellcheck disable=SC1091
source "$VENV/bin/activate"

log "配方：$PROFILE | model=$MODEL"
if [ "$PROFILE" = "prequant" ]; then
  log "量化权重："
  log "  transformer = $DIT_WEIGHTS"
  log "  text_encoder = $TE_WEIGHTS"
  log "  video_vae   = $VIDEO_VAE_WEIGHTS"
  log "  audio_vae   = $AUDIO_VAE_WEIGHTS"
fi
log "拓扑：gpus=$NUM_GPUS ulysses=$ULYSSES_DEGREE${SGLANG_TP_SIZE:+ tp=$SGLANG_TP_SIZE}，端口：$PORT"
if [ "$PROFILE" != "bf16" ]; then
  log "offload：dit+text_encoder+vae，DiT 常驻 $RESIDENT 层，video_vae 常驻 $VAE_RESIDENT 层"
  log "OOM 降档（官方顺序）：SGLANG_DIT_RESIDENT_LAYERS=0 → 仍冲突则 SGLANG_VIDEO_VAE_RESIDENT=24"
fi
log "服务就绪后测试：curl http://localhost:$PORT/health"
exec sglang serve \
  --model-path "$MODEL" \
  --model-variant ref2va \
  "${TOPO_ARGS[@]}" \
  "${RECIPE_ARGS[@]}" \
  --host "$HOST" \
  --port "$PORT" \
  "${EXTRA[@]}" \
  "${PASSTHROUGH[@]}"
