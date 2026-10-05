#!/usr/bin/env bash
# =============================================================================
# CloudStudio × MiniMax-H3 Ref2VA 应用模板 - 模型下载脚本（A10 24GB 量化方案）
#
# 默认下载「量化版」组合（合计约 42.5GB，可在单张 24GB 卡 A10 / RTX 3090 / 4090
# 上配合 layerwise offload 运行），分两部分：
#
#   A. 官方仓库 MiniMax/MiniMax-H3 的「精简配置树」（仅几十 MB）：
#      model_index.json + Ref2VA/{model_index.json,processor,tokenizer,
#      transformer/*.json,text_encoder 配置,video_vae/audio_vae 代码与配置}
#      —— 不下任何官方权重分片（BF16 全量 Ref2VA 约 144GB）。
#   B. Comfy-Org/MiniMax-H3 的 4 个量化权重文件（魔搭镜像，公开可访问）：
#        DiT ........ minimax_h3_ref2va_pruned_int8_convrot.safetensors  20.97GB
#                     （AdaLN 精简 + ConvRot INT8；推理用不到 AdaLN 调制分支）
#        文本编码器 . qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors      15.69GB
#                     （NVFP4-AWQ 为「压缩存储、BF16 计算」，无需 Blackwell）
#        视频 VAE ... minimax_h3_video_vae_fp16.safetensors              5.21GB
#        音频 VAE ... minimax_h3_audio_vae_fp32.safetensors              0.61GB
#
# 依据：
#   SGLang Cookbook（1×RTX 4090 24GB 实测配方 / 量化格式表）：
#     https://docs.sglang.io/cookbook/diffusion/MiniMax/MiniMax-H3
#   SGLang 量化文档：comfy-int8-convrot（CC 非 9.0/10.0/12.0/12.1 时走
#     comfy-kitchen，支持 Turing+，A10 为 Ampere SM8.6 可用）、NVFP4-AWQ：
#     https://docs.sglang.io/docs/sglang-diffusion/quantization
#
# 用法（CloudStudio 工作目录 /workspace）：
#   bash /workspace/scripts/download-model.sh                 # 默认 quant（约43GB）
#
#   H3_WEIGHTS=online bash /workspace/scripts/download-model.sh
#       # 下载官方 BF16 全量 Ref2VA（约 144GB），供在线 --quantization convrot_int8
#       # 即官方 1×RTX 4090 实测路径；磁盘/内存充足时与量化版二选一
#
#   H3_DIT_QUANT=w6a8 bash /workspace/scripts/download-model.sh
#       # DiT 改用 W6A8（约 16GB，更省显存，画质有损失，需 comfy-kitchen>=0.2.27）
#   H3_TE_QUANT=int8_convrot bash /workspace/scripts/download-model.sh
#       # 文本编码器改用 ConvRot INT8（约 27.1GB）
#
#   bash /workspace/scripts/download-model.sh /workspace/models/MiniMax-H3
#       # 自定义官方精简树目录（第 1 个位置参数）；量化文件目录用 COMFY_MODEL_DIR
#
#   MAX_WORKERS=8 bash /workspace/scripts/download-model.sh   # 下载并发数
#
# 特性：ModelScope 国内 CDN、断点续传，中断后重新执行同一命令即可续传。
# =============================================================================
set -euo pipefail

log()  { printf '\033[1;36m[modelscope]\033[0m %s\n' "$*"; }
warn() { printf '\033[1;33m[modelscope]\033[0m %s\n' "$*" >&2; }
die()  { printf '\033[1;31m[modelscope]\033[0m %s\n' "$*" >&2; exit 1; }

# 项目根目录 = 本脚本所在目录的上一级（CloudStudio 中解析为 /workspace）
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

OFFICIAL_ID="MiniMax/MiniMax-H3"
COMFY_ID="Comfy-Org/MiniMax-H3"

# H3_WEIGHTS：quant（默认，量化版）| online（官方 BF16 全量，供在线量化）
# 兼容旧变量 FULL_REPO=1
if [ "${FULL_REPO:-0}" = "1" ] && [ -z "${H3_WEIGHTS:-}" ]; then
  H3_WEIGHTS="online"
fi
H3_WEIGHTS="${H3_WEIGHTS:-quant}"
case "$H3_WEIGHTS" in
  quant|online) ;;
  *) die "H3_WEIGHTS 只能是 quant（默认量化版）或 online（官方全量 BF16），当前：$H3_WEIGHTS" ;;
esac

# 量化变体选择
H3_DIT_QUANT="${H3_DIT_QUANT:-pruned_int8_convrot}"
H3_TE_QUANT="${H3_TE_QUANT:-nvfp4_awq}"
case "$H3_DIT_QUANT" in
  pruned_int8_convrot) DIT_FILE="diffusion_models/minimax_h3_ref2va_pruned_int8_convrot.safetensors" ;;
  int8_convrot)        DIT_FILE="diffusion_models/minimax_h3_ref2va_int8_convrot.safetensors" ;;
  w6a8)                DIT_FILE="diffusion_models/minimax_h3_ref2va_pruned_w6a8.safetensors" ;;
  *) die "不支持的 H3_DIT_QUANT=$H3_DIT_QUANT（可选：pruned_int8_convrot | int8_convrot | w6a8）" ;;
esac
case "$H3_TE_QUANT" in
  nvfp4_awq)     TE_FILE="text_encoders/qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors" ;;
  int8_convrot)  TE_FILE="text_encoders/qwen3vl_32b_minimax_h3_int8_convrot.safetensors" ;;
  *) die "不支持的 H3_TE_QUANT=$H3_TE_QUANT（可选：nvfp4_awq | int8_convrot）" ;;
esac
VIDEO_VAE_FILE="vae/minimax_h3_video_vae_fp16.safetensors"
AUDIO_VAE_FILE="vae/minimax_h3_audio_vae_fp32.safetensors"

# 默认目录：官方精简树 /workspace/models/MiniMax-H3，量化权重 /workspace/models/Comfy-MiniMax-H3
MODEL_DIR="${1:-${MODEL_DIR:-$PROJECT_DIR/models/MiniMax-H3}}"
COMFY_MODEL_DIR="${COMFY_MODEL_DIR:-$PROJECT_DIR/models/Comfy-MiniMax-H3}"

# 缓存目录统一放在工作目录下
export MODELSCOPE_CACHE="${MODELSCOPE_CACHE:-$PROJECT_DIR/.cache/modelscope}"
mkdir -p "$MODEL_DIR" "$COMFY_MODEL_DIR" "$(dirname "$MODELSCOPE_CACHE")"

# -----------------------------------------------------------------------------
# 1. 定位 modelscope CLI（优先使用 /workspace/.venv 虚拟环境中的）
# -----------------------------------------------------------------------------
VENV="${SGLANG_VENV:-$PROJECT_DIR/.venv}"
MS_BIN=""
if [ -x "$VENV/bin/modelscope" ]; then
  MS_BIN="$VENV/bin/modelscope"
elif command -v modelscope >/dev/null 2>&1; then
  MS_BIN="$(command -v modelscope)"
else
  die "未找到 modelscope CLI。请先执行安装：bash $SCRIPT_DIR/install-sglang.sh"
fi
log "使用 ModelScope CLI：$MS_BIN ($("$MS_BIN" --version 2>/dev/null || echo version unknown))"

# 当前两个仓库在 ModelScope 均为公开，通常不需要 token；保留登录入口
if [ -n "${MODELSCOPE_API_TOKEN:-}" ]; then
  log "检测到 MODELSCOPE_API_TOKEN，执行 modelscope login ..."
  "$MS_BIN" login --token "$MODELSCOPE_API_TOKEN"
fi

WORKERS_ARGS=()
if [ -n "${MAX_WORKERS:-}" ]; then
  WORKERS_ARGS=(--max-workers "$MAX_WORKERS")
fi

# -----------------------------------------------------------------------------
# 2. 磁盘空间检查
# -----------------------------------------------------------------------------
AVAIL_GB="$(df -Pk "$MODEL_DIR" | awk 'NR==2 {printf "%d", $4/1024/1024}')"
if [ "$H3_WEIGHTS" = "quant" ]; then
  NEED_GB=60
else
  NEED_GB=160
fi
if [ "$AVAIL_GB" -lt "$NEED_GB" ] 2>/dev/null; then
  warn "目标分区可用空间约 ${AVAIL_GB}GB：当前方案（$H3_WEIGHTS）建议预留 ${NEED_GB}GB 以上。"
fi

# -----------------------------------------------------------------------------
# 3. 下载官方仓库内容（quant=精简配置树；online=Ref2VA 全量）
#    modelscope CLI 的 --include 接受多个空格分隔 glob（nargs='+'）。
# -----------------------------------------------------------------------------
if [ "$H3_WEIGHTS" = "online" ]; then
  log "下载官方 BF16 全量 Ref2VA：${OFFICIAL_ID}（model_index.json + Ref2VA/*，约 144GB）..."
  "$MS_BIN" download \
    --model "$OFFICIAL_ID" \
    --local_dir "$MODEL_DIR" \
    --include "model_index.json" "Ref2VA/*" \
    "${WORKERS_ARGS[@]}"
else
  log "下载官方精简配置树：${OFFICIAL_ID}（仅配置/代码，不含权重分片，约几十 MB）..."
  "$MS_BIN" download \
    --model "$OFFICIAL_ID" \
    --local_dir "$MODEL_DIR" \
    --include \
      "model_index.json" \
      "Ref2VA/model_index.json" \
      "Ref2VA/processor/*" \
      "Ref2VA/tokenizer/*" \
      "Ref2VA/transformer/*.json" \
      "Ref2VA/text_encoder/*.json" \
      "Ref2VA/text_encoder/*.txt" \
      "Ref2VA/audio_vae/*.py" \
      "Ref2VA/audio_vae/*.json" \
      "Ref2VA/audio_vae/*.yaml" \
      "Ref2VA/video_vae/*.py" \
      "Ref2VA/video_vae/*.json" \
      "Ref2VA/video_vae/source/*.json" \
    "${WORKERS_ARGS[@]}"

  [ -f "$MODEL_DIR/model_index.json" ] || die "精简配置树下载不完整：缺少 model_index.json"
  [ -f "$MODEL_DIR/Ref2VA/model_index.json" ] || die "精简配置树下载不完整：缺少 Ref2VA/model_index.json"

  # ---------------------------------------------------------------------------
  # 4. 下载 Comfy-Org 量化权重（A10 24GB 组合）
  # ---------------------------------------------------------------------------
  log "下载量化权重：${COMFY_ID}"
  log "  DiT ：$DIT_FILE（$H3_DIT_QUANT）"
  log "  文本编码器：$TE_FILE（$H3_TE_QUANT）"
  log "  视频 VAE：$VIDEO_VAE_FILE"
  log "  音频 VAE：$AUDIO_VAE_FILE"
  "$MS_BIN" download \
    --model "$COMFY_ID" \
    --local_dir "$COMFY_MODEL_DIR" \
    --include "$DIT_FILE" "$TE_FILE" "$VIDEO_VAE_FILE" "$AUDIO_VAE_FILE" \
    "${WORKERS_ARGS[@]}"

  for f in "$DIT_FILE" "$TE_FILE" "$VIDEO_VAE_FILE" "$AUDIO_VAE_FILE"; do
    [ -f "$COMFY_MODEL_DIR/$f" ] || die "量化权重下载不完整：缺少 $COMFY_MODEL_DIR/$f"
  done
fi

# -----------------------------------------------------------------------------
# 5. 结果提示
# -----------------------------------------------------------------------------
BASE_SIZE="$(du -sh "$MODEL_DIR" 2>/dev/null | cut -f1 || echo '?')"
cat <<EOF

========================== 下载完成 ==========================
方案：$H3_WEIGHTS
官方目录：${MODEL_DIR}（占用 ${BASE_SIZE}）
EOF
if [ "$H3_WEIGHTS" = "quant" ]; then
  COMFY_SIZE="$(du -sh "$COMFY_MODEL_DIR" 2>/dev/null | cut -f1 || echo '?')"
  cat <<EOF
量化目录：${COMFY_MODEL_DIR}（占用 ${COMFY_SIZE}）
DiT 量化：${H3_DIT_QUANT}    文本编码器：${H3_TE_QUANT}

A10 / RTX 3090 / 4090 等 24GB 单卡通过 layerwise offload 运行：
  - 显存峰值约 18GB（官方同档位 RTX 4090 实测）
  - 建议主机内存 ≥64GB（32GB 亦可，但需 NVMe SSD 且更慢）
  - 768p 5 秒片段生成耗时约数分钟量级
启动：bash ${SCRIPT_DIR}/serve.sh
EOF
else
  cat <<EOF
已下载官方 BF16 全量 Ref2VA。serve.sh 将以在线量化 convrot_int8 +
layerwise offload 启动（即官方 1×RTX 4090 24GB 实测路径）。
启动：bash ${SCRIPT_DIR}/serve.sh
EOF
fi
printf '==============================================================\n'
