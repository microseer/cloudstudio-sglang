#!/usr/bin/env bash
# =============================================================================
# CloudStudio × MiniMax-H3 Ref2VA 应用模板 - SGLang(Diffusion) 一键安装脚本
#
# 功能：
#   1. 使用 uv 创建独立 Python 3.10+ 虚拟环境（默认 <项目根>/.venv）
#   2. 安装官方 MiniMax-H3 要求的 SGLang Diffusion 版本：
#        uv pip install "sglang[diffusion]" --prerelease=allow
#      参考：https://docs.sglang.io/cookbook/diffusion/MiniMax/MiniMax-H3
#   3. 同时安装 ModelScope CLI，用于国内高速下载 MiniMax/MiniMax-H3
#   4. 安装 comfy-kitchen：ConvRot INT8 量化内核后端。
#      SGLang 自带的 JIT ConvRot 算子仅覆盖 CC 9.0/10.0/12.0/12.1
#      （Hopper/Blackwell）；A10 / RTX 3090 / 4090 等卡（CC 8.6/8.9）
#      自动走 comfy-kitchen 的 int8_linear（Turing+ 支持）。
#   5. 幂等：可重复执行；安装完成后自动验证 torch.cuda 是否可用
#
# 路径约定（CloudStudio 中项目根目录即 /workspace）：
#   虚拟环境：/workspace/.venv
#   模型目录：/workspace/models
#   缓存目录：/workspace/.cache
#
# 可用环境变量：
#   SGLANG_VENV=/path/venv          自定义虚拟环境目录（默认 /workspace/.venv）
#   SGLANG_SPEC='sglang[diffusion]==x.y.z'
#                                   强制指定 SGLang 安装规格
#   SGLANG_SKIP_KITCHEN=1           不安装 comfy-kitchen（仅 Hopper/Blackwell
#                                   使用自带 JIT 算子时可跳过）
#   USE_CN_MIRROR=0                 关闭清华 PyPI 镜像，使用官方源
#
# 说明：媒体处理依赖 ffmpeg/ffprobe（H3 参考视频准备与 MP4 封装需要），
#       由 scripts/install-cuda.sh 通过 apt 一并安装；也可自行：
#       sudo apt-get install -y ffmpeg
# =============================================================================
set -euo pipefail

log()  { printf '\033[1;32m[sglang]\033[0m %s\n' "$*"; }
warn() { printf '\033[1;33m[sglang]\033[0m %s\n' "$*" >&2; }
die()  { printf '\033[1;31m[sglang]\033[0m %s\n' "$*" >&2; exit 1; }

# 项目根目录 = 本脚本所在目录的上一级（CloudStudio 中解析为 /workspace）
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
VENV="${SGLANG_VENV:-$PROJECT_DIR/.venv}"

# ModelScope / HF 缓存统一放到工作目录下
export MODELSCOPE_CACHE="${MODELSCOPE_CACHE:-$PROJECT_DIR/.cache/modelscope}"
mkdir -p "$PROJECT_DIR/models" "$(dirname "$MODELSCOPE_CACHE")"

INDEX_ARGS=()
if [ "${USE_CN_MIRROR:-1}" = "1" ]; then
  log "使用清华 PyPI 镜像（设置 USE_CN_MIRROR=0 可切换官方源）"
  INDEX_ARGS=(--index-url "https://pypi.tuna.tsinghua.edu.cn/simple")
else
  log "使用 PyPI 官方源"
fi

# -----------------------------------------------------------------------------
# 1. 安装 uv
# -----------------------------------------------------------------------------
if ! command -v uv >/dev/null 2>&1; then
  log "安装 uv（高速 Python 包管理器）..."
  python3 -m pip install -q --upgrade pip "${INDEX_ARGS[@]}"
  python3 -m pip install -q uv "${INDEX_ARGS[@]}"
fi

# -----------------------------------------------------------------------------
# 2. 创建虚拟环境（系统 Python >= 3.10 则直接使用，否则由 uv 自动下载 3.12）
# -----------------------------------------------------------------------------
PY_OK=0
if command -v python3 >/dev/null 2>&1; then
  if python3 -c 'import sys; sys.exit(0 if sys.version_info[:2] >= (3, 10) else 1)'; then
    PY_OK=1
  fi
fi

if [ ! -x "$VENV/bin/python" ]; then
  if [ "$PY_OK" = "1" ]; then
    log "使用系统 $(python3 -V 2>&1) 创建虚拟环境：$VENV"
    uv venv -q "$VENV"
  else
    warn "系统 Python 版本低于 3.10，将由 uv 自动下载 Python 3.12 ..."
    uv venv -q --python 3.12 "$VENV"
  fi
else
  log "虚拟环境已存在：$VENV"
fi

# -----------------------------------------------------------------------------
# 3. 安装规格与驱动检查
#    MiniMax-H3 是新模型，需要最新 SGLang Diffusion（>= 其合入 H3 支持的版本），
#    官方安装命令即：uv pip install "sglang[diffusion]" --prerelease=allow
#    最新 SGLang 默认走 CUDA 13 / PyTorch 2.14 预编译通道，建议驱动 >= 580；
#    驱动较旧时此处只告警（可用 SGLANG_SPEC 钉选可用版本自行解决）。
# -----------------------------------------------------------------------------
DRIVER_MAJOR=0
if command -v nvidia-smi >/dev/null 2>&1; then
  DRIVER_MAJOR="$(nvidia-smi --query-gpu=driver_version --format=csv,noheader \
                  | head -n1 | tr -d '[:space:]' | cut -d. -f1)"
  GPU_COUNT="$(nvidia-smi -L 2>/dev/null | wc -l | tr -d ' ')"
else
  GPU_COUNT=0
fi

SGLANG_SPEC="${SGLANG_SPEC:-sglang[diffusion]}"

if [ "$DRIVER_MAJOR" -eq 0 ]; then
  warn "未检测到 NVIDIA GPU：仍会安装 SGLang（PyTorch 自带 CUDA 运行时），"
  warn "但 MiniMax-H3 推理必须在 CloudStudio GPU 算力规格下进行（A10 24GB 单卡即可）。"
elif [ "$DRIVER_MAJOR" -lt 580 ]; then
  warn "检测到 NVIDIA 驱动主版本为 ${DRIVER_MAJOR}（< 580）。"
  warn "最新 SGLang 默认使用 CUDA 13 预编译包，建议切换到驱动 ≥ 580 的 GPU 镜像；"
  warn "A10（Ampere）走 comfy-kitchen 预编译内核，对驱动的要求以 PyTorch 官方兼容矩阵为准；"
  warn "若安装后 torch 报 CUDA 版本不匹配，请更新 GPU 算力规格，或用 SGLANG_SPEC 指定兼容版本。"
fi

if [ "$GPU_COUNT" -gt 0 ]; then
  log "检测到 ${GPU_COUNT} 张 GPU。单张 24GB 卡（如 A10 / RTX 3090 / 4090）"
  log "使用本模板默认的量化权重 + layerwise offload 即可运行 Ref2VA（官方 1×4090 实测）。"
fi

log "安装规格：${SGLANG_SPEC} + modelscope（driver=${DRIVER_MAJOR}, gpus=${GPU_COUNT}）"

# -----------------------------------------------------------------------------
# 4. 安装 SGLang[diffusion] 与 ModelScope CLI
#    --prerelease=allow：SGLang 部分依赖仅发布预编译版本（官方文档要求）
# -----------------------------------------------------------------------------
log "开始安装（PyTorch / SGLang Diffusion 体积较大，首次安装请耐心等待）..."
uv pip install \
  --python "$VENV/bin/python" \
  --prerelease=allow \
  "${INDEX_ARGS[@]}" \
  "$SGLANG_SPEC" modelscope

# -----------------------------------------------------------------------------
# 4b. 安装 comfy-kitchen（A10/3090/4090 等非 Hopper/Blackwell 卡的 INT8 ConvRot 后端）
#     W4A8 量化（H3_DIT_QUANT=w6a8）要求 comfy-kitchen>=0.2.27
# -----------------------------------------------------------------------------
if [ "${SGLANG_SKIP_KITCHEN:-0}" = "1" ]; then
  warn "SGLANG_SKIP_KITCHEN=1：跳过 comfy-kitchen。"
  warn "注意：A10 / RTX 3090 / 4090（CC 8.6/8.9）运行 ConvRot INT8 必须安装它。"
else
  log "安装 comfy-kitchen（A10 等 Ampere/Ada 卡的 ConvRot INT8 内核后端）..."
  uv pip install \
    --python "$VENV/bin/python" \
    --prerelease=allow \
    "${INDEX_ARGS[@]}" \
    "comfy-kitchen${SGLANG_KITCHEN_SPEC:-}"
fi

# -----------------------------------------------------------------------------
# 5. 写入自动激活到 ~/.bashrc
# -----------------------------------------------------------------------------
MARK_BEGIN="# >>> sglang venv (cloudstudio-minimax-h3) >>>"
MARK_END="# <<< sglang venv (cloudstudio-minimax-h3) <<<"
if ! grep -qF "$MARK_BEGIN" "$HOME/.bashrc" 2>/dev/null; then
  cat >> "$HOME/.bashrc" <<EOF

${MARK_BEGIN}
export SGLANG_USE_MODELSCOPE=true
export MODELSCOPE_CACHE=${MODELSCOPE_CACHE}
source ${VENV}/bin/activate
${MARK_END}
EOF
  log "已将虚拟环境自动激活与缓存目录写入 ~/.bashrc（新开终端即自动进入）"
fi

# -----------------------------------------------------------------------------
# 6. 安装校验
# -----------------------------------------------------------------------------
log "校验安装结果 ..."
"$VENV/bin/python" - <<'PY'
import importlib
import sys

info = {}
for mod in ("sglang", "torch", "modelscope"):
    try:
        m = importlib.import_module(mod)
        info[mod] = getattr(m, "__version__", "unknown")
    except Exception as e:  # noqa: BLE001
        print(f"[sglang] [错误] 导入 {mod} 失败: {e}")
        sys.exit(1)

import importlib.metadata

import torch
print(f"[sglang] sglang     : {info['sglang']}（含 diffusion extra）")
print(f"[sglang] torch      : {info['torch']} (built with CUDA {torch.version.cuda})")
print(f"[sglang] modelscope : {info['modelscope']}")
try:
    print(f"[sglang] comfy-kitchen : {importlib.metadata.version('comfy-kitchen')}")
except importlib.metadata.PackageNotFoundError:
    print("[sglang] [警告] 未安装 comfy-kitchen：A10/3090/4090 运行量化权重需要它"
          "（SGLANG_SKIP_KITCHEN=1 可跳过安装）")
if torch.cuda.is_available():
    print(f"[sglang] GPU 可用    : {torch.cuda.get_device_name(0)}")
    print(f"[sglang] GPU 数量    : {torch.cuda.device_count()}")
    major, minor = torch.cuda.get_device_capability(0)
    print(f"[sglang] 计算能力    : sm_{major}{minor}"
          + ("（ConvRot INT8 走 comfy-kitchen）" if (major, minor) not in {(9, 0), (10, 0), (12, 0), (12, 1)} else ""))
else:
    print("[sglang] GPU 暂不可用：请确认 CloudStudio 已切换为 GPU 算力规格。")
PY

cat <<EOF

========================== 安装完成 ==========================
新开终端会自动激活虚拟环境；当前终端可执行：
    source ${VENV}/bin/activate

下一步（工作目录为 ${PROJECT_DIR}，CloudStudio 中即 /workspace）：
  1) 下载模型：  bash ${SCRIPT_DIR}/download-model.sh
                 （默认量化版约 43GB，A10 等 24GB 单卡可跑；
                  H3_WEIGHTS=online 可改下载官方 BF16 全量约 144GB）
  2) 启动服务：  bash ${SCRIPT_DIR}/serve.sh
  3) 发起生成：  bash ${SCRIPT_DIR}/request-ref2va.sh
==============================================================
EOF
