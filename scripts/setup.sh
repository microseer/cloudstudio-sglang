#!/usr/bin/env bash
# =============================================================================
# CloudStudio × MiniMax-H3 Ref2VA 应用模板 - 一键全量初始化（新空间只需执行这一条）
#
# 面向 A10（24GB，Ampere）等单卡环境，默认下载量化版权重（约 43GB），
# 通过 ConvRot INT8 + NVFP4-AWQ + layerwise offload 在单张 24GB 卡运行。
#
# 依次完成：
#   1. 安装 CloudStudio 运行配置 .vscode/preview.yml（用于点击「运行」启动服务）
#   2. CUDA Toolkit + ffmpeg 一键安装（scripts/install-cuda.sh）
#   3. SGLang[diffusion] + ModelScope CLI + comfy-kitchen（scripts/install-sglang.sh）
#   4. 下载 Ref2VA 量化权重（scripts/download-model.sh，默认约 43GB）
#
# 用法：
#   bash /workspace/scripts/setup.sh                      # 全部执行（默认量化版）
#   H3_WEIGHTS=online bash /workspace/scripts/setup.sh    # 改下官方 BF16 全量（约144GB）
#   SETUP_SKIP_MODEL=1 bash /workspace/scripts/setup.sh   # 只装环境，不下载模型
#
# 可用环境变量（透传给各子脚本）：
#   SETUP_SKIP_MODEL=1  跳过模型下载
#   H3_WEIGHTS=quant    quant（默认，约43GB 量化版）| online（约144GB 官方全量）
#   H3_DIT_QUANT / H3_TE_QUANT  量化变体选择（见 download-model.sh 头部说明）
#   MAX_WORKERS=8       ModelScope 下载并发数
#   CUDA_VERSION        强制 CUDA 版本（如 12-4）
#   SGLANG_SPEC         强制 SGLang 安装规格（如 'sglang[diffusion]==x.y.z'）
#   USE_CN_MIRROR=0     改用 PyPI 官方源
# =============================================================================
set -euo pipefail

log()  { printf '\n\033[1;35m[setup]\033[0m %s\n' "$*"; }
die()  { printf '\033[1;31m[setup]\033[0m %s\n' "$*" >&2; exit 1; }

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

H3_WEIGHTS="${H3_WEIGHTS:-quant}"
export H3_WEIGHTS

log "项目根目录：${PROJECT_DIR}（CloudStudio 中应为 /workspace）｜权重方案：${H3_WEIGHTS}"

# -----------------------------------------------------------------------------
# 1. 安装 CloudStudio 运行配置 .vscode/preview.yml
#    （仓库中以 config/preview.yml 形式保存，setup 时复制到 CloudStudio 约定位置）
# -----------------------------------------------------------------------------
log "安装运行配置 .vscode/preview.yml ..."
mkdir -p "$PROJECT_DIR/.vscode"
if [ -f "$PROJECT_DIR/config/preview.yml" ]; then
  cp -n "$PROJECT_DIR/config/preview.yml" "$PROJECT_DIR/.vscode/preview.yml" 2>/dev/null \
    || cp "$PROJECT_DIR/config/preview.yml" "$PROJECT_DIR/.vscode/preview.yml"
  log "已写入 $PROJECT_DIR/.vscode/preview.yml（应用模式下点击「运行」即可启动 MiniMax-H3 Ref2VA）"
else
  die "缺少 $PROJECT_DIR/config/preview.yml，请确认仓库完整。"
fi

# -----------------------------------------------------------------------------
# 2. CUDA Toolkit + ffmpeg（无 GPU / 已安装时脚本内部会自动跳过 CUDA）
# -----------------------------------------------------------------------------
log "步骤 1/3：安装 CUDA Toolkit 与 ffmpeg 媒体依赖 ..."
bash "$SCRIPT_DIR/install-cuda.sh"

# -----------------------------------------------------------------------------
# 3. SGLang[diffusion] + ModelScope CLI + comfy-kitchen
# -----------------------------------------------------------------------------
log "步骤 2/3：安装 SGLang[diffusion]、ModelScope CLI 与 comfy-kitchen ..."
bash "$SCRIPT_DIR/install-sglang.sh"

# -----------------------------------------------------------------------------
# 4. 下载模型
# -----------------------------------------------------------------------------
if [ "${SETUP_SKIP_MODEL:-0}" = "1" ]; then
  log "步骤 3/3：SETUP_SKIP_MODEL=1，已跳过模型下载。"
else
  if [ "$H3_WEIGHTS" = "quant" ]; then
    log "步骤 3/3：下载 Ref2VA 量化权重（Comfy-Org INT8/NVFP4-AWQ，约 43GB）..."
  else
    log "步骤 3/3：下载官方 BF16 全量 Ref2VA（约 144GB，供在线量化）..."
  fi
  bash "$SCRIPT_DIR/download-model.sh"
fi

cat <<EOF

====================== 一键初始化完成 ======================
硬件提醒：默认量化方案面向单张 24GB 卡（A10 / RTX 3090 / 4090），
显存峰值约 18GB；建议主机内存 ≥64GB，至少 32GB + NVMe SSD。
请确认 CloudStudio 已切换到 A10（或同级别）GPU 算力规格。

启动 SGLang 服务（任选其一）：
  1. 点击 CloudStudio 顶部「运行」（读取 .vscode/preview.yml，端口 30011）
  2. 终端执行：bash ${SCRIPT_DIR}/serve.sh

发起一次 Ref2VA 生成（提交→轮询→下载 MP4）：
  bash ${SCRIPT_DIR}/request-ref2va.sh

环境就绪后，可在 IDE 中通过「文件 → 发布自定义模板」把当前空间
保存为团队模板并生成分享链接/徽章。
============================================================
EOF
