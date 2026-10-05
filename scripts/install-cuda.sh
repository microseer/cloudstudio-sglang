#!/usr/bin/env bash
# =============================================================================
# CloudStudio × MiniMax-H3 Ref2VA 应用模板 - CUDA Toolkit / 媒体依赖 一键安装脚本
#
# 功能：
#   1. 自动检测 NVIDIA 驱动版本，按驱动能力选择合适的 CUDA Toolkit 版本
#   2. 通过 NVIDIA 官方 apt 仓库安装（Ubuntu 20.04 / 22.04 / 24.04）
#   3. 同时安装 MiniMax-H3 必需的 ffmpeg / ffprobe（参考视频准备与 MP4 封装）
#   4. 自动配置 CUDA_HOME / PATH / LD_LIBRARY_PATH 到 ~/.bashrc
#   5. 幂等：已安装 nvcc 时仍会确保 ffmpeg 存在；CUDA_FORCE=1 可强制重装 CUDA
#   6. 自动修复基础镜像中损坏的第三方 apt 源（如 URL 带反引号/密钥缺失的 github-cli 源）
#
# 可用环境变量：
#   CUDA_VERSION=12-4   强制指定 CUDA 版本（如 13-0 / 12-9 / 12-8 / 12-6 / 12-4）
#   CUDA_FORCE=1        即使已存在 nvcc 也强制重新安装 CUDA Toolkit
# =============================================================================
set -euo pipefail

log()  { printf '\033[1;32m[cuda]\033[0m %s\n' "$*"; }
warn() { printf '\033[1;33m[cuda]\033[0m %s\n' "$*" >&2; }
die()  { printf '\033[1;31m[cuda]\033[0m %s\n' "$*" >&2; exit 1; }

SUDO=""
if [ "$(id -u)" -ne 0 ]; then
  SUDO="sudo"
fi

# 本脚本的绝对路径（CloudStudio 中为 /workspace/scripts/install-cuda.sh）
SELF="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/$(basename "${BASH_SOURCE[0]}")"

# -----------------------------------------------------------------------------
# apt 辅助函数：修复坏源 / 容错更新 / 确保媒体依赖
# -----------------------------------------------------------------------------
# 修复基础镜像中损坏的 github-cli apt 源（URL 带反引号、GPG 密钥 NO_PUBKEY 等）。
# 只处理包含 cli.github.com 的源文件；密钥刷新失败（网络受限）则禁用该源——
# 本模板不依赖 gh CLI，禁用不影响任何功能。
repair_broken_apt_sources() {
  local f
  shopt -s nullglob
  for f in /etc/apt/sources.list /etc/apt/sources.list.d/*; do
    [ -f "$f" ] || continue
    grep -q 'cli\.github\.com' "$f" 2>/dev/null || continue
    if [ "${f##*.}" = "list" ] && grep -q '`' "$f"; then
      warn "检测到 $f 的 URL 含反引号（基础镜像写入错误），已重写该源。"
      printf 'deb [arch=amd64 signed-by=/usr/share/keyrings/githubcli-archive-keyring.gpg] https://cli.github.com/packages stable main\n' \
        | $SUDO tee "$f" >/dev/null
    fi
    if command -v wget >/dev/null 2>&1 \
      && wget -q --timeout=15 -O /tmp/.githubcli-keyring.gpg \
        https://cli.github.com/packages/githubcli-archive-keyring.gpg \
      && $SUDO tee /usr/share/keyrings/githubcli-archive-keyring.gpg \
        < /tmp/.githubcli-keyring.gpg >/dev/null; then
      rm -f /tmp/.githubcli-keyring.gpg
      log "已刷新 github-cli 源 GPG 密钥。"
    else
      warn "github-cli 源密钥不可用（网络受限或已失效），禁用该源：$f"
      $SUDO mv -f "$f" "${f}.disabled-by-h3" 2>/dev/null || true
    fi
  done
  shopt -u nullglob
}

# apt-get update 容错：个别第三方源失败时索引仍会部分刷新，不应阻断整体流程
apt_update() {
  $SUDO apt-get update -qq || warn "apt-get update 部分软件源失败（已忽略），继续使用可用索引。"
}

# 幂等确保 ffmpeg / ffprobe 存在；失败仅告警（只影响生成后 ffprobe 探测，不影响服务）
ensure_ffmpeg() {
  if command -v ffmpeg >/dev/null 2>&1 && command -v ffprobe >/dev/null 2>&1; then
    return 0
  fi
  log "安装 MiniMax-H3 媒体依赖 ffmpeg / ffprobe ..."
  repair_broken_apt_sources
  apt_update
  if $SUDO apt-get install -y -qq --no-install-recommends ffmpeg; then
    log "ffmpeg / ffprobe 安装完成。"
  else
    warn "ffmpeg 安装失败（不影响模型服务，仅影响生成后 ffprobe 探测）。"
    warn "可稍后手动执行：sudo apt-get update && sudo apt-get install -y ffmpeg"
  fi
  command -v ffmpeg >/dev/null 2>&1 && command -v ffprobe >/dev/null 2>&1
}

# -----------------------------------------------------------------------------
# 1. 已安装 nvcc 时跳过 CUDA 安装，但仍确保 ffmpeg 媒体依赖存在
# -----------------------------------------------------------------------------
if command -v nvcc >/dev/null 2>&1 && [ "${CUDA_FORCE:-0}" != "1" ]; then
  log "检测到已安装 $(nvcc --version | tail -1 | sed 's/^ *//')，跳过 CUDA 安装。"
  ensure_ffmpeg || true
  log "如需强制重装 CUDA，请执行：CUDA_FORCE=1 bash $SELF"
  exit 0
fi

# -----------------------------------------------------------------------------
# 2. 检测 GPU 驱动
# -----------------------------------------------------------------------------
if ! command -v nvidia-smi >/dev/null 2>&1; then
  warn "未检测到 nvidia-smi：当前环境没有挂载 NVIDIA 驱动（可能是 CPU 算力规格）。"
  warn "CUDA Toolkit 安装已跳过；仍继续安装 ffmpeg / ffprobe 媒体依赖。"
  warn "请在 CloudStudio 中切换到 GPU 算力后重新运行：bash $SELF"
  ensure_ffmpeg || true
  exit 0
fi

DRIVER_FULL="$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -n1 | tr -d '[:space:]')"
DRIVER_MAJOR="$(printf '%s' "$DRIVER_FULL" | cut -d. -f1)"
log "检测到 NVIDIA 驱动版本：${DRIVER_FULL}"

# -----------------------------------------------------------------------------
# 3. 选择 CUDA 版本
#    驱动与 CUDA 最低版本对应关系（Linux）：
#      CUDA 13.0 -> 580.x    CUDA 12.9 -> 575.x
#      CUDA 12.8 -> 570.x    CUDA 12.6 -> 560.x
#      CUDA 12.5 -> 555.x    CUDA 12.4 -> 550.x
#      CUDA 12.3 -> 545.x    CUDA 12.2 -> 535.x
# -----------------------------------------------------------------------------
CUDA_VER="${CUDA_VERSION:-}"
if [ -z "$CUDA_VER" ]; then
  case "$DRIVER_MAJOR" in
    58[0-9]|59[0-9]|[6-9][0-9][0-9]) CUDA_VER="13-0" ;;
    57[5-9])                          CUDA_VER="12-9" ;;
    57[0-4])                          CUDA_VER="12-8" ;;
    56[0-9])                          CUDA_VER="12-6" ;;
    55[5-9])                          CUDA_VER="12-5" ;;
    55[0-4])                          CUDA_VER="12-4" ;;
    54[5-9])                          CUDA_VER="12-3" ;;
    53[5-9])                          CUDA_VER="12-2" ;;
    *)
      die "驱动版本 ${DRIVER_FULL} 过旧（建议 >= 535）。
       请先在 CloudStudio 切换到更新的 GPU 镜像/算力规格，或手动安装旧版 CUDA：
       https://developer.nvidia.com/cuda-toolkit-archive"
      ;;
  esac
fi

# Ubuntu 24.04 的官方 CUDA 仓库最低提供 12.4，旧驱动自动上调到 12-4
if [ -r /etc/os-release ]; then
  # shellcheck disable=SC1091
  . /etc/os-release
  if [ "${ID}${VERSION_ID/./}" = "ubuntu2404" ]; then
    case "$CUDA_VER" in
      12-2|12-3)
        warn "Ubuntu 24.04 不提供 cuda-toolkit-${CUDA_VER}，自动改用 12-4。"
        CUDA_VER="12-4"
        ;;
    esac
  fi
fi
log "将安装 CUDA Toolkit ${CUDA_VER}（可用 CUDA_VERSION=12-4 等覆盖此选择）"

# -----------------------------------------------------------------------------
# 4. 识别系统发行版
# -----------------------------------------------------------------------------
[ -r /etc/os-release ] || die "无法读取 /etc/os-release，本脚本仅支持 Ubuntu 20.04/22.04/24.04。"
# shellcheck disable=SC1091
. /etc/os-release
REPO_DIR="${ID}${VERSION_ID/./}"
case "$REPO_DIR" in
  ubuntu2004|ubuntu2204|ubuntu2404) ;;
  *)
    die "自动安装仅支持 Ubuntu 20.04/22.04/24.04，当前为：${PRETTY_NAME:-$REPO_DIR}
       请参考 https://developer.nvidia.com/cuda-downloads 手动安装。"
    ;;
esac

# -----------------------------------------------------------------------------
# 5. 添加 NVIDIA 官方 apt 仓库并安装
# -----------------------------------------------------------------------------
log "安装基础依赖（wget / gnupg / ca-certificates）与媒体依赖（ffmpeg）..."
repair_broken_apt_sources
apt_update
$SUDO apt-get install -y -qq --no-install-recommends wget gnupg ca-certificates
ensure_ffmpeg || true

KEYRING_DEB="cuda-keyring_1.1-1_all.deb"
KEYRING_URL="https://developer.download.nvidia.com/compute/cuda/repos/${REPO_DIR}/x86_64/${KEYRING_DEB}"
log "下载 NVIDIA CUDA 仓库密钥：${KEYRING_URL}"
wget -q --progress=dot:giga "$KEYRING_URL" -O "/tmp/${KEYRING_DEB}"
$SUDO dpkg -i "/tmp/${KEYRING_DEB}" >/dev/null

log "通过 apt 安装 cuda-toolkit-${CUDA_VER}（体积约 3-5 GB，耗时较长，请耐心等待）..."
apt_update
$SUDO apt-get install -y --no-install-recommends "cuda-toolkit-${CUDA_VER}"

# -----------------------------------------------------------------------------
# 6. 配置环境变量到 ~/.bashrc（CloudStudio 新建终端会自动加载）
# -----------------------------------------------------------------------------
CUDA_HOME_VAL="/usr/local/cuda-${CUDA_VER/-/.}"
[ -d "$CUDA_HOME_VAL" ] || CUDA_HOME_VAL="/usr/local/cuda"
MARK_BEGIN="# >>> cuda env (cloudstudio-sglang) >>>"
MARK_END="# <<< cuda env (cloudstudio-sglang) <<<"

if ! grep -qF "$MARK_BEGIN" "$HOME/.bashrc" 2>/dev/null; then
  cat >> "$HOME/.bashrc" <<EOF

${MARK_BEGIN}
export CUDA_HOME=${CUDA_HOME_VAL}
export PATH=\$CUDA_HOME/bin:\$PATH
export LD_LIBRARY_PATH=\$CUDA_HOME/lib64:\$LD_LIBRARY_PATH
${MARK_END}
EOF
  log "已将 CUDA 环境变量写入 ~/.bashrc"
else
  log "~/.bashrc 中已存在 CUDA 环境变量配置，跳过写入"
fi

# shellcheck disable=SC1090
export CUDA_HOME="$CUDA_HOME_VAL"
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME_VAL/lib64:${LD_LIBRARY_PATH:-}"

# -----------------------------------------------------------------------------
# 7. 校验
# -----------------------------------------------------------------------------
log "安装完成：$(nvcc --version | tail -1 | sed 's/^ *//')"
log "当前终端若提示找不到 nvcc，请执行：source ~/.bashrc（或新开一个终端）"
