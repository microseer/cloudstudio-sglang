#!/usr/bin/env bash
# =============================================================================
# CloudStudio × MiniMax-H3 Ref2VA 应用模板 - Ref2VA 生成请求一键脚本
#
# 调用 SGLang 的异步视频生成接口（OpenAI 兼容风格）：
#   POST /v1/videos                 提交任务，返回 {"id": ...}
#   GET  /v1/videos/{id}            查询状态（queued/processing/completed/failed）
#   GET  /v1/videos/{id}/content    下载生成的 MP4（H.264 24fps + 立体声 AAC）
#
# 用法：
#   bash /workspace/scripts/request-ref2va.sh
#       # 使用内置示例（官方可复现的公开参考视频 + 简短提示词）
#
#   REF_IMAGE_URL='https://example.com/a.jpg' \
#   PROMPT='让照片中的人物微笑着挥手打招呼，保持场景与光线一致' \
#   bash /workspace/scripts/request-ref2va.sh
#
#   REF_VIDEO_URL='https://.../x.mp4' REF_AUDIO_URL='https://.../y.mp3' \
#   DURATION=5 SHORT_EDGE=768 ASPECT_RATIO=auto SEED=42 \
#   bash /workspace/scripts/request-ref2va.sh
#
# 参考输入约束（来自 MiniMax-H3 模型卡）：
#   图像 ≤ 9 张（≤30MB，jpg/png/webp/heic）；视频 ≤ 3 段，每段 2–15s 且总计 ≤15s
#   （≤50MB，h264/h265）；音频 ≤ 3 段（≤15MB，wav/mp3）且必须与图像或视频搭配，
#   不能作为唯一输入；混合输入总计 ≤ 12 个文件；整体请求体 < 64MB，大文件优先用 URL。
#
# 可用环境变量：
#   SGLANG_PORT=30011     SGLANG_HOST=127.0.0.1
#   OUTPUT=自定义输出路径  DURATION=4..15（秒）  SHORT_EDGE=768
#   ASPECT_RATIO=auto     SEED=0    MAX_WAIT=1800（轮询超时秒）  POLL_INTERVAL=10
# =============================================================================
set -euo pipefail

log()  { printf '\033[1;36m[ref2va]\033[0m %s\n' "$*"; }
warn() { printf '\033[1;33m[ref2va]\033[0m %s\n' "$*" >&2; }
die()  { printf '\033[1;31m[ref2va]\033[0m %s\n' "$*" >&2; exit 1; }

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
VENV="${SGLANG_VENV:-$PROJECT_DIR/.venv}"
PY_BIN="$VENV/bin/python"
[ -x "$PY_BIN" ] || PY_BIN="python3"

HOST="${SGLANG_HOST:-127.0.0.1}"
PORT="${SGLANG_PORT:-30011}"
BASE="http://${HOST}:${PORT}"
DURATION="${DURATION:-5}"
SHORT_EDGE="${SHORT_EDGE:-768}"
ASPECT_RATIO="${ASPECT_RATIO:-auto}"
SEED="${SEED:-0}"
MAX_WAIT="${MAX_WAIT:-1800}"
POLL_INTERVAL="${POLL_INTERVAL:-10}"
OUTPUT="${OUTPUT:-$PROJECT_DIR/outputs/ref2va-$(date +%Y%m%d-%H%M%S).mp4}"

# 官方可复现示例的公开参考视频（video editing 类素材）
REF_VIDEO_URL="${REF_VIDEO_URL:-https://cdn.hailuoai.com/prod/hailuo_demo/testsets/h3_promo_eval_ref2va/gallery/sr_v2p26_trio_seed42_20260724/inputs/297573323635_00_%E8%A7%86%E9%A2%911_YnyRbxEwio_video_20260525_163755_1927e9d3.mp4}"
REF_IMAGE_URL="${REF_IMAGE_URL:-}"
REF_AUDIO_URL="${REF_AUDIO_URL:-}"
PROMPT="${PROMPT:-保持原视频的人物、场景与运镜，让人物面向镜头微笑着挥手打招呼，并生成与画面同步的环境音。}"

# -----------------------------------------------------------------------------
# 0. 校验参考输入：Ref2VA 至少需要一个图像或视频引用；音频不能单独存在
#    （本脚本通过 URL 提交单条引用；多引用请按 README 直接构造 JSON 请求）
# -----------------------------------------------------------------------------
[ -n "$REF_VIDEO_URL" ] || [ -n "$REF_IMAGE_URL" ] \
  || die "Ref2VA 至少需要一个参考输入：请设置 REF_IMAGE_URL 或 REF_VIDEO_URL。"
if [ -n "$REF_AUDIO_URL" ] && [ -z "$REF_VIDEO_URL" ] && [ -z "$REF_IMAGE_URL" ]; then
  die "音频不能作为唯一参考输入，请同时提供 REF_IMAGE_URL 或 REF_VIDEO_URL。"
fi
mkdir -p "$(dirname "$OUTPUT")"

# -----------------------------------------------------------------------------
# 1. 健康检查
# -----------------------------------------------------------------------------
log "检查服务 ${BASE}/health ..."
if ! curl -fsS --max-time 5 "${BASE}/health" >/dev/null 2>&1; then
  die "服务未就绪（${BASE}/health 不可达）。请先执行：bash $SCRIPT_DIR/serve.sh"
fi

# -----------------------------------------------------------------------------
# 2. 构造请求 JSON（用 Python 做安全转义，避免提示词中的引号/换行破坏 JSON）
# -----------------------------------------------------------------------------
REQ_BODY="$(
  REF_VIDEO_URL="$REF_VIDEO_URL" REF_IMAGE_URL="$REF_IMAGE_URL" REF_AUDIO_URL="$REF_AUDIO_URL" \
  PROMPT="$PROMPT" DURATION="$DURATION" SHORT_EDGE="$SHORT_EDGE" \
  ASPECT_RATIO="$ASPECT_RATIO" SEED="$SEED" \
  "$PY_BIN" - <<'PY'
import json, os

conditions = []
for typ, env in (("video", "REF_VIDEO_URL"),
                 ("image", "REF_IMAGE_URL"),
                 ("audio", "REF_AUDIO_URL")):
    uri = os.environ.get(env, "").strip()
    if uri:
        conditions.append({"type": typ, "uri": uri, "role": "reference"})

body = {
    "task": "ref2va",
    "prompt": os.environ["PROMPT"],
    "conditions": conditions,
    "target": {
        "short_edge": int(os.environ["SHORT_EDGE"]),
        "aspect_ratio": os.environ["ASPECT_RATIO"],
        "duration_seconds": float(os.environ["DURATION"]),
    },
    "seed": int(os.environ["SEED"]),
}
print(json.dumps(body, ensure_ascii=False))
PY
)"
log "提交 Ref2VA 任务（duration=${DURATION}s, ${SHORT_EDGE}p, ratio=${ASPECT_RATIO}, seed=${SEED}） ..."

RESP_FILE="$(mktemp)"
trap 'rm -f "$RESP_FILE"' EXIT
HTTP_CODE="$(curl -sS -o "$RESP_FILE" -w '%{http_code}' \
  -X POST "${BASE}/v1/videos" \
  -H 'Content-Type: application/json' \
  --data-binary "$REQ_BODY" || true)"

if [ "$HTTP_CODE" != "200" ] && [ "$HTTP_CODE" != "201" ]; then
  warn "提交失败（HTTP ${HTTP_CODE}），服务端响应："
  cat "$RESP_FILE" >&2
  exit 1
fi

VIDEO_ID="$("$PY_BIN" -c 'import sys,json;print(json.load(open(sys.argv[1])).get("id",""))' "$RESP_FILE")"
[ -n "$VIDEO_ID" ] || { warn "未能从响应解析任务 id："; cat "$RESP_FILE" >&2; exit 1; }
log "任务已创建：id=${VIDEO_ID}"

# -----------------------------------------------------------------------------
# 3. 轮询任务状态
# -----------------------------------------------------------------------------
WAITED=0
while :; do
  STATUS="$(curl -sS "${BASE}/v1/videos/${VIDEO_ID}" \
    | "$PY_BIN" -c 'import sys,json
try:
    print(json.load(sys.stdin).get("status",""))
except Exception:
    print("")' || true)"

  case "$STATUS" in
    completed|succeeded)
      log "任务完成，开始下载视频 ..."
      break
      ;;
    failed|error|canceled|cancelled)
      warn "任务失败（status=${STATUS}）。详情："
      curl -sS "${BASE}/v1/videos/${VIDEO_ID}" >&2 || true
      exit 1
      ;;
    *)
      log "状态：${STATUS:-unknown}，${POLL_INTERVAL}s 后重试 ..."
      ;;
  esac

  if [ "$WAITED" -ge "$MAX_WAIT" ]; then
    die "等待超时（${MAX_WAIT}s），任务仍未完成。可稍后手动查询：curl ${BASE}/v1/videos/${VIDEO_ID}"
  fi
  sleep "$POLL_INTERVAL"
  WAITED=$((WAITED + POLL_INTERVAL))
done

# -----------------------------------------------------------------------------
# 4. 下载 MP4 并校验
# -----------------------------------------------------------------------------
curl -fsS "${BASE}/v1/videos/${VIDEO_ID}/content" --output "$OUTPUT"
log "视频已保存：${OUTPUT}"

if command -v ffprobe >/dev/null 2>&1; then
  log "ffprobe 校验："
  ffprobe -v error -show_entries \
    stream=index,codec_name,width,height,r_frame_rate,sample_rate,channels \
    -of default=noprint_wrappers=1 "$OUTPUT" || warn "ffprobe 解析失败，请手动检查输出文件。"
fi

cat <<EOF

========================== 生成完成 ==========================
输出文件：${OUTPUT}
正常输出应为：H.264 视频 24fps + 立体声 AAC 音频 32kHz。
再次生成：bash ${SCRIPT_DIR}/request-ref2va.sh
============================================================
EOF
