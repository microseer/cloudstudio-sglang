#!/usr/bin/env bash
# =============================================================================
# CloudStudio × MiniMax-H3 Ref2VA 应用模板 - Comfy INT8 embedding 设备对齐修复
#
# 修复的 bug（首次生成请求即崩，日志特征）：
#   [MiniMaxH3TextEncodingStage] Error during execution ...:
#   RuntimeError: Expected all tensors to be on the same device, but got
#   index is on cpu, different from other tensors on cuda:0
#     ... runtime/layers/quantization/comfy_int8.py", line 93, in embedding
#     scale = F.embedding(input_, layer.weight_scale).to(self.output_dtype)
#
# 根因：
#   量化配方同时启用 --layerwise-offload-components text_encoder 时，
#   layerwise_offload.py 的"词表驻留主机"优化（host_resident_table_names =
#   ["embed_tokens"]，Qwen3-VL 32B 的 int8 词表 ≥256MiB 命中阈值）会把该层
#   weight 搬到 CPU，并挂 hook 把 token 索引也强制送到 CPU 做 gather；
#   但该层反量化用的 weight_scale 仍留在 cuda:0，ComfyInt8EmbeddingMethod
#   的第二级查表 F.embedding(input_cpu, weight_scale_cuda) 设备不一致 → 崩。
#   （bf16 全精度无 weight_scale，不触发；仅 量化TE + TE offload 组合命中。）
#
# 修复：
#   在 comfy_int8.py 的 embedding() 里，把 weight_scale 对齐到 weight 所在
#   设备再查表（scale 仅约 0.6MB，开销可忽略；权重在 GPU 时为空操作）。
#
# 特性：
#   - 幂等：可重复执行，已修复则直接跳过
#   - 修改前备份为 *.bak-pre-int8fix；回滚：cp <备份> <目标文件>
#   - 补丁后自动 py_compile 语法校验
#
# 用法：
#   bash scripts/fix-comfy-int8-embedding.sh
#   修复后需重启服务生效：bash scripts/serve.sh
#
# 可用环境变量：
#   SGLANG_VENV=/path/venv          自定义虚拟环境目录（默认 <项目根>/.venv）
# =============================================================================
set -euo pipefail

log()  { printf '\033[1;32m[fix-int8]\033[0m %s\n' "$*"; }
warn() { printf '\033[1;33m[fix-int8]\033[0m %s\n' "$*" >&2; }
die()  { printf '\033[1;31m[fix-int8]\033[0m %s\n' "$*" >&2; exit 1; }

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
VENV="${SGLANG_VENV:-$PROJECT_DIR/.venv}"
PY="$VENV/bin/python"

[ -x "$PY" ] || die "未找到虚拟环境 Python：$PY
       请先执行：bash $SCRIPT_DIR/install-sglang.sh"

# -----------------------------------------------------------------------------
# 1. 定位目标文件（通过 import 解析，兼容任意 Python 版本目录）
# -----------------------------------------------------------------------------
SG_PKG_DIR="$("$PY" -c 'import os, sglang; print(os.path.dirname(sglang.__file__))')" \
  || die "导入 sglang 失败，请确认虚拟环境完整（$PY）"

TARGET="$SG_PKG_DIR/multimodal_gen/runtime/layers/quantization/comfy_int8.py"
[ -f "$TARGET" ] || die "目标文件不存在：$TARGET
       当前 SGLang 版本可能不含 multimodal_gen（diffusion extra）。"

# -----------------------------------------------------------------------------
# 2. 幂等检查
# -----------------------------------------------------------------------------
if grep -qF "weight_scale.to(weight.device)" "$TARGET"; then
  log "已修复过（检测到设备对齐代码），跳过。"
  exit 0
fi

# -----------------------------------------------------------------------------
# 3. 备份并打补丁
# -----------------------------------------------------------------------------
BACKUP="$TARGET.bak-pre-int8fix"
[ -f "$BACKUP" ] || { cp -p "$TARGET" "$BACKUP"; log "已备份原文件：$BACKUP"; }

TARGET="$TARGET" "$PY" - <<'PYEOF'
import os
from pathlib import Path

p = Path(os.environ["TARGET"])
s = p.read_text(encoding="utf-8")

old = """        weight = F.embedding(input_, layer.weight)
        if self.tensorwise:
            # scalar-scale exports multiply in FP32 before rounding to the activation dtype
            return (weight.float() * layer.weight_scale).to(self.output_dtype)
        scale = F.embedding(input_, layer.weight_scale).to(self.output_dtype)
        return weight.to(self.output_dtype) * scale"""
new = """        weight = F.embedding(input_, layer.weight)
        # Host-resident tables (layerwise offload) keep `weight` on CPU and the
        # gather hooks send indices to the host too; `weight_scale` may still
        # live on the device, so align it with the table before the lookup.
        scale = layer.weight_scale.to(weight.device)
        if self.tensorwise:
            # scalar-scale exports multiply in FP32 before rounding to the activation dtype
            return (weight.float() * scale).to(self.output_dtype)
        scale = F.embedding(input_, scale).to(self.output_dtype)
        return weight.to(self.output_dtype) * scale"""

if old not in s:
    die_msg = (
        "未匹配到已知代码块：SGLang 版本与补丁不兼容。\n"
        f"请手工检查：{p}\n"
        "定位 embedding() 中 scale = F.embedding(input_, layer.weight_scale) 一行，\n"
        "在其之前加入：scale_pre = layer.weight_scale.to(layer.weight.device)\n"
        "并让两处 scale 查表改用对齐后的张量（参考仓库 scripts/fix-comfy-int8-embedding.sh 注释）。"
    )
    raise SystemExit(die_msg)

p.write_text(s.replace(old, new, 1), encoding="utf-8")
print("patched:", p)
PYEOF

# -----------------------------------------------------------------------------
# 4. 校验
# -----------------------------------------------------------------------------
grep -qF "weight_scale.to(weight.device)" "$TARGET" \
  || die "补丁未生效（未找到标记代码），请手工检查：$TARGET"

"$PY" -m py_compile "$TARGET" \
  || die "补丁后语法校验失败，请回滚：cp '$BACKUP' '$TARGET'"

log "语法校验通过。"
log "修复完成：$TARGET"
warn "请重启服务生效：bash $SCRIPT_DIR/serve.sh"
