#!/usr/bin/env bash
# 不依赖前端，直接启动研究循环（research_loop.py）
# 日志：默认追加到 $OUTPUTS_ROOT/research.log（与 Flask 子进程相同，见 paths.RESEARCH_LOG_PATH）
#
# 用法示例：
#   ./scripts/run_research_loop.sh
#   ./scripts/run_research_loop.sh --max-iters 3 --batch-size 2 --seed "价量因子挖掘"
#   AUTOALPHA_USE_LLM=0 ./scripts/run_research_loop.sh   # 强制仅用 EA

set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

MAX_ITERS="${MAX_ITERS:-5}"
BATCH="${BATCH:-2}"
SEED="${SEED:-价量因子挖掘}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --max-iters) MAX_ITERS="$2"; shift 2 ;;
    --batch-size) BATCH="$2"; shift 2 ;;
    --seed) SEED="$2"; shift 2 ;;
    -h|--help)
      echo "Usage: $0 [--max-iters N] [--batch-size N] [--seed \"prompt\"]"
      exit 0
      ;;
    *) echo "Unknown arg: $1"; exit 1 ;;
  esac
done

export AUTOALPHA_USE_LLM="${AUTOALPHA_USE_LLM:-1}"
export ALPHACLAW_USE_LLM="${ALPHACLAW_USE_LLM:-$AUTOALPHA_USE_LLM}"

echo "=== AutoAlpha research_loop (no frontend) ==="
echo "  cwd: $ROOT"
echo "  AUTOALPHA_USE_LLM=$AUTOALPHA_USE_LLM"
echo "  max-iters=$MAX_ITERS batch-size=$BATCH"
echo "  seed: $SEED"
echo "  leaderboard: will read PYTHON paths (DATA_ROOT/outputs/leaderboard.json)"
echo ""

# 需要能读 parquet 的环境（conda 的 python 通常带 pyarrow）；勿用系统 /usr/bin/python3
PY="${PYTHON:-$(command -v python)}"
if ! "$PY" -c "import pyarrow" 2>/dev/null; then
  echo "ERROR: current python has no pyarrow: $PY" >&2
  echo "  Try: conda activate autoalpha   OR   export PYTHON=/path/to/conda/python" >&2
  exit 1
fi

exec "$PY" -u research_loop.py \
  --max-iters "$MAX_ITERS" \
  --batch-size "$BATCH" \
  --seed-prompt "$SEED"
