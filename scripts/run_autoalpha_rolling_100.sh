#!/usr/bin/env bash
# AutoAlpha 专用长实验：
# 1. 持续挖掘直到拿到 100 个有效因子（PassGates + 已导出 parquet）
# 2. 用这批因子做半年训练 / 半年测试的 rolling 线性模型 + LightGBM 实验
# 3. 输出 JSON / Markdown / 图表到 autoalpha/model_lab/run_*/
#
# 示例：
#   ./scripts/run_autoalpha_rolling_100.sh
#   ./scripts/run_autoalpha_rolling_100.sh --target-valid 100 --train-days 126 --test-days 126
#   ./scripts/run_autoalpha_rolling_100.sh --max-rounds 30 --allow-partial

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

TARGET_VALID="${TARGET_VALID:-100}"
IDEAS_PER_ROUND="${IDEAS_PER_ROUND:-3}"
EVAL_DAYS="${EVAL_DAYS:-0}"
MAX_ROUNDS="${MAX_ROUNDS:-0}"
SLEEP_SECONDS="${SLEEP_SECONDS:-2}"
TRAIN_DAYS="${TRAIN_DAYS:-126}"
TEST_DAYS="${TEST_DAYS:-126}"
STEP_DAYS="${STEP_DAYS:-126}"
ALLOW_PARTIAL="${ALLOW_PARTIAL:-0}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --target-valid) TARGET_VALID="$2"; shift 2 ;;
    --ideas-per-round) IDEAS_PER_ROUND="$2"; shift 2 ;;
    --eval-days) EVAL_DAYS="$2"; shift 2 ;;
    --max-rounds) MAX_ROUNDS="$2"; shift 2 ;;
    --sleep-seconds) SLEEP_SECONDS="$2"; shift 2 ;;
    --train-days) TRAIN_DAYS="$2"; shift 2 ;;
    --test-days) TEST_DAYS="$2"; shift 2 ;;
    --step-days) STEP_DAYS="$2"; shift 2 ;;
    --allow-partial) ALLOW_PARTIAL=1; shift 1 ;;
    -h|--help)
      echo "Usage: $0 [--target-valid N] [--ideas-per-round N] [--eval-days N] [--max-rounds N] [--allow-partial]"
      exit 0
      ;;
    *)
      echo "Unknown arg: $1" >&2
      exit 1
      ;;
  esac
done

PY="${PYTHON:-$(command -v python)}"
for module in pyarrow sklearn lightgbm matplotlib; do
  if ! "$PY" -c "import ${module}" 2>/dev/null; then
    echo "ERROR: current python missing dependency '${module}': $PY" >&2
    echo "Try: conda activate autoalpha   OR   export PYTHON=/path/to/conda/python" >&2
    exit 1
  fi
done

echo "=== AutoAlpha Rolling 100-Factor Lab ==="
echo "  cwd: $ROOT"
echo "  python: $PY"
echo "  target-valid: $TARGET_VALID"
echo "  ideas-per-round: $IDEAS_PER_ROUND"
echo "  eval-days: $EVAL_DAYS (0 = full history)"
echo "  max-rounds: $MAX_ROUNDS (0 = until target)"
echo "  rolling train/test/step: $TRAIN_DAYS / $TEST_DAYS / $STEP_DAYS"
echo "  allow-partial: $ALLOW_PARTIAL"
echo ""

ARGS=(
  --target-valid "$TARGET_VALID"
  --ideas-per-round "$IDEAS_PER_ROUND"
  --eval-days "$EVAL_DAYS"
  --max-rounds "$MAX_ROUNDS"
  --sleep-seconds "$SLEEP_SECONDS"
  --train-days "$TRAIN_DAYS"
  --test-days "$TEST_DAYS"
  --step-days "$STEP_DAYS"
)

if [[ "$ALLOW_PARTIAL" == "1" ]]; then
  ARGS+=(--allow-partial)
fi

exec "$PY" -u -m autoalpha.rolling_model_lab "${ARGS[@]}"
