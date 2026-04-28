#!/usr/bin/env bash
set -u

cd /Volumes/T7 || exit 1

ROUNDS="${AUTOALPHA_V3_ROUNDS:-0}"
IDEAS="${AUTOALPHA_V3_IDEAS:-2}"
DAYS="${AUTOALPHA_V3_DAYS:-0}"
TARGET_VALID="${AUTOALPHA_V3_TARGET_VALID:-100}"
SLEEP_SEC="${AUTOALPHA_V3_RESTART_SLEEP_SEC:-15}"
PYTHON_BIN="${AUTOALPHA_PYTHON_BIN:-/opt/miniconda3/bin/python3}"

while true; do
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] autoalpha_v3 mining start: rounds=${ROUNDS} ideas=${IDEAS} days=${DAYS} target_valid=${TARGET_VALID}"
  args=(--rounds "${ROUNDS}" --ideas "${IDEAS}" --days "${DAYS}")
  if [ "${TARGET_VALID}" -gt 0 ] 2>/dev/null; then
    args+=(--target-valid "${TARGET_VALID}")
  fi
  PYTHONUNBUFFERED=1 "${PYTHON_BIN}" -u -m autoalpha_v3.loop "${args[@]}"
  code=$?
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] autoalpha_v3 mining exited code=${code}; restarting in ${SLEEP_SEC}s"
  sleep "${SLEEP_SEC}"
done
