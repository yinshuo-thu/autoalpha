#!/usr/bin/env bash
set -Eeuo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FRONTEND_DIR="${ROOT_DIR}/frontend"

BACKEND_HOST="${BACKEND_HOST:-127.0.0.1}"
BACKEND_PORT="${BACKEND_PORT:-8080}"
FRONTEND_HOST="${FRONTEND_HOST:-127.0.0.1}"
FRONTEND_PORT="${FRONTEND_PORT:-3000}"

BACKEND_LOG="${ROOT_DIR}/server.log"
FRONTEND_LOG="${ROOT_DIR}/frontend.log"
BACKEND_PID_FILE="${ROOT_DIR}/server.pid"
FRONTEND_PID_FILE="${ROOT_DIR}/frontend.pid"
LAUNCH_AGENT_DIR="${HOME}/Library/LaunchAgents"
LAUNCH_LOG_DIR="${HOME}/Library/Logs/Scientech"
BACKEND_LABEL="com.scientech.autoalpha.backend"
FRONTEND_LABEL="com.scientech.autoalpha.frontend"
BACKEND_PLIST="${LAUNCH_AGENT_DIR}/${BACKEND_LABEL}.plist"
FRONTEND_PLIST="${LAUNCH_AGENT_DIR}/${FRONTEND_LABEL}.plist"
LAUNCH_DOMAIN="gui/$(id -u)"

export PATH="/opt/homebrew/bin:/usr/local/bin:${PATH}"
export NO_PROXY="localhost,127.0.0.1,${BACKEND_HOST},${FRONTEND_HOST},${NO_PROXY:-}"
export no_proxy="${NO_PROXY}"

usage() {
  cat <<EOF
Usage: ./start_all.sh [--reuse] [--no-install]

Starts the Scientech backend and frontend.

Options:
  --reuse       Reuse healthy services already listening on the target ports.
  --no-install  Skip npm install when frontend/node_modules is missing.

Environment overrides:
  BACKEND_HOST=${BACKEND_HOST}
  BACKEND_PORT=${BACKEND_PORT}
  FRONTEND_HOST=${FRONTEND_HOST}
  FRONTEND_PORT=${FRONTEND_PORT}
EOF
}

REUSE=false
INSTALL_DEPS=true
for arg in "$@"; do
  case "$arg" in
    --reuse) REUSE=true ;;
    --no-install) INSTALL_DEPS=false ;;
    -h|--help) usage; exit 0 ;;
    *) echo "[start_all] Unknown option: $arg" >&2; usage; exit 2 ;;
  esac
done

wait_for_url() {
  local url="$1"
  local name="$2"
  local attempts="${3:-45}"
  local delay="${4:-1}"

  for ((i=1; i<=attempts; i++)); do
    if curl -fsS --max-time 2 "$url" >/dev/null 2>&1; then
      echo "[start_all] ${name} ready: ${url}"
      return 0
    fi
    sleep "$delay"
  done

  echo "[start_all] ${name} failed to become ready: ${url}" >&2
  return 1
}

is_alive() {
  local pid="${1:-}"
  [[ -n "$pid" ]] && kill -0 "$pid" >/dev/null 2>&1
}

stop_pid_file() {
  local file="$1"
  local label="$2"
  if [[ ! -f "$file" ]]; then
    return 0
  fi
  local pid
  pid="$(tr -cd '0-9' < "$file" || true)"
  if is_alive "$pid"; then
    echo "[start_all] Stopping ${label} pid=${pid}"
    kill "$pid" >/dev/null 2>&1 || true
    sleep 1
    if is_alive "$pid"; then
      kill -9 "$pid" >/dev/null 2>&1 || true
    fi
  fi
  rm -f "$file"
}

release_port() {
  local port="$1"
  local label="$2"
  local pids
  pids="$(lsof -tiTCP:"$port" -sTCP:LISTEN 2>/dev/null || true)"
  if [[ -z "$pids" ]]; then
    return 0
  fi
  echo "[start_all] Releasing ${label} port ${port}: ${pids}"
  kill $pids >/dev/null 2>&1 || true
  sleep 1
  kill -9 $pids >/dev/null 2>&1 || true
}

write_launch_agent() {
  local plist="$1"
  local label="$2"
  local workdir="$3"
  local stdout_log="$4"
  local stderr_log="$5"
  shift 5

  xml_escape() {
    sed -e 's/&/\&amp;/g' -e 's/</\&lt;/g' -e 's/>/\&gt;/g' -e 's/"/\&quot;/g' -e "s/'/\&apos;/g"
  }

  mkdir -p "$LAUNCH_AGENT_DIR"
  mkdir -p "$LAUNCH_LOG_DIR"
  {
    cat <<EOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>Label</key>
  <string>${label}</string>
  <key>WorkingDirectory</key>
  <string>${workdir}</string>
  <key>ProgramArguments</key>
  <array>
EOF
    for arg in "$@"; do
      printf '    <string>%s</string>\n' "$(printf '%s' "$arg" | xml_escape)"
    done
    cat <<EOF
  </array>
  <key>EnvironmentVariables</key>
  <dict>
    <key>PATH</key>
    <string>/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin</string>
    <key>NO_PROXY</key>
    <string>${NO_PROXY}</string>
    <key>no_proxy</key>
    <string>${NO_PROXY}</string>
  </dict>
  <key>RunAtLoad</key>
  <true/>
  <key>KeepAlive</key>
  <true/>
  <key>StandardOutPath</key>
  <string>${stdout_log}</string>
  <key>StandardErrorPath</key>
  <string>${stderr_log}</string>
</dict>
</plist>
EOF
  } > "$plist"
}

launch_agent_stop() {
  local label="$1"
  local plist="$2"
  launchctl bootout "$LAUNCH_DOMAIN" "$plist" >/dev/null 2>&1 || true
  launchctl remove "$label" >/dev/null 2>&1 || true
}

launch_agent_start() {
  local label="$1"
  local plist="$2"
  launchctl bootstrap "$LAUNCH_DOMAIN" "$plist" >/dev/null 2>&1 || true
  launchctl kickstart -k "${LAUNCH_DOMAIN}/${label}" >/dev/null 2>&1 || true
}

if [[ -f "${ROOT_DIR}/.env" ]]; then
  set -a
  # shellcheck disable=SC1091
  source "${ROOT_DIR}/.env"
  set +a
fi

if [[ -x "${ROOT_DIR}/.venv/bin/python" ]]; then
  PYTHON="${ROOT_DIR}/.venv/bin/python"
else
  PYTHON="$(command -v python3 || command -v python)"
fi
NPM="$(command -v npm)"

echo "[start_all] Root: ${ROOT_DIR}"
echo "[start_all] Python: ${PYTHON}"
echo "[start_all] npm: ${NPM}"

BACKEND_URL="http://${BACKEND_HOST}:${BACKEND_PORT}/api/health"
FRONTEND_URL="http://${FRONTEND_HOST}:${FRONTEND_PORT}"

if [[ "$REUSE" == true ]]; then
  backend_ok=false
  frontend_ok=false
  if curl -fsS --max-time 2 "$BACKEND_URL" >/dev/null 2>&1; then
    backend_ok=true
    echo "[start_all] Reusing healthy backend on ${BACKEND_URL}"
  fi
  if curl -fsS --max-time 2 "$FRONTEND_URL" >/dev/null 2>&1; then
    frontend_ok=true
    echo "[start_all] Reusing healthy frontend on ${FRONTEND_URL}"
  fi
else
  if [[ "$(uname -s)" == "Darwin" && -x "$(command -v launchctl)" ]]; then
    launch_agent_stop "$BACKEND_LABEL" "$BACKEND_PLIST"
    launch_agent_stop "$FRONTEND_LABEL" "$FRONTEND_PLIST"
  fi
  stop_pid_file "$BACKEND_PID_FILE" "backend"
  stop_pid_file "$FRONTEND_PID_FILE" "frontend"
  release_port "$BACKEND_PORT" "backend"
  release_port "$FRONTEND_PORT" "frontend"
  backend_ok=false
  frontend_ok=false
fi

USE_LAUNCHCTL=false
if [[ "$(uname -s)" == "Darwin" && -x "$(command -v launchctl)" ]]; then
  USE_LAUNCHCTL=true
fi
if [[ "$USE_LAUNCHCTL" == true ]]; then
  mkdir -p "$LAUNCH_LOG_DIR"
  BACKEND_LOG="${LAUNCH_LOG_DIR}/backend.log"
  FRONTEND_LOG="${LAUNCH_LOG_DIR}/frontend.log"
fi

if [[ "${backend_ok:-false}" != true ]]; then
  echo "[start_all] Starting backend: ${BACKEND_URL}"
  if [[ "$USE_LAUNCHCTL" == true ]]; then
    : > "$BACKEND_LOG"
    backend_cmd="cd \"${ROOT_DIR}\" && exec \"${PYTHON}\" \"${ROOT_DIR}/server.py\" >> \"${BACKEND_LOG}\" 2>&1"
    write_launch_agent "$BACKEND_PLIST" "$BACKEND_LABEL" "/tmp" "${LAUNCH_LOG_DIR}/backend.launchd.log" "${LAUNCH_LOG_DIR}/backend.launchd.log" "/bin/zsh" "-lc" "$backend_cmd"
    launch_agent_start "$BACKEND_LABEL" "$BACKEND_PLIST"
  else
    cd "$ROOT_DIR"
    nohup "$PYTHON" server.py > "$BACKEND_LOG" 2>&1 < /dev/null &
    BACKEND_PID=$!
    echo "$BACKEND_PID" > "$BACKEND_PID_FILE"
    disown -h "$BACKEND_PID" 2>/dev/null || true
  fi
fi

if [[ "${frontend_ok:-false}" != true ]]; then
  if [[ "$INSTALL_DEPS" == true && ! -d "${FRONTEND_DIR}/node_modules" ]]; then
    echo "[start_all] Installing frontend dependencies..."
    cd "$FRONTEND_DIR"
    "$NPM" install
  fi

  echo "[start_all] Starting frontend: ${FRONTEND_URL}"
  if [[ "$USE_LAUNCHCTL" == true ]]; then
    : > "$FRONTEND_LOG"
    frontend_cmd="cd \"${FRONTEND_DIR}\" && exec \"${NPM}\" run dev -- --host \"${FRONTEND_HOST}\" --port \"${FRONTEND_PORT}\" >> \"${FRONTEND_LOG}\" 2>&1"
    write_launch_agent "$FRONTEND_PLIST" "$FRONTEND_LABEL" "/tmp" "${LAUNCH_LOG_DIR}/frontend.launchd.log" "${LAUNCH_LOG_DIR}/frontend.launchd.log" "/bin/zsh" "-lc" "$frontend_cmd"
    launch_agent_start "$FRONTEND_LABEL" "$FRONTEND_PLIST"
  else
    cd "$FRONTEND_DIR"
    nohup "$NPM" run dev -- --host "$FRONTEND_HOST" --port "$FRONTEND_PORT" > "$FRONTEND_LOG" 2>&1 < /dev/null &
    FRONTEND_PID=$!
    echo "$FRONTEND_PID" > "$FRONTEND_PID_FILE"
    disown -h "$FRONTEND_PID" 2>/dev/null || true
  fi
fi

wait_for_url "$BACKEND_URL" "Backend"
wait_for_url "$FRONTEND_URL" "Frontend"

if [[ "$USE_LAUNCHCTL" == true ]]; then
  lsof -tiTCP:"$BACKEND_PORT" -sTCP:LISTEN 2>/dev/null | head -1 > "$BACKEND_PID_FILE" || true
  lsof -tiTCP:"$FRONTEND_PORT" -sTCP:LISTEN 2>/dev/null | head -1 > "$FRONTEND_PID_FILE" || true
fi

sleep 2
if ! curl -fsS --max-time 2 "$BACKEND_URL" >/dev/null 2>&1; then
  echo "[start_all] Backend became unhealthy after startup. See ${BACKEND_LOG}" >&2
  exit 1
fi
if ! curl -fsS --max-time 2 "$FRONTEND_URL" >/dev/null 2>&1; then
  echo "[start_all] Frontend became unhealthy after startup. See ${FRONTEND_LOG}" >&2
  exit 1
fi

cat <<EOF

===========================================
Scientech services are running
Backend : ${BACKEND_URL}
Frontend: ${FRONTEND_URL}

Logs:
  Backend : ${BACKEND_LOG}
  Frontend: ${FRONTEND_LOG}

Stop:
  launchctl bootout ${LAUNCH_DOMAIN} ${BACKEND_PLIST} 2>/dev/null || true
  launchctl bootout ${LAUNCH_DOMAIN} ${FRONTEND_PLIST} 2>/dev/null || true
===========================================
EOF
