#!/usr/bin/env bash
set -Eeuo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
FRONTEND_DIR="${ROOT_DIR}/frontend"
PYTHON_BIN="${PYTHON_BIN:-${ROOT_DIR}/.venv/bin/python}"
BACKEND_HOST="${BACKEND_HOST:-127.0.0.1}"
BACKEND_PORT="${BACKEND_PORT:-8080}"
APP_BASE="${AUTOALPHA_APP_BASE:-/v2/}"
TUNNEL_NAME="${AUTOALPHA_CLOUDFLARE_TUNNEL_NAME:-autoalpha}"
TUNNEL_HOSTNAME="${AUTOALPHA_PUBLIC_HOSTNAME:-autoalpha.cn}"
CF_CONFIG_DIR="${HOME}/.cloudflared"
CF_CONFIG_PATH="${CF_CONFIG_DIR}/config.yml"
CF_CREDENTIALS_PATH="${CF_CONFIG_DIR}/6769c500-6f73-413e-804e-a855f8c05a0a.json"
LAUNCH_AGENT_DIR="${HOME}/Library/LaunchAgents"
LAUNCH_LOG_DIR="${HOME}/Library/Logs/autoalpha_v2"
TUNNEL_AGENT_LABEL="com.autoalpha.v2.tunnel"
TUNNEL_AGENT_PLIST="${LAUNCH_AGENT_DIR}/${TUNNEL_AGENT_LABEL}.plist"
BACKEND_LOG="${ROOT_DIR}/server.public.log"
TUNNEL_LOG="${ROOT_DIR}/cloudflared.public.log"
TUNNEL_LAUNCH_LOG="${LAUNCH_LOG_DIR}/cloudflared.public.log"
BACKEND_PID_FILE="${ROOT_DIR}/server.public.pid"
TUNNEL_PID_FILE="${ROOT_DIR}/cloudflared.public.pid"
TUNNEL_METRICS="${AUTOALPHA_TUNNEL_METRICS:-127.0.0.1:20243}"

if [[ ! -x "${PYTHON_BIN}" ]]; then
  if [[ -x "/opt/miniconda3/bin/python" ]]; then
    PYTHON_BIN="/opt/miniconda3/bin/python"
  else
    echo "[deploy_web] Missing runnable python: ${PYTHON_BIN}" >&2
    exit 1
  fi
fi

if [[ ! -f "${CF_CREDENTIALS_PATH}" ]]; then
  echo "[deploy_web] Missing Cloudflare tunnel credentials: ${CF_CREDENTIALS_PATH}" >&2
  exit 1
fi

mkdir -p "${CF_CONFIG_DIR}"

stop_pid_file() {
  local file="$1"
  if [[ ! -f "${file}" ]]; then
    return 0
  fi
  local pid
  pid="$(tr -cd '0-9' < "${file}" || true)"
  if [[ -n "${pid}" ]] && kill -0 "${pid}" >/dev/null 2>&1; then
    kill "${pid}" >/dev/null 2>&1 || true
    sleep 1
    kill -9 "${pid}" >/dev/null 2>&1 || true
  fi
  rm -f "${file}"
}

release_port() {
  local port="$1"
  local pids
  pids="$(lsof -tiTCP:"${port}" -sTCP:LISTEN 2>/dev/null || true)"
  if [[ -n "${pids}" ]]; then
    kill ${pids} >/dev/null 2>&1 || true
    sleep 1
    kill -9 ${pids} >/dev/null 2>&1 || true
  fi
}

wait_for_url() {
  local url="$1"
  local attempts="${2:-40}"
  for ((i=1; i<=attempts; i++)); do
    if curl -fsS --max-time 2 "${url}" >/dev/null 2>&1; then
      return 0
    fi
    sleep 1
  done
  return 1
}

write_tunnel_launch_agent() {
  mkdir -p "${LAUNCH_AGENT_DIR}"
  mkdir -p "${LAUNCH_LOG_DIR}"
  cat > "${TUNNEL_AGENT_PLIST}" <<EOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>Label</key>
  <string>${TUNNEL_AGENT_LABEL}</string>
  <key>ProgramArguments</key>
  <array>
    <string>/opt/homebrew/bin/cloudflared</string>
    <string>--metrics</string>
    <string>${TUNNEL_METRICS}</string>
    <string>--config</string>
    <string>${CF_CONFIG_PATH}</string>
    <string>tunnel</string>
    <string>run</string>
  </array>
  <key>WorkingDirectory</key>
  <string>/tmp</string>
  <key>EnvironmentVariables</key>
  <dict>
    <key>HOME</key>
    <string>${HOME}</string>
    <key>USER</key>
    <string>$(id -un)</string>
    <key>PATH</key>
    <string>/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin</string>
  </dict>
  <key>RunAtLoad</key>
  <true/>
  <key>KeepAlive</key>
  <true/>
  <key>StandardOutPath</key>
  <string>${TUNNEL_LAUNCH_LOG}</string>
  <key>StandardErrorPath</key>
  <string>${TUNNEL_LAUNCH_LOG}</string>
</dict>
</plist>
EOF
}

echo "[deploy_web] Building frontend for ${APP_BASE}"
cd "${ROOT_DIR}"
AUTOALPHA_APP_BASE="${APP_BASE}" npm --prefix "${FRONTEND_DIR}" run build

echo "[deploy_web] Writing Cloudflare config ${CF_CONFIG_PATH}"
cat > "${CF_CONFIG_PATH}" <<EOF
tunnel: 6769c500-6f73-413e-804e-a855f8c05a0a
credentials-file: ${CF_CREDENTIALS_PATH}

ingress:
  - hostname: ${TUNNEL_HOSTNAME}
    service: http://${BACKEND_HOST}:${BACKEND_PORT}
  - service: http_status:404
EOF

echo "[deploy_web] Stopping stale public services"
stop_pid_file "${BACKEND_PID_FILE}"
stop_pid_file "${TUNNEL_PID_FILE}"
release_port "${BACKEND_PORT}"

echo "[deploy_web] Starting backend on ${BACKEND_HOST}:${BACKEND_PORT}"
: > "${BACKEND_LOG}"
nohup env AUTOALPHA_APP_BASE="${APP_BASE}" "${PYTHON_BIN}" "${ROOT_DIR}/server.py" > "${BACKEND_LOG}" 2>&1 < /dev/null &
BACKEND_PID=$!
echo "${BACKEND_PID}" > "${BACKEND_PID_FILE}"

if ! wait_for_url "http://${BACKEND_HOST}:${BACKEND_PORT}/api/health" 45; then
  echo "[deploy_web] Backend failed to become healthy. See ${BACKEND_LOG}" >&2
  exit 1
fi

echo "[deploy_web] Starting Cloudflare tunnel ${TUNNEL_NAME}"
if [[ "$(uname -s)" == "Darwin" ]] && command -v launchctl >/dev/null 2>&1; then
  write_tunnel_launch_agent
  launchctl bootout "gui/$(id -u)" "${HOME}/Library/LaunchAgents/com.cloudflare.cloudflared.plist" >/dev/null 2>&1 || true
  launchctl bootout "gui/$(id -u)" "${HOME}/Library/LaunchAgents/com.scientech.autoalpha.tunnel.plist" >/dev/null 2>&1 || true
  launchctl bootout "gui/$(id -u)" "${TUNNEL_AGENT_PLIST}" >/dev/null 2>&1 || true
  : > "${TUNNEL_LAUNCH_LOG}"
  launchctl bootstrap "gui/$(id -u)" "${TUNNEL_AGENT_PLIST}"
  launchctl kickstart -k "gui/$(id -u)/${TUNNEL_AGENT_LABEL}"
  sleep 3
else
  : > "${TUNNEL_LOG}"
  nohup cloudflared tunnel --config "${CF_CONFIG_PATH}" run "${TUNNEL_NAME}" > "${TUNNEL_LOG}" 2>&1 < /dev/null &
  TUNNEL_PID=$!
  echo "${TUNNEL_PID}" > "${TUNNEL_PID_FILE}"
  sleep 3
fi

echo
echo "==========================================="
echo "AutoAlpha public web deployment is running"
echo "Public URL : https://${TUNNEL_HOSTNAME}${APP_BASE%/}"
echo "Backend    : http://${BACKEND_HOST}:${BACKEND_PORT}"
echo "Logs:"
echo "  Backend  : ${BACKEND_LOG}"
echo "  Tunnel   : ${TUNNEL_LAUNCH_LOG}"
echo "==========================================="
