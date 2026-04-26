#!/usr/bin/env bash
set -euo pipefail

ROOT="${AUTOALPHA_DISPLAY_ROOT:-/Volumes/T7/autoalpha_v2_display}"
LABEL="${AUTOALPHA_DISPLAY_LABEL:-com.autoalpha.v2.backend}"
PLIST="$HOME/Library/LaunchAgents/${LABEL}.plist"
PYTHON="${PYTHON:-/opt/miniconda3/bin/python}"

if [[ ! -x "$PYTHON" ]]; then
  PYTHON="$(command -v python3)"
fi

mkdir -p "$HOME/Library/LaunchAgents" "$ROOT/logs"
mkdir -p "$HOME/Library/Logs/autoalpha_v2_display"

launchctl bootout "gui/$(id -u)" "$PLIST" >/dev/null 2>&1 || true
launchctl bootout "gui/$(id -u)" "$HOME/Library/LaunchAgents/com.autoalpha.v2.frontend.plist" >/dev/null 2>&1 || true

cat > "$PLIST" <<PLIST
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>Label</key>
  <string>${LABEL}</string>
  <key>ProgramArguments</key>
  <array>
    <string>/bin/sh</string>
    <string>-c</string>
    <string>cd ${ROOT} &amp;&amp; exec ${PYTHON} ${ROOT}/display_server.py</string>
  </array>
  <key>WorkingDirectory</key>
  <string>/</string>
  <key>EnvironmentVariables</key>
  <dict>
    <key>AUTOALPHA_DISPLAY_HOST</key>
    <string>127.0.0.1</string>
    <key>AUTOALPHA_DISPLAY_PORT</key>
    <string>${AUTOALPHA_DISPLAY_PORT:-8080}</string>
    <key>PYTHONUNBUFFERED</key>
    <string>1</string>
  </dict>
  <key>RunAtLoad</key>
  <true/>
  <key>KeepAlive</key>
  <true/>
  <key>StandardOutPath</key>
  <string>${HOME}/Library/Logs/autoalpha_v2_display/display_server.out.log</string>
  <key>StandardErrorPath</key>
  <string>${HOME}/Library/Logs/autoalpha_v2_display/display_server.err.log</string>
</dict>
</plist>
PLIST

launchctl bootstrap "gui/$(id -u)" "$PLIST"
launchctl kickstart -k "gui/$(id -u)/${LABEL}" >/dev/null 2>&1 || true

echo "Display server started on http://127.0.0.1:${AUTOALPHA_DISPLAY_PORT:-8080}/v2/"
