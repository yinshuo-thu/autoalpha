#!/bin/bash

# Scientech AutoAlpha - One-Click Start Script

echo "=========================================="
echo "    Scientech 2026 Engine Booting up...   "
echo "=========================================="

# 1. Activate autoalpha conda environment gracefully
source ~/.bashrc 2>/dev/null || true
if [ -f "/opt/miniconda3/etc/profile.d/conda.sh" ]; then
    . "/opt/miniconda3/etc/profile.d/conda.sh"
else
    export PATH="/opt/miniconda3/bin:$PATH"
fi

conda activate autoalpha || echo "Could not activate conda autoalpha. Trying without."

# 2. Start Python Backend API (port 8080)
echo "[1/2] Starting Scientech Backend API on port 8080..."
python server.py > server.log 2>&1 &
BACKEND_PID=$!
echo $BACKEND_PID > server.pid

# 3. Start Frontend (port 3000)
echo "[2/2] Starting Vite Frontend on port 3000..."
cd frontend
npm run dev &
FRONTEND_PID=$!
cd ..

echo " "
echo "✅ Systems Online!"
echo "   Backend : http://localhost:8080"
echo "   Frontend: http://localhost:3000   <-- (Open this in Chrome)"
echo " "
echo "Press CTRL+C to stop both applications."

trap "echo 'Shutting down servers...'; kill $BACKEND_PID $FRONTEND_PID; exit 0" SIGINT SIGTERM
wait
