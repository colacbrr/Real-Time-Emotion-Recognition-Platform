#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

BACKEND_DIR="$ROOT_DIR/backend"
FRONTEND_DIR="$ROOT_DIR/frontend"

cd "$BACKEND_DIR"
if [ ! -d ".venv" ]; then
  python3 -m venv .venv
fi
. .venv/bin/activate
python -m pip install -r requirements.txt
uvicorn server:app --reload --port 8000 &
BACKEND_PID=$!

cleanup() {
  kill "$BACKEND_PID" 2>/dev/null || true
}
trap cleanup EXIT

cd "$FRONTEND_DIR"
if [ ! -d "node_modules" ]; then
  if [ -f "package-lock.json" ]; then
    npm ci
  else
    npm install
  fi
fi
printf "Backend: http://localhost:8000\nFrontend: http://localhost:5173\n"
npm run dev
