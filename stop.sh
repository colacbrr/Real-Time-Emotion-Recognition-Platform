#!/usr/bin/env bash
set -euo pipefail

pkill -f "uvicorn search_server:app" 2>/dev/null || true
pkill -f "vite" 2>/dev/null || true
