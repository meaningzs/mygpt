#!/usr/bin/env bash
# Docker 없이 로컬/서버에서 바로 띄우는 스크립트
#   PORT=8501 OPENAI_API_KEY=sk-... ./serve.sh
set -euo pipefail
cd "$(dirname "$0")"

PORT="${PORT:-8501}"
PY="${PYTHON:-python3.11}"

if [ ! -d .venv ]; then
  "$PY" -m venv .venv
  .venv/bin/pip install --upgrade pip
  .venv/bin/pip install -r requirements.txt
fi

exec .venv/bin/streamlit run main.py \
  --server.address=0.0.0.0 --server.port="$PORT" \
  --server.headless=true --browser.gatherUsageStats=false
