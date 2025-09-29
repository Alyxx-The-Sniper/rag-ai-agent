#!/usr/bin/env bash
set -euo pipefail

uvicorn main:app --host 0.0.0.0 --port 8000 &   # internal API
API_PID=$!

cleanup() { kill -TERM "$API_PID" 2>/dev/null || true; }
trap cleanup INT TERM

# Use Railway's PORT (fallback to 8501 locally)
streamlit run ui.py --server.address 0.0.0.0 --server.port "${PORT:-8501}"
