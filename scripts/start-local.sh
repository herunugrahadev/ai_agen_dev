#!/usr/bin/env bash
set -euo pipefail

# ==============================================================================
# OpenMausBot + 9Router - Native Local Runner (Tanpa Docker)
# ==============================================================================
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

PIDS=()

cleanup() {
  echo ""
  echo ">>> Menghentikan seluruh layanan lokal..."
  for pid in "${PIDS[@]:-}"; do
    if kill -0 "$pid" 2>/dev/null; then
      kill "$pid" 2>/dev/null || true
    fi
  done
  wait 2>/dev/null || true
  echo ">>> Selesai."
}
trap cleanup EXIT INT TERM

# 1. Pastikan .env tersedia
if [[ ! -f ".env" ]]; then
  cp .env.example .env
  echo ">>> File .env dibuat dari .env.example (isi API key LLM Anda di .env)."
fi

# 2. Override environment agar berjalan di localhost (SQLite fallback jika tidak pakai Postgres)
export NINE_ROUTER_BASE_URL="${NINE_ROUTER_BASE_URL:-http://localhost:20128/v1}"
export REDIS_URL="${REDIS_URL:-redis://localhost:6379/0}"
export DATABASE_URL="${DATABASE_URL:-sqlite+aiosqlite:///./openmausbot.db}"

# 3. Siapkan Python virtual environment & dependensi
if [[ ! -d ".venv" ]]; then
  echo ">>> Membuat Python virtual environment (.venv)..."
  if command -v uv >/dev/null 2>&1; then
    uv venv .venv
  else
    python3 -m venv .venv
  fi
fi

# shellcheck disable=SC1091
source .venv/bin/activate

if ! python -c "import fastapi, openai, aiosqlite" 2>/dev/null; then
  echo ">>> Menginstall dependensi Python..."
  if command -v uv >/dev/null 2>&1; then
    uv pip install -e ".[dev]"
  else
    pip install -e ".[dev]"
  fi
fi

# 4. Jalankan Redis lokal (jika tersedia); jika tidak, aktifkan mode sync tanpa Redis
if command -v redis-server >/dev/null 2>&1; then
  if ! redis-cli ping >/dev/null 2>&1; then
    echo ">>> Menjalankan redis-server di localhost:6379..."
    redis-server --port 6379 --daemonize no >/tmp/openmausbot-redis.log 2>&1 &
    PIDS+=($!)
    sleep 1
  else
    echo ">>> Redis sudah berjalan di localhost:6379."
  fi
  export REDIS_ENABLED=true
else
  echo ">>> [INFO] redis-server tidak ditemukan. Berjalan dalam mode sinkron (REDIS_ENABLED=false)."
  export REDIS_ENABLED=false
fi

# 5. Jalankan 9Router Gateway di port 20128
if command -v npx >/dev/null 2>&1; then
  echo ">>> Menjalankan 9Router di http://localhost:20128/v1 ..."
  npx --yes 9router >/tmp/openmausbot-9router.log 2>&1 &
  PIDS+=($!)
  sleep 2
else
  echo ">>> [PERINGATAN] Node.js / npx tidak ditemukan. Pastikan 9Router sudah berjalan di $NINE_ROUTER_BASE_URL."
fi

# 6. Jalankan OpenMausBot Worker (jika Redis aktif)
if [[ "${REDIS_ENABLED}" == "true" ]]; then
  echo ">>> Menjalankan OpenMausBot Worker..."
  python -m worker.main >/tmp/openmausbot-worker.log 2>&1 &
  PIDS+=($!)
fi

# 7. Jalankan OpenMausBot REST API & Webhook (foreground)
echo ">>> Menjalankan OpenMausBot API di http://localhost:8000 ..."
echo ">>> Tekan Ctrl+C untuk menghentikan semua layanan."
uvicorn api.main:app --host 0.0.0.0 --port 8000
