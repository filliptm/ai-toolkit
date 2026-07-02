#!/usr/bin/env sh
set -eu

UI_PORT="${UI_PORT:-3000}"
SCRIPT_DIR="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"

echo "Checking port ${UI_PORT}..."

PIDS=""
if command -v lsof >/dev/null 2>&1; then
    PIDS="$(lsof -ti "tcp:${UI_PORT}" -sTCP:LISTEN 2>/dev/null || true)"
elif command -v fuser >/dev/null 2>&1; then
    PIDS="$(fuser "${UI_PORT}/tcp" 2>/dev/null || true)"
elif command -v ss >/dev/null 2>&1; then
    PIDS="$(ss -ltnp 2>/dev/null | awk -v port=":${UI_PORT}" '$4 ~ port "$" { match($0, /pid=[0-9]+/); if (RSTART) print substr($0, RSTART + 4, RLENGTH - 4) }' || true)"
fi

if [ -n "$PIDS" ]; then
    echo "Killing process(es) on port ${UI_PORT}: ${PIDS}"
    kill $PIDS 2>/dev/null || true
    sleep 1
    if command -v lsof >/dev/null 2>&1; then
        REMAINING="$(lsof -ti "tcp:${UI_PORT}" -sTCP:LISTEN 2>/dev/null || true)"
        if [ -n "$REMAINING" ]; then
            kill -9 $REMAINING 2>/dev/null || true
        fi
    fi
else
    echo "Port ${UI_PORT} is available."
fi

cd "${SCRIPT_DIR}/ui"
npm run dev
