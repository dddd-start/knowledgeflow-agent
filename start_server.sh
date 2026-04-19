#!/usr/bin/env bash
# 本地演示服务管理:./start_server.sh start|stop|restart|status
set -uo pipefail
cd "$(dirname "$0")"
PID_FILE="logs/server.pid"
mkdir -p logs
case "${1:-start}" in
  start)
    if [ -f "$PID_FILE" ] && kill -0 "$(cat "$PID_FILE")" 2>/dev/null; then
      echo "already running: $(cat "$PID_FILE")"; exit 0
    fi
    nohup .venv/bin/python -m uvicorn app.api.main:app --host 0.0.0.0 --port "${PORT:-8000}" > logs/server.log 2>&1 &
    echo $! > "$PID_FILE"
    sleep 2
    echo "started pid=$(cat "$PID_FILE") port=${PORT:-8000}"
    ;;
  stop)
    if [ -f "$PID_FILE" ]; then kill "$(cat "$PID_FILE")" 2>/dev/null || true; rm -f "$PID_FILE"; echo stopped; fi
    ;;
  restart)
    "$0" stop; sleep 1; "$0" start
    ;;
  status)
    if [ -f "$PID_FILE" ] && kill -0 "$(cat "$PID_FILE")" 2>/dev/null; then echo "running pid=$(cat "$PID_FILE")"; else echo "stopped"; fi
    ;;
esac
