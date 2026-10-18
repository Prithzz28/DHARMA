#!/usr/bin/env bash
# =============================================================================
# demo.sh — Web3 Transaction Firewall: 1-Click Demo Launcher
# =============================================================================
# Starts services if not already running:
#   1. Inference Engine (port 8001)
#   2. Vite Frontend    (port 5173)
#
# Usage:
#   bash demo.sh          # Start all services
#   bash demo.sh --stop   # Stop all services
#   bash demo.sh --curls  # Print cURL demo commands only
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BACKEND_DIR="$SCRIPT_DIR/backend"
FRONTEND_DIR="$SCRIPT_DIR/frontend"

# Colors
RED='\033[91m'
GREEN='\033[92m'
YELLOW='\033[93m'
CYAN='\033[96m'
BOLD='\033[1m'
DIM='\033[2m'
RESET='\033[0m'

banner() {
  echo ""
  echo -e "${CYAN}${BOLD}╔══════════════════════════════════════════════════════════════╗${RESET}"
  echo -e "${CYAN}${BOLD}║${RESET}  🛡️  ${CYAN}${BOLD}Web3 Transaction Firewall — Live Demo${RESET}                  ${CYAN}${BOLD}║${RESET}"
  echo -e "${CYAN}${BOLD}╚══════════════════════════════════════════════════════════════╝${RESET}"
  echo ""
}

is_port_active() {
  lsof -iTCP:"$1" -sTCP:LISTEN -t >/dev/null 2>&1
}

open_browser() {
  local url="$1"
  if [ -n "${DISPLAY:-}" ] || [ -n "${WAYLAND_DISPLAY:-}" ]; then
    echo -e "  ${CYAN}🚀 Launching dashboard in browser: ${BOLD}$url${RESET}"
    if command -v xdg-open >/dev/null 2>&1; then
      xdg-open "$url" >/dev/null 2>&1 &
    elif command -v firefox >/dev/null 2>&1; then
      firefox "$url" >/dev/null 2>&1 &
    elif command -v google-chrome >/dev/null 2>&1; then
      google-chrome "$url" >/dev/null 2>&1 &
    elif command -v python3 >/dev/null 2>&1; then
      python3 -m webbrowser "$url" >/dev/null 2>&1 &
    fi
  else
    echo -e "  ${YELLOW}ℹ️  Headless environment detected. Please open $url in your browser.${RESET}"
  fi
}

start_services() {
  banner

  # --- Ensure RF model exists (auto-download on first run) ---
  MODEL_FILE="$BACKEND_DIR/models/random_forest_fraud_model.joblib"
  if [ ! -f "$MODEL_FILE" ]; then
    echo -e "  ${YELLOW}📦 Model not found — running download_model.py...${RESET}"
    PYTHON_BIN="python"
    if [ -f "$SCRIPT_DIR/.venv/bin/python" ]; then
      PYTHON_BIN="$SCRIPT_DIR/.venv/bin/python"
    elif command -v python3 >/dev/null 2>&1; then
      PYTHON_BIN="python3"
    fi
    cd "$BACKEND_DIR"
    "$PYTHON_BIN" download_model.py || {
      echo -e "  ${RED}❌ Model download failed. Check your internet connection and try again.${RESET}"
      exit 1
    }
    echo -e "  ${GREEN}✓${RESET} Model downloaded successfully."
  fi

  # --- Inference Engine (port 8001) ---
  if is_port_active 8001 && curl -s -m 1 http://127.0.0.1:8001/health >/dev/null 2>&1; then
    echo -e "  ${GREEN}✓${RESET} Inference Engine already running and healthy on ${BOLD}:8001${RESET}"
  else
    if is_port_active 8001; then
      echo -e "  ${YELLOW}⚠️  Port 8001 occupied but unresponsive, restarting...${RESET}"
      lsof -iTCP:8001 -sTCP:LISTEN -t 2>/dev/null | xargs kill -9 2>/dev/null || true
      sleep 0.5
    fi
    echo -e "  ${YELLOW}⏳${RESET} Starting Inference Engine on :8001 ..."
    cd "$BACKEND_DIR"
    PYTHON_BIN="python"
    if [ -f "$SCRIPT_DIR/.venv/bin/python" ]; then
      PYTHON_BIN="$SCRIPT_DIR/.venv/bin/python"
    elif command -v python3 >/dev/null 2>&1; then
      PYTHON_BIN="python3"
    fi
    PYTHONUNBUFFERED=1 PYTHONPATH=. nohup "$PYTHON_BIN" -m inference_engine.api.main > /tmp/firewall_inference.log 2>&1 &
    disown $! || true
  fi

  # Wait for Inference Engine readiness
  echo -ne "  ${YELLOW}⏳${RESET} Verifying Inference Engine (:8001)... "
  local ie_ready=0
  for i in $(seq 1 30); do
    if curl -s -m 1 http://127.0.0.1:8001/health >/dev/null 2>&1; then
      ie_ready=1
      echo -e "${GREEN}Ready!${RESET}"
      break
    fi
    sleep 0.5
  done
  if [ "$ie_ready" -eq 0 ]; then
    echo -e "${YELLOW}Still loading models in background... (log: /tmp/firewall_inference.log)${RESET}"
  fi

  # --- Vite Frontend (port 5173) ---
  if is_port_active 5173 && curl -s -m 1 http://127.0.0.1:5173 >/dev/null 2>&1; then
    echo -e "  ${GREEN}✓${RESET} Vite Frontend already running and responsive on ${BOLD}:5173${RESET}"
  else
    if is_port_active 5173; then
      echo -e "  ${YELLOW}⚠️  Port 5173 occupied but unresponsive, restarting...${RESET}"
      lsof -iTCP:5173 -sTCP:LISTEN -t 2>/dev/null | xargs kill -9 2>/dev/null || true
      sleep 0.5
    fi
    echo -e "  ${YELLOW}⏳${RESET} Starting Vite Frontend on :5173 ..."
    cd "$FRONTEND_DIR"
    nohup ./node_modules/.bin/vite --host 0.0.0.0 --port 5173 > /tmp/firewall_frontend.log 2>&1 &
    disown $! || true
  fi

  # Wait for Frontend readiness
  echo -ne "  ${YELLOW}⏳${RESET} Verifying Frontend (:5173)... "
  local fe_ready=0
  for i in $(seq 1 30); do
    if curl -s -m 1 http://127.0.0.1:5173 >/dev/null 2>&1; then
      fe_ready=1
      echo -e "${GREEN}Ready!${RESET}"
      break
    fi
    sleep 0.5
  done
  if [ "$fe_ready" -eq 0 ]; then
    echo -e "${YELLOW}Still starting dev server in background... (log: /tmp/firewall_frontend.log)${RESET}"
  fi

  echo ""
  echo -e "${CYAN}${BOLD}╔══════════════════════════════════════════════════════════════╗${RESET}"
  echo -e "${CYAN}${BOLD}║${RESET}  ${GREEN}✓${RESET} ${BOLD}Services Active and Ready!${RESET}                                  ${CYAN}${BOLD}║${RESET}"
  echo -e "${CYAN}${BOLD}║${RESET}  • Dashboard:          ${BOLD}${CYAN}http://localhost:5173${RESET}                  ${CYAN}${BOLD}║${RESET}"
  echo -e "${CYAN}${BOLD}║${RESET}  • Inference API Docs: ${BOLD}${CYAN}http://localhost:8001/docs${RESET}             ${CYAN}${BOLD}║${RESET}"
  echo -e "${CYAN}${BOLD}╚══════════════════════════════════════════════════════════════╝${RESET}"
  echo ""

  # Automatically launch dashboard in web browser
  open_browser "http://localhost:5173"

  print_curls
}

stop_services() {
  banner
  echo -e "  ${RED}Stopping services...${RESET}"
  for port in 8001 5173; do
    pids=$(lsof -iTCP:"$port" -sTCP:LISTEN -t 2>/dev/null || true)
    if [ -n "$pids" ]; then
      echo "$pids" | xargs kill -9 2>/dev/null || true
      echo -e "  ${RED}✗${RESET} Stopped processes on :$port"
    else
      echo -e "  ${DIM}  Nothing running on :$port${RESET}"
    fi
  done
  pkill -f "inference_engine.api.main" 2>/dev/null || true
  pkill -f "node_modules/.bin/vite" 2>/dev/null || true
  echo -e "  ${GREEN}✓${RESET} All services stopped cleanly."
  echo ""
}

print_curls() {
  echo -e "${CYAN}${BOLD}═══ Sample cURL API Commands ═══${RESET}"
  echo ""

  echo -e "${GREEN}${BOLD}1. Health & Model Status${RESET}"
  cat << 'CURL1'
curl -s http://localhost:8001/health | python3 -m json.tool
CURL1
  echo ""

  echo -e "${RED}${BOLD}2. Suspicious Transaction Check (/predict)${RESET}"
  cat << 'CURL2'
curl -s -X POST http://localhost:8001/predict \
  -H "Content-Type: application/json" \
  -d '{
    "length_transaction_hash": 66.0,
    "length_to": 42.0,
    "length_from": 42.0,
    "is_same_address": 1.0,
    "index": 1.0,
    "block_number": 19500000.0,
    "chain_id": 1.0,
    "cumulative_gas_used": 848000.0,
    "gas_used": 848000.0,
    "gas_efficiency": 1.0,
    "effective_gas_price": 2040000000.0,
    "total_gas_cost": 1729920000000000.0,
    "gas_price_ratio": 3.8,
    "gas_per_log_event": 848000.0,
    "value": 0.0,
    "normalized_token_transfer": 0.0,
    "log_count": 0.0,
    "log_removed": 0.0,
    "length_log": 0.0,
    "event_activity_flag": 0.0
  }' | python3 -m json.tool
CURL2
  echo ""
}

# ---- Main ----
case "${1:-}" in
  --stop)     stop_services ;;
  --restart)  stop_services; sleep 1; start_services ;;
  --curls)    banner; print_curls ;;
  *)          start_services ;;
esac
