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

BACKEND_PATTERN="inference_engine.api.main"
FRONTEND_PATTERN="node_modules/.bin/vite"

# stop_port PORT PATTERN — stop the listeners on PORT that belong to this project
# (command line matches PATTERN): SIGTERM, wait up to 5s, then SIGKILL.
# Returns 1 if the port is held by an unrelated process (which is left alone).
stop_port() {
  local port="$1" pattern="$2" pid ours=() foreign=0
  for pid in $(lsof -iTCP:"$port" -sTCP:LISTEN -t 2>/dev/null || true); do
    if tr '\0' ' ' < "/proc/$pid/cmdline" 2>/dev/null | grep -q -- "$pattern"; then
      ours+=("$pid")
    else
      foreign=1
    fi
  done
  if [ "${#ours[@]}" -gt 0 ]; then
    kill -TERM "${ours[@]}" 2>/dev/null || true
    for _ in $(seq 1 10); do
      kill -0 "${ours[@]}" 2>/dev/null || break
      sleep 0.5
    done
    kill -KILL "${ours[@]}" 2>/dev/null || true
  fi
  if [ "$foreign" -eq 1 ]; then
    echo -e "  ${RED}✗ Port :$port is used by another program — not touching it. Free the port and retry.${RESET}"
    return 1
  fi
  return 0
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

  # --- Ensure the DeFiGuard model is present ---
  MODEL_FILE="$BACKEND_DIR/models/defi_guard_model.joblib"
  if [ ! -f "$MODEL_FILE" ]; then
    echo -e "  ${RED}❌ DeFiGuard model not found: $MODEL_FILE${RESET}"
    echo -e "     It is versioned in git — run ${BOLD}git checkout -- backend/models/defi_guard_model.joblib${RESET},"
    echo -e "     or copy ${BOLD}checkpoints/defi_guard_model.joblib${RESET} from the DeFiGuard training repository."
    exit 1
  fi

  # --- Inference Engine (port 8001) ---
  if is_port_active 8001 && curl -s -m 1 http://127.0.0.1:8001/health >/dev/null 2>&1; then
    echo -e "  ${GREEN}✓${RESET} Inference Engine already running and healthy on ${BOLD}:8001${RESET}"
  else
    if is_port_active 8001; then
      echo -e "  ${YELLOW}⚠️  Port 8001 occupied but unresponsive, restarting...${RESET}"
      stop_port 8001 "$BACKEND_PATTERN" || exit 1
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
      stop_port 5173 "$FRONTEND_PATTERN" || exit 1
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
  local failed=0
  for spec in "8001:$BACKEND_PATTERN" "5173:$FRONTEND_PATTERN"; do
    local port="${spec%%:*}" pattern="${spec#*:}"
    if is_port_active "$port"; then
      if stop_port "$port" "$pattern"; then
        echo -e "  ${RED}✗${RESET} Stopped service on :$port"
      else
        failed=1
      fi
    else
      echo -e "  ${DIM}  Nothing running on :$port${RESET}"
    fi
  done
  if [ "$failed" -eq 1 ]; then
    return 1
  fi
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

  echo -e "${RED}${BOLD}2. Scan a Transaction (/scan)${RESET}"
  cat << 'CURL2'
curl -s -X POST http://localhost:8001/scan \
  -H "Content-Type: application/json" \
  -d '{
    "from": "0x3b2a1f88e4c5d6a7890bcdef2345678901bcde22",
    "to": "0x8c5d7e12f3a4b5c6d7890abcdef3456789012cd3",
    "value_eth": 0.45,
    "gas_price_gwei": 22,
    "gas_limit": 21000,
    "nonce": 57
  }' | python3 -m json.tool
CURL2
  echo ""

  echo -e "${CYAN}${BOLD}3. Wallet RPC Firewall (/rpc) — add http://localhost:5173/rpc to your wallet${RESET}"
  cat << 'CURL3'
curl -s http://localhost:8001/rpc/info | python3 -m json.tool
curl -s -X POST http://localhost:8001/rpc \
  -H "Content-Type: application/json" \
  -d '{"jsonrpc":"2.0","id":1,"method":"eth_blockNumber","params":[]}'
CURL3
  echo ""
}

# ---- Main ----
case "${1:-}" in
  --stop)     stop_services ;;
  --restart)  stop_services; sleep 1; start_services ;;
  --curls)    banner; print_curls ;;
  *)          start_services ;;
esac
