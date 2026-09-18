#!/usr/bin/env bash
#
# Launch the Bedrock-backed V3 demo miner.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${REPO_ROOT}"

if [[ -f ".venv/bin/activate" ]]; then
  # shellcheck disable=SC1091
  source .venv/bin/activate
else
  echo "[start_demo_miner] WARNING: .venv not found; using current Python." >&2
fi
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"

if [[ -f ".env" ]]; then
  set -a
  # shellcheck disable=SC1091
  source .env
  set +a
else
  echo "[start_demo_miner] WARNING: .env not found; using process environment." >&2
fi

: "${BEDROCK_API_KEY:?set BEDROCK_API_KEY in .env or the process environment}"
: "${NETUID:?set NETUID in .env or the process environment}"
: "${WALLET_NAME:?set WALLET_NAME in .env or the process environment}"
: "${WALLET_HOTKEY:?set WALLET_HOTKEY in .env or the process environment}"
: "${SUBTENSOR_NETWORK:=test}"
export BEDROCK_API_KEY NETUID WALLET_NAME WALLET_HOTKEY SUBTENSOR_NETWORK

echo "[start_demo_miner] netuid=${NETUID} network=${SUBTENSOR_NETWORK} wallet=${WALLET_NAME}/${WALLET_HOTKEY}"
exec python scripts/run_demo_miner.py
