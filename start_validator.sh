#!/usr/bin/env bash
#
# Launch the production RLVR validator after fail-closed local preflights.
#
# First run:
#   ./setup_validator.sh --wallet-name NAME --wallet-hotkey HOTKEY
#   ./start_validator.sh

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${REPO_ROOT}"

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
  sed -n '2,8p' "$0"
  exit 0
fi
if (( $# != 0 )); then
  echo "[start_validator] ERROR: unknown argument: $1" >&2
  exit 2
fi

if [[ ! -f ".venv/bin/activate" || ! -x ".venv/bin/python" ]]; then
  echo "[start_validator] ERROR: .venv is missing; run ./setup_validator.sh first." >&2
  exit 1
fi
if [[ ! -f ".env" ]]; then
  echo "[start_validator] ERROR: .env is missing; run ./setup_validator.sh first." >&2
  exit 1
fi

# shellcheck disable=SC1091
source .venv/bin/activate
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"

set -a
# shellcheck disable=SC1091
source .env
set +a

: "${NETUID:?set NETUID in .env}"
: "${WALLET_NAME:?set WALLET_NAME in .env}"
: "${WALLET_HOTKEY:?set WALLET_HOTKEY in .env}"
: "${PROBLEM_SERVER_URL:?set PROBLEM_SERVER_URL in .env}"
: "${SUBTENSOR_NETWORK:?set SUBTENSOR_NETWORK in .env}"

if [[ "${WALLET_NAME}" == "YOUR_WALLET_NAME" ||
      "${WALLET_HOTKEY}" == "YOUR_WALLET_HOTKEY" ]]; then
  echo "[start_validator] ERROR: set WALLET_NAME and WALLET_HOTKEY in .env." >&2
  exit 1
fi

"${REPO_ROOT}/scripts/preflight_validator.sh"

: "${LOGLEVEL:=info}"
export LOGLEVEL
echo "[start_validator] netuid=${NETUID} network=${SUBTENSOR_NETWORK} wallet=${WALLET_NAME}/${WALLET_HOTKEY}"
exec python -u scripts/run_validator.py
