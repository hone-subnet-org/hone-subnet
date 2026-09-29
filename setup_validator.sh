#!/usr/bin/env bash
#
# Idempotent RLVR validator bootstrap.
#
# Usage:
#   ./setup_validator.sh --wallet-name NAME --wallet-hotkey HOTKEY
#   ./setup_validator.sh  # then edit WALLET_NAME/HOTKEY in .env

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${REPO_ROOT}"

"${REPO_ROOT}/scripts/require_service_user.sh" setup_validator

wallet_name=""
wallet_hotkey=""
while (( $# )); do
  case "$1" in
    --wallet-name)
      [[ $# -ge 2 ]] || { echo "missing value for --wallet-name" >&2; exit 2; }
      wallet_name="$2"
      shift 2
      ;;
    --wallet-hotkey)
      [[ $# -ge 2 ]] || { echo "missing value for --wallet-hotkey" >&2; exit 2; }
      wallet_hotkey="$2"
      shift 2
      ;;
    -h|--help)
      sed -n '2,8p' "$0"
      exit 0
      ;;
    *)
      echo "[setup_validator] ERROR: unknown argument: $1" >&2
      exit 2
      ;;
  esac
done

for wallet_value in "${wallet_name}" "${wallet_hotkey}"; do
  if [[ -n "${wallet_value}" &&
        ! "${wallet_value}" =~ ^[A-Za-z0-9._-]+$ ]]; then
    echo "[setup_validator] ERROR: wallet names may contain only letters, numbers, ., _, and -." >&2
    exit 2
  fi
done

select_python() {
  local candidate
  for candidate in "${PYTHON:-}" python3.12 python3.11 python3.10; do
    [[ -n "${candidate}" ]] || continue
    command -v "${candidate}" >/dev/null 2>&1 || continue
    if "${candidate}" -c \
      'import sys; raise SystemExit(not ((3, 10) <= sys.version_info[:2] < (3, 13)))'
    then
      command -v "${candidate}"
      return 0
    fi
  done
  return 1
}

if [[ ! -x ".venv/bin/python" ]]; then
  python_bin="$(select_python)" || {
    echo "[setup_validator] ERROR: install Python 3.10, 3.11, or 3.12." >&2
    exit 1
  }
  echo "[setup_validator] creating .venv with ${python_bin}"
  "${python_bin}" -m venv .venv
fi

if ! .venv/bin/python -c \
  'import sys; raise SystemExit(not ((3, 10) <= sys.version_info[:2] < (3, 13)))'
then
  echo "[setup_validator] ERROR: existing .venv is not Python 3.10-3.12." >&2
  exit 1
fi

echo "[setup_validator] installing the pinned validator dependencies"
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -e '.[chain]'

if [[ ! -f ".env" ]]; then
  cat > .env <<'ENV'
# Only the wallet is required. Optional settings and their defaults are
# listed in .env.example; add a line here to change one.
NETUID=5
SUBTENSOR_NETWORK=finney
WALLET_NAME=YOUR_WALLET_NAME
WALLET_HOTKEY=YOUR_WALLET_HOTKEY
ENV
  chmod 0600 .env
  echo "[setup_validator] created .env"
fi

if [[ -n "${wallet_name}" || -n "${wallet_hotkey}" ]]; then
  RLVR_SETUP_WALLET_NAME="${wallet_name}" \
  RLVR_SETUP_WALLET_HOTKEY="${wallet_hotkey}" \
  .venv/bin/python - <<'PY'
import os
from pathlib import Path

path = Path(".env")
updates = {
    key: value
    for key, value in {
        "WALLET_NAME": os.environ.get("RLVR_SETUP_WALLET_NAME", ""),
        "WALLET_HOTKEY": os.environ.get("RLVR_SETUP_WALLET_HOTKEY", ""),
    }.items()
    if value
}
lines = path.read_text(encoding="utf-8").splitlines()
seen = set()
result = []
for line in lines:
    key = line.split("=", 1)[0] if "=" in line and not line.startswith("#") else ""
    if key in updates:
        result.append(f"{key}={updates[key]}")
        seen.add(key)
    else:
        result.append(line)
for key, value in updates.items():
    if key not in seen:
        result.append(f"{key}={value}")
path.write_text("\n".join(result) + "\n", encoding="utf-8")
PY
fi

set -a
# shellcheck disable=SC1091
source .env
set +a
"${REPO_ROOT}/scripts/preflight_validator.sh" --pull

if [[ "${WALLET_NAME:-}" == "YOUR_WALLET_NAME" ||
      "${WALLET_HOTKEY:-}" == "YOUR_WALLET_HOTKEY" ]]; then
  echo
  echo "[setup_validator] Set WALLET_NAME and WALLET_HOTKEY in .env, then run:"
else
  echo
  echo "[setup_validator] Ready. Run:"
fi
echo "  ./start_validator.sh"
