#!/usr/bin/env bash
#
# Verify the sandbox image and problem service before any challenge is leased.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

"${REPO_ROOT}/scripts/require_service_user.sh" preflight

pull_image=false
while (( $# )); do
  case "$1" in
    --pull) pull_image=true ;;
    *) echo "[preflight] ERROR: usage: $0 [--pull]" >&2; exit 2 ;;
  esac
  shift
done
if (( $# != 0 )); then
  echo "[preflight] ERROR: usage: $0 [--pull]" >&2
  exit 2
fi

if [[ -f ".env" ]]; then
  set -a
  # shellcheck disable=SC1091
  source .env
  set +a
fi

python_bin="${REPO_ROOT}/.venv/bin/python"
if [[ ! -x "${python_bin}" ]]; then
  python_bin="$(command -v python3 || true)"
fi
if [[ -z "${python_bin}" ]]; then
  echo "[preflight] ERROR: Python 3 is required for bounded preflight checks." >&2
  exit 1
fi

run_timed() {
  local timeout_s="$1"
  shift
  "${python_bin}" - "${timeout_s}" "$@" <<'PY'
import subprocess
import sys

timeout_s = float(sys.argv[1])
command = sys.argv[2:]
try:
    completed = subprocess.run(command, timeout=timeout_s, check=False)
except subprocess.TimeoutExpired:
    print(
        f"[preflight] ERROR: command timed out after {timeout_s:g}s: {command[0]}",
        file=sys.stderr,
    )
    raise SystemExit(124)
raise SystemExit(completed.returncode)
PY
}

if [[ ! -v PROBLEM_SERVER_URL ]]; then
  # Unset means the release default, the production V3 problem server.
  PROBLEM_SERVER_URL="$("${python_bin}" -c 'from rlvr.config import Settings; print(Settings.model_fields["problem_server_url"].default)')"
  export PROBLEM_SERVER_URL
elif [[ -z "${PROBLEM_SERVER_URL}" ]]; then
  echo "[preflight] ERROR: PROBLEM_SERVER_URL is set but empty; unset it for the default or set a server." >&2
  exit 1
else
  # A .env from the previous release names the old server; the validator
  # itself moves to the current default, so the probe must too.
  resolved="$("${python_bin}" -c 'import os; from rlvr.config import Settings; print(Settings(_env_file=None, problem_server_url=os.environ["PROBLEM_SERVER_URL"]).problem_server_url)')"
  if [[ "${resolved}" != "${PROBLEM_SERVER_URL}" ]]; then
    echo "[preflight] NOTE: PROBLEM_SERVER_URL names the previous release's server; using ${resolved}. Remove it from .env."
    PROBLEM_SERVER_URL="${resolved}"
    export PROBLEM_SERVER_URL
  fi
fi

if [[ "${SUBTENSOR_NETWORK:-}" == "finney" && "${NETUID:-}" != "5" ]]; then
  echo "[preflight] ERROR: this Finney release is configured for NETUID=5." >&2
  exit 1
fi
v3_image="$("${python_bin}" - <<'PY'
from rlvr.policy import RELEASE_POLICY
print(RELEASE_POLICY.v3_image)
PY
)"
command -v docker >/dev/null 2>&1 || {
  echo "[preflight] ERROR: Docker CLI not found. Install and start Docker." >&2
  exit 1
}
command -v chmod >/dev/null 2>&1 && command -v rm >/dev/null 2>&1 || {
  echo "[preflight] ERROR: chmod and rm are required for workspace cleanup." >&2
  exit 1
}
run_timed 20 docker info >/dev/null
if [[ "${pull_image}" == true ]]; then
  echo "[preflight] pulling sandbox image ${v3_image}"
  run_timed 900 docker pull "${v3_image}"
elif ! run_timed 20 docker image inspect "${v3_image}" >/dev/null 2>&1; then
  echo "[preflight] ERROR: sandbox image is missing; rerun ./setup_validator.sh." >&2
  exit 1
fi

smoke="$(
  run_timed 45 docker run --rm --pull=never --network=none --read-only \
    --cap-drop=ALL --security-opt=no-new-privileges \
    --user=65534:65534 "${v3_image}" \
    sh -ec 'git --version >/dev/null; command -v timeout >/dev/null; printf rlvr-preflight-ok'
)"
if [[ "${smoke}" != "rlvr-preflight-ok" ]]; then
  echo "[preflight] ERROR: V3 sandbox smoke test failed." >&2
  exit 1
fi
echo "[preflight] sandbox image ready"

"${python_bin}" - <<'PY'
import email.utils
import os
import time
import urllib.parse
import urllib.error
import urllib.request

base = os.environ["PROBLEM_SERVER_URL"].rstrip("/")
parsed = urllib.parse.urlparse(base)
if os.environ.get("SUBTENSOR_NETWORK") == "finney" and parsed.scheme != "https":
    raise SystemExit("[preflight] ERROR: Finney problem server must use HTTPS")
url = f"{base}/v3/challenges/lease"
request = urllib.request.Request(
    url,
    data=b"{}",
    headers={"Content-Type": "application/json"},
    method="POST",
)


class NoRedirects(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


opener = urllib.request.build_opener(NoRedirects)
try:
    with opener.open(request, timeout=20) as response:
        status = response.status
        date_header = response.headers.get("Date", "")
except urllib.error.HTTPError as error:
    status = error.code
    date_header = error.headers.get("Date", "")
except Exception as error:
    raise SystemExit(
        f"[preflight] ERROR: problem server reachability check failed: {error}; "
        "check network access and NTP"
    )
if status != 401:
    raise SystemExit(
        f"[preflight] ERROR: unsigned lease probe returned HTTP {status}; expected 401"
    )
if not date_header:
    raise SystemExit("[preflight] ERROR: server omitted Date; cannot verify clock")
try:
    server_time = email.utils.parsedate_to_datetime(date_header).timestamp()
except (TypeError, ValueError, OverflowError):
    raise SystemExit("[preflight] ERROR: server returned an invalid Date header")
skew = abs(time.time() - server_time)
if skew > 5:
    raise SystemExit(
        f"[preflight] ERROR: system clock differs from server by {skew:.1f}s; "
        "enable NTP before starting"
    )
print("[preflight] problem server reachable; unsigned lease rejected; clock synchronized")
PY
