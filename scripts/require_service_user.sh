#!/usr/bin/env bash
#
# Refuse to run the validator as root, and print the way out.
#
#   scripts/require_service_user.sh LABEL
#
# The validator runs miner code in containers, so it must run as an ordinary
# user in the docker group. When run as root this prints the exact commands
# that move an installation to such a user, with the real paths filled in.

set -euo pipefail

label="${1:-validator}"
if [[ "$(id -u)" -ne 0 && "$(id -g)" -ne 0 ]]; then
  exit 0
fi

user="${RLVR_SERVICE_USER:-hone}"
home="/home/${user}"
repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
wallets="${HOME:-/root}/.bittensor/wallets"

if [[ "${repo}" == "${home}/hone-subnet" ]]; then
  copy="  # The repository is already at ${home}/hone-subnet."
else
  copy="  mkdir -p ${home}/hone-subnet
  cp -a $(printf '%q' "${repo}")/. ${home}/hone-subnet/"
fi

cat >&2 <<MESSAGE
[${label}] ERROR: the validator must not run as root. It runs miner code in
containers, so it needs an ordinary user in the docker group.

Move this installation to one. First stop and remove the root-run validator
from your process manager. Then, as root:

  adduser --disabled-password --gecos "" ${user}
  usermod -aG docker ${user}
  mkdir -p ${home}/.bittensor/wallets
  cp -a $(printf '%q' "${wallets}")/. ${home}/.bittensor/wallets/
${copy}
  rm -rf ${home}/hone-subnet/.venv
  chown -R ${user}:${user} ${home}

Then as that user:

  sudo -iu ${user}
  cd ~/hone-subnet
  ./setup_validator.sh
  ./start_validator.sh

The copy keeps .env and data/validator_scores.json. The venv is rebuilt
because one made by root has root's paths in it.
MESSAGE
exit 1
