#!/usr/bin/env bash
# Build and verify the V3 grading image. Publishing requires --push REF.

set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${repo_root}"

push_ref=""
if [[ "${1:-}" == "--push" ]]; then
  [[ $# -eq 2 ]] || { echo "usage: $0 [--push REGISTRY/IMAGE:TAG]" >&2; exit 2; }
  push_ref="$2"
elif (( $# != 0 )); then
  echo "usage: $0 [--push REGISTRY/IMAGE:TAG]" >&2
  exit 2
fi

local_tag="hone-polyglot-sandbox:local"
docker build --pull -t "${local_tag}" docker/polyglot-sandbox
docker run --rm --network=none --read-only "${local_tag}" \
  bash --noprofile --norc -c \
  'command -v bash cargo g++ gcc git go java javac node npm python3 rustc timeout tsc >/dev/null'

if [[ -z "${push_ref}" ]]; then
  image_id="$(docker image inspect --format '{{.Id}}' "${local_tag}")"
  echo "Local image verified: ${image_id}"
  echo "A local image ID is not a fleet identity. Use --push to obtain a RepoDigest."
  exit 0
fi

docker tag "${local_tag}" "${push_ref}"
docker push "${push_ref}"
repo_name="${push_ref%:*}"
repo_digest="$(
  docker image inspect --format '{{range .RepoDigests}}{{println .}}{{end}}' \
    "${push_ref}" | grep -F "${repo_name}@" | head -n 1
)"
[[ "${repo_digest}" == *'@sha256:'* ]] || {
  echo "published image has no RepoDigest" >&2
  exit 1
}
echo "Published and verified: ${repo_digest}"
