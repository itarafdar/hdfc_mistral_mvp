#!/usr/bin/env bash
set -euo pipefail
if [[ $# -lt 1 || $# -gt 2 ]]; then
  echo "Usage: $0 IMAGE [PLATFORM]" >&2
  exit 2
fi
image=$1
platform=${2:-linux/amd64}
repo_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
docker buildx build --platform "$platform" --tag "$image" --load "$repo_dir"
