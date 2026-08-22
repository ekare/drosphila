#!/usr/bin/env bash
set -euo pipefail

if [[ $# -eq 0 ]]; then
  echo "usage: $0 command [args...]" >&2
  exit 2
fi

# GPU selection is intentionally left to the caller through the standard
# CUDA_VISIBLE_DEVICES environment variable.
exec "$@"
