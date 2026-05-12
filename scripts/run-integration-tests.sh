#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

if [[ -f ".env" ]]; then
  set -a
  # shellcheck disable=SC1091
  source ".env"
  set +a
else
  echo "warning: .env not found; only already-exported environment variables will be used" >&2
fi

if [[ "$#" -gt 0 ]]; then
  exec uv run python -m pytest "$@"
fi

exec uv run python -m pytest \
  tests/test_lambdadb_adapter.py \
  tests/test_qdrant_adapter.py \
  tests/test_pinecone_adapter.py
