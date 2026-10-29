#!/usr/bin/env bash
# Local preview: ./scripts/serve.sh
set -euo pipefail
cd "$(dirname "$0")/.."

hugo_bin="${HUGO_BIN:-}"
if [[ -z "$hugo_bin" ]]; then
  if [[ -x .tools/hugo ]]; then
    hugo_bin=.tools/hugo
  else
    hugo_bin=hugo
  fi
fi

exec "$hugo_bin" server --baseURL http://localhost:1313/ --buildDrafts --buildFuture --disableFastRender "$@"
