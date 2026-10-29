#!/usr/bin/env bash
# Create a new post: ./scripts/new.sh my-post-slug
set -euo pipefail
cd "$(dirname "$0")/.."

slug="${1:?usage: ./scripts/new.sh my-post-slug}"
hugo_bin="${HUGO_BIN:-}"
if [[ -z "$hugo_bin" ]]; then
  if [[ -x .tools/hugo ]]; then
    hugo_bin=.tools/hugo
  else
    hugo_bin=hugo
  fi
fi

"$hugo_bin" new content "posts/${slug}.md"
