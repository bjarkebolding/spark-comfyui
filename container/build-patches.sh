#!/usr/bin/env bash
# Image-build half of the patch list: merge the user's comfyui-patches.list
# entries (pr:<N> | branch:<name> | remote:<url> <branch>) on top of the
# pinned ComfyUI commit onto a spark-patched branch, mirroring the native
# apply_patches. Runs inside the image build: no GPU, no prompts, and a
# merge conflict fails the build loudly (a patch that no longer applies
# must never ship silently unpatched).
set -euo pipefail

LIST=/opt/spark/comfyui-patches.list
cd /opt/ComfyUI

if [[ ! -f "$LIST" ]]; then
  echo "no patch list — plain upstream at the pinned commit"
  exit 0
fi

export GIT_TERMINAL_PROMPT=0
git config user.email "spark-comfyui@localhost"
git config user.name  "spark-comfyui build"

applied=0
while IFS= read -r line || [[ -n "$line" ]]; do
  line="${line%%#*}"
  line="${line#"${line%%[![:space:]]*}"}"
  line="${line%"${line##*[![:space:]]}"}"
  [[ -z "$line" ]] && continue
  if [[ "$applied" == 0 ]]; then
    git checkout -q -B spark-patched
  fi
  case "$line" in
    pr:*)
      n="${line#pr:}"
      timeout 300 git fetch -q origin "pull/$n/head" </dev/null
      git merge -q --no-edit FETCH_HEAD ;;
    branch:*)
      b="${line#branch:}"
      timeout 300 git fetch -q origin "$b" </dev/null
      git merge -q --no-edit FETCH_HEAD ;;
    remote:*)
      rest="${line#remote:}"
      url="${rest%% *}" b="${rest#* }"
      [[ "$url" != "$b" ]] || { echo "remote: entry needs URL and branch: $line" >&2; exit 1; }
      timeout 300 git fetch -q "$url" "$b" </dev/null
      git merge -q --no-edit FETCH_HEAD ;;
    *)
      echo "unknown patch-list entry: $line" >&2
      exit 1 ;;
  esac
  echo "merged: $line"
  applied=$((applied+1))
done < "$LIST"

if [[ "$applied" == 0 ]]; then
  echo "patch list has no active entries — plain upstream at the pinned commit"
else
  echo "spark-patched branch built: $applied entr$( [[ $applied == 1 ]] && echo y || echo ies) on top of the pinned commit"
fi
