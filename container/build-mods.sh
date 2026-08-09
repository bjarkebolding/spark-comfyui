#!/usr/bin/env bash
# The mod pass. Since 2026-08-08 this is the ONLY place mods run: a mod is a
# build-time modification to the image, full stop. The two that used to run at
# launch were never really mods (see mods/README.md); they are an
# ensure_torch_cuda function and container/manager-config.py now.
#
# The list below is deliberately explicit and NOT a glob over mods/*/, for the
# same reason lib/ is sourced by name in the entry point: a half-copied
# checkout must fail loudly, not silently bake an image with fewer patches
# applied. A globbed pass that quietly skipped unified-memory-free would ship
# the get_free_memory offload cliff back into the image and say nothing
# (golden rule 5). Adding a mod means adding it here, on purpose.
set -euo pipefail

log()  { printf '==> %s\n' "$*"; }
warn() { printf '[warn] %s\n' "$*"; }
info() { printf '[info] %s\n' "$*"; }
die()  { printf '[error] %s\n' "$*" >&2; exit 1; }

: "${INSTALL_DIR:?INSTALL_DIR must be set}" "${VENV_DIR:?VENV_DIR must be set}"

for m in setuptools-compat unified-memory-free; do
  (
    # shellcheck disable=SC1091
    source /opt/spark/mods/_lib/mod_common.sh
    MOD_DIR="/opt/spark/mods/$m"
    export MOD_DIR
    # shellcheck disable=SC1090
    source "$MOD_DIR/run.sh"
    log "mod $m: $(mod_describe)"
    status="$(mod_apply)" || die "mod $m apply failed: ${status:-<no output>}"
    echo "    $status"
    mod_verify || die "mod $m failed verification after apply"
  )
done
log "build-time mods applied"
