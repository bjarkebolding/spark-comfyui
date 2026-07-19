#!/usr/bin/env bash
# Build-time mod pass for the container image. Runs only the mods that need
# neither a GPU nor user content: 05-setuptools-compat (venv package) and
# 10-unified-memory-free (source patch). Mods 20/40/50 are GPU-gated and run
# their verification in the entrypoint; 30 writes to the bind-mounted user
# dir and runs there too. Reuses the exact mod contract and helpers the
# native path uses — the recipe is defined once, in mods/.
set -euo pipefail

log()  { printf '==> %s\n' "$*"; }
warn() { printf '[warn] %s\n' "$*"; }
info() { printf '[info] %s\n' "$*"; }
die()  { printf '[error] %s\n' "$*" >&2; exit 1; }

: "${INSTALL_DIR:?INSTALL_DIR must be set}" "${VENV_DIR:?VENV_DIR must be set}"

for m in 05-setuptools-compat 10-unified-memory-free; do
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
