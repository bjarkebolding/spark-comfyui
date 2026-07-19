#!/usr/bin/env bash
# Container entrypoint: the run-time half of the mod system. The image is
# immutable, so anything that needs the GPU (the live kernel gates), the
# bind-mounted user content (Manager config), or the user's installed custom
# nodes runs here, before every launch.
set -euo pipefail

log()  { printf '\n\033[1;32m==> %s\033[0m\n' "$*"; }
warn() { printf '\033[1;33m[warn] %s\033[0m\n' "$*"; }
info() { printf '\033[1;36m[info] %s\033[0m\n' "$*"; }
die()  { printf '\033[1;31m[error] %s\033[0m\n' "$*" >&2; exit 1; }

: "${INSTALL_DIR:?}" "${VENV_DIR:?}"
cd "$INSTALL_DIR"

# 1. Custom-node requirements. Manager clones nodes into the mounted
#    custom_nodes dir, but their pip deps land in the container layer and
#    vanish on recreation, so re-install on every start. The pip cache
#    volume makes this cheap after the first pass. A failing node is a
#    warning, not a dead server — same policy as restore.
log "Custom-node requirements"
shopt -s nullglob
for req in "$INSTALL_DIR"/custom_nodes/*/requirements.txt; do
  node="$(basename "$(dirname "$req")")"
  info "custom node $node: installing requirements"
  pip install -q -r "$req" </dev/null \
    || warn "pip install for custom node $node failed — the node may not load"
done
shopt -u nullglob

# shellcheck disable=SC1091
source /opt/spark/mods/_lib/mod_common.sh

# 2. Torch guard AFTER the node installs — that is exactly when torch gets
#    clobbered. Same mod, same diag output as the native pre-launch pass.
log "Torch CUDA guard"
(
  MOD_DIR=/opt/spark/mods/20-torch-repair
  export MOD_DIR
  # shellcheck disable=SC1091
  source "$MOD_DIR/run.sh"
  mod_prerun
) || die "torch CUDA check failed — see the diag lines above. If the wheel
set itself is broken, rebuild the image: spark-comfyui.sh container build"

# 3. SageAttention live kernel gate. The image build compiled it blind (no
#    GPU exists at build time); this is where golden rule 3 now lives.
log "SageAttention kernel gate"
if sage_kernel_ok; then
  sage_flag=(--use-sage-attention)
  info "SageAttention enabled (kernel verified live)"
else
  die "SageAttention kernel FAILED on this GPU — refusing to launch degraded.
Rebuild the image: spark-comfyui.sh container build"
fi

# 4. Manager config lives under the bind-mounted user/ dir, so it must be
#    (re-)asserted at run time, not baked into the image.
log "Manager config"
python /opt/spark/mods/30-manager-config/configure.py apply \
  || warn "Manager config apply failed — continuing, Manager may be gated"

# 5. Launch. Flags mirror the native cmd_run; exposure is controlled by the
#    host's port mapping, so --listen 0.0.0.0 here is scoped to the
#    container's own network namespace.
extra_flags=()
if [[ "${SPARK_BF16:-1}" == "1" ]]; then
  extra_flags+=(--bf16-unet --bf16-vae --bf16-text-enc)
fi
if [[ "${SPARK_STATIC_VRAM:-0}" == "1" ]]; then
  extra_flags+=(--disable-dynamic-vram)
fi

log "Launching ComfyUI"
exec python main.py \
  --listen 0.0.0.0 \
  --port 8188 \
  --enable-manager \
  --preview-method auto \
  --disable-pinned-memory \
  "${sage_flag[@]}" \
  "${extra_flags[@]}" \
  "$@"
