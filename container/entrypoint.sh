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

# Package installs prefer uv, falling back to pip. This matters because 'run'
# is --rm: the custom-node requirements pass below happens on EVERY launch, and
# uv resolves and downloads in parallel against the same cache volume pip uses.
# uv arrives with ComfyUI-Manager's own requirements, so it is present but not
# guaranteed — a probe, not an assumption. pip stays the fallback for a second
# reason: uv's resolver is stricter, and a node with sloppy pins that pip
# tolerates must not be what stops the server from starting.
UV="$(command -v uv || true)"
INSTALLER="pip"
[[ -n "$UV" ]] && INSTALLER="uv"
py_install() {
  if [[ -n "$UV" ]]; then
    "$UV" pip install -q --python "$VENV_DIR/bin/python" "$@" </dev/null && return 0
    warn "uv could not install this set — retrying the same set with pip"
  fi
  pip install -q "$@" </dev/null
}

# 1. Custom-node requirements. Manager clones nodes into the mounted
#    custom_nodes dir, but their pip deps land in the container's writable
#    layer and vanish on recreation, so a fresh container must install them.
#    A RESTARTED container (service restart policy, reboots) still has
#    them; the marker lives in /tmp — the same writable layer, so it dies
#    exactly when the installed packages die and can never wrongly skip on
#    a fresh container. All requirement files go to pip in one invocation
#    (one resolver pass instead of one per node); a failure falls back to
#    per-node installs to isolate the culprit. A failing node is a warning,
#    not a dead server — same policy as restore.
log "Custom-node requirements"
shopt -s nullglob
req_files=("$INSTALL_DIR"/custom_nodes/*/requirements.txt)
shopt -u nullglob
marker=/tmp/spark-node-reqs.sha256
if (( ${#req_files[@]} == 0 )); then
  info "no custom nodes with requirements"
elif [[ -f "$marker" ]] \
     && [[ "$(sha256sum "${req_files[@]}")" == "$(cat "$marker")" ]]; then
  info "requirements unchanged since this container's last start — skipping"
else
  rm -f "$marker"
  info "installing requirements for ${#req_files[@]} custom node(s) (via $INSTALLER)"
  pip_args=()
  for req in "${req_files[@]}"; do pip_args+=(-r "$req"); done
  if py_install "${pip_args[@]}"; then
    sha256sum "${req_files[@]}" > "$marker"
  else
    warn "combined install failed — retrying per node to isolate it"
    req_fail=0
    for req in "${req_files[@]}"; do
      node="$(basename "$(dirname "$req")")"
      info "custom node $node: installing requirements"
      py_install -r "$req" \
        || { warn "installing requirements for custom node $node failed — the node may not load"; req_fail=1; }
    done
    # No marker after a failure: every start retries until the set heals.
    (( req_fail )) || sha256sum "${req_files[@]}" > "$marker"
  fi
fi

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
set itself is broken, rebuild the image: spark-comfyui.sh update"

# 3. onnxruntime GPU guard, same placement logic as the torch guard: a custom
#    node that pip installs onnxruntime shadows the sm_121 GPU wheel through
#    the shared import path, with no pip conflict to notice. DWPose and the
#    ControlNet preprocessors then silently run on CPU. The healthy case is
#    one provider query (~0.06s measured), so this is cheap on every start;
#    only a genuinely shadowed install pays the reinstall, which is a 57 MB
#    wheel and took ~3s when tested against a real shadow.
#    Warn, never die, unlike torch above: CPU onnxruntime is a slowdown, not
#    a broken server, so it must not stop a launch. The '||' also suspends
#    set -e inside the function, which is what keeps a failed pip from
#    killing the entrypoint.
log "onnxruntime GPU guard"
ensure_onnx_gpu \
  || warn "onnxruntime GPU guard did not complete — DWPose and ControlNet
preprocessors may fall back to CPU this session (spark-comfyui.sh doctor re-checks)"

# 4. SageAttention live kernel gate. The image build compiled it blind (no
#    GPU exists at build time); this is where golden rule 3 now lives.
log "SageAttention kernel gate"
if sage_kernel_ok; then
  sage_flag=(--use-sage-attention)
  info "SageAttention enabled (kernel verified live)"
else
  die "SageAttention kernel FAILED on this GPU — refusing to launch degraded.
Rebuild the image: spark-comfyui.sh update"
fi

# 5. Manager config lives under the bind-mounted user/ dir, so it must be
#    (re-)asserted at run time, not baked into the image.
log "Manager config"
python /opt/spark/mods/30-manager-config/configure.py apply \
  || warn "Manager config apply failed — continuing, Manager may be gated"

# 6. Launch. Flags mirror the native cmd_run; exposure is controlled by the
#    host's port mapping, so --listen 0.0.0.0 here is scoped to the
#    container's own network namespace.
extra_flags=()
# bf16 is the GB10 fast path. The VAE half is split out because LTX-2.3's
# audio VAE never casts the input waveform to the VAE dtype, so --bf16-vae
# fails it with "Input type (float) and bias type (c10::BFloat16)". That hits
# the stock LoadAudio node too, i.e. every LTX-2.3 audio workflow. SPARK_BF16_VAE=0
# drops that one flag and keeps the unet and text-encoder speedups.
if [[ "${SPARK_BF16:-1}" == "1" ]]; then
  extra_flags+=(--bf16-unet --bf16-text-enc)
  if [[ "${SPARK_BF16_VAE:-1}" == "1" ]]; then
    extra_flags+=(--bf16-vae)
  else
    info "bf16 VAE disabled (SPARK_BF16_VAE=0) — needed for LTX-2.3 audio workflows"
  fi
fi
if [[ "${SPARK_STATIC_VRAM:-0}" == "1" ]]; then
  extra_flags+=(--disable-dynamic-vram)
fi
# Optional VRAM reserve (GB) kept free for the OS / a co-resident CUDA
# process. Unset by default (ComfyUI's own default headroom applies); set
# SPARK_RESERVE_VRAM at run time to harden against the overcommit freeze when
# pushing large models. Numeric guard so a stray value can't inject flags.
if [[ "${SPARK_RESERVE_VRAM:-}" =~ ^[0-9]+([.][0-9]+)?$ ]]; then
  extra_flags+=(--reserve-vram "$SPARK_RESERVE_VRAM")
elif [[ -n "${SPARK_RESERVE_VRAM:-}" ]]; then
  warn "ignoring SPARK_RESERVE_VRAM='$SPARK_RESERVE_VRAM' (not a number of GB)"
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
