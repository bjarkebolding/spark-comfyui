# shellcheck shell=bash
# =============================================================================
#  mod: 30-manager-config
#  Configures ComfyUI-Manager for the Spark: network_mode=personal_cloud
#  (required on a 0.0.0.0 listener), relaxed security_level, uv, file logging,
#  and a torch downgrade_blacklist. Not a source patch — a config-tree mod —
#  which is why it uses its own apply/verify rather than the py_patch_file
#  helper. Also removes a stale pip_auto_fix.list from before 2026-07 — see
#  configure.py's docstring for why that mechanism was retired (it crashed
#  Manager's own version parser on every launch for any CUDA torch build).
#  Writes: user/__manager/config.ini
# =============================================================================
mod_describe() {
  echo "ComfyUI-Manager config (personal_cloud, uv, downgrade_blacklist)"
}

mod_apply() {
  # uv backs 'use_uv = True' — much faster node dependency installs.
  [[ -x "$VENV_DIR/bin/uv" ]] || pip install -q uv >/dev/null 2>&1 || true
  INSTALL_DIR="$INSTALL_DIR" python3 "$MOD_DIR/configure.py" apply
}

mod_verify() {
  INSTALL_DIR="$INSTALL_DIR" python3 "$MOD_DIR/configure.py" verify
}
