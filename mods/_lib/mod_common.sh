# shellcheck shell=bash
# =============================================================================
#  mod_common.sh — shared helpers for spark-comfyui mods
# =============================================================================
#  The mod contract, and nothing else. Sourced by container/build-mods.sh
#  during the image build, which is the only place a mod ever runs, and by
#  each mods/<name>/run.sh through it.
#
#  A mod's run.sh must define:
#    mod_describe -> one-line human description (echo)
#    mod_apply    -> apply the mod; echo a short status; return 0 on success
#    mod_verify   -> return 0 if the mod is currently active, 1 otherwise
#  and may rely on these, set by container/build-mods.sh:
#    INSTALL_DIR  -> ComfyUI checkout root
#    VENV_DIR     -> python virtualenv
#    MOD_DIR      -> this mod's own directory (for supporting files)
#
#  This file was three quarters GB10 probe-and-guard code until 2026-08-08,
#  which had nothing to do with mods and only lived here because this was the
#  one file that both shipped into the image and was sourced by the host
#  script. It is container/gb10.sh now. If you are looking for
#  sage_kernel_ok, ensure_torch_cuda, ensure_onnx_gpu, kitchen_nvfp4_ok or
#  torch_cuda_diag, that is where they are.
#
#  A `mod_prerun` hook, plus MOD_CRITICAL / MOD_STREAM, went in the same
#  change as the single mod that ran at launch rather than at build. Do not
#  reintroduce them: a launch step belongs in the entrypoint's pipeline,
#  beside the others.
#
#  NOT a standalone library: this file is a sourced fragment that assumes the
#  sourcing shell already provides the print helpers `log`/`warn`/`die` and
#  the globals `INSTALL_DIR`, `VENV_DIR`, `TORCH_INDEX` (ensure_torch_cuda) and
#  `ORT_WHEEL_URL` (ensure_onnx_gpu). Sourcing it anywhere else (tests, other
#  scripts) requires stubbing those first.
# =============================================================================

# Marker embedded in patched files so apply/verify are idempotent.
mod_marker() { echo "# spark-comfyui:${1:?mod_marker needs a tag}"; }

# Idempotently transform a Python source file with a python snippet.
#   py_patch_file <relpath-under-INSTALL_DIR> <tag> <python-transform-file>
# The transform file is a python script reading the source on stdin and
# writing the patched source to stdout; it must be a no-op-returning-input
# when it cannot find its anchor. Handles the marker check, a backup
# (<file>.spark-orig) refreshed on every apply, and reports one of:
# applied | present | skipped:<why>.
py_patch_file() {
  local rel="$1" tag="$2" transform="$3"
  local path="$INSTALL_DIR/$rel"
  local marker; marker="$(mod_marker "$tag")"
  if [[ ! -f "$path" ]]; then echo "skipped:missing $rel"; return 1; fi
  if grep -qF "$marker" "$path"; then echo "present"; return 0; fi

  local out; out="$(MARKER="$marker" python3 "$transform" < "$path" 2>/dev/null)" || {
    echo "skipped:transform-error"; return 1; }
  if [[ -z "$out" ]] || [[ "$out" == "$(cat "$path")" ]]; then
    echo "skipped:anchor-not-found"; return 1
  fi
  # Refresh the backup on EVERY apply, not just the first: mods re-apply
  # after each git pull, so a once-only backup goes stale and the revert
  # below would restore months-old upstream code over the current file.
  # Safe here — the marker check above already returned, so $path is
  # guaranteed to be current-upstream, unpatched content.
  cp -f "$path" "$path.spark-orig"
  printf '%s' "$out" > "$path"
  # Guarantee we never leave invalid Python behind.
  if ! python3 -c "import ast,sys; ast.parse(open('$path',encoding='utf-8').read())" 2>/dev/null; then
    cp -f "$path.spark-orig" "$path"
    echo "skipped:would-break-python"; return 1
  fi
  echo "applied"
}

# verify helper: is the marker present in a given file?
py_marker_present() {
  local rel="$1" tag="$2"
  local marker; marker="$(mod_marker "$tag")"
  [[ -f "$INSTALL_DIR/$rel" ]] && grep -qF "$marker" "$INSTALL_DIR/$rel"
}
