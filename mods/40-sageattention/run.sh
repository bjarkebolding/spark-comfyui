# shellcheck shell=bash
# =============================================================================
#  mod: 40-sageattention
#  Builds SageAttention natively for GB10 (TORCH_CUDA_ARCH_LIST="12.1+PTX")
#  and verifies it with a live multi-shape GPU kernel test before it's ever
#  considered active — see build_and_verify_sage in mod_common.sh. Pinned to
#  $SAGE_REF (not upstream's default branch — see where SAGE_REF is defined
#  in spark-comfyui.sh). Rebuilds only when needed: marker missing, local
#  checkout doesn't match the pin (e.g. SAGE_REF was deliberately bumped),
#  or torch was just upgraded (SPARK_TORCH_UPGRADED, set by cmd_update --torch).
#  Critical: a broken build means ComfyUI silently falls back to slower
#  attention — failure must abort loudly, not degrade quietly. Streamed: the
#  build takes 10-30 minutes and the user needs to see it happening.
# =============================================================================
# shellcheck disable=SC2034  # read by _run_mod's 'flags' verb in the main script
MOD_CRITICAL=1
# shellcheck disable=SC2034  # read by _run_mod's 'flags' verb in the main script
MOD_STREAM=1

mod_describe() {
  echo "SageAttention built natively for sm_121, live-kernel-verified"
}

mod_apply() {
  local rebuild=0
  if [[ ! -f "$SAGE_MARKER" ]]; then
    rebuild=1
  elif [[ -d "$SAGE_SRC/.git" ]]; then
    git -C "$SAGE_SRC" fetch -q origin
    local local_rev pinned_rev
    local_rev="$(git -C "$SAGE_SRC" rev-parse HEAD)"
    # An unresolvable SAGE_REF (bad pin, shallow history) must trigger a
    # rebuild, not silently pass — build_and_verify_sage's own checkout
    # will then fail loudly (critical mod) instead of this staying quiet.
    if pinned_rev="$(git -C "$SAGE_SRC" rev-parse "$SAGE_REF" 2>/dev/null)"; then
      [[ "$local_rev" != "$pinned_rev" ]] && rebuild=1
    else
      rebuild=1
    fi
  else
    rebuild=1
  fi
  [[ "${SPARK_TORCH_UPGRADED:-0}" == "1" ]] && rebuild=1

  if [[ "$rebuild" -eq 1 ]]; then
    build_and_verify_sage
    mod_export "SAGE_ACTION=rebuilt & verified"
    mod_export "STATUS=applied rebuilt & verified"
  else
    echo "SageAttention: OK — verified, no rebuild needed"
    mod_export "SAGE_ACTION=verified (no rebuild needed)"
    mod_export "STATUS=present verified, no rebuild needed"
  fi
}

mod_verify() {
  [[ -f "$SAGE_MARKER" ]] && sage_kernel_ok
}
