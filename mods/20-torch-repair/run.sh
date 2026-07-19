# shellcheck shell=bash
# =============================================================================
#  mod: 20-torch-repair
#  requirements.txt / custom nodes / build deps can silently replace the CUDA
#  build of torch with a CPU-only one. This mod verifies and repairs it — at
#  install/update time (mod_apply) AND before every single 'run' (mod_prerun,
#  ComfyUI's quick guard against a custom node clobbering torch since the
#  last launch). Critical: a broken torch means nothing on the GPU works, so
#  failure aborts loudly rather than being reported as a soft skip. Streamed:
#  a repair reinstalls the full cu130 wheel set (>1 GB), so the user should
#  see progress, not a silent multi-minute pause.
# =============================================================================
# shellcheck disable=SC2034  # read by _run_mod's 'flags' verb in the main script
MOD_CRITICAL=1
# shellcheck disable=SC2034  # read by _run_mod's 'flags' verb in the main script
MOD_STREAM=1

mod_describe() {
  echo "torch CUDA 13 build verified/repaired (install-time + pre-launch guard)"
}

mod_apply() {
  if mod_verify; then
    mod_export "STATUS=present"
    return 0
  fi
  repair_torch
  if mod_verify; then
    mod_export "STATUS=applied repaired CUDA torch"
  else
    # still broken after repair — critical, this must abort loudly, and the
    # diag names the actual CUDA error (is_available() swallows it)
    torch_cuda_diag
    return 1
  fi
}

mod_prerun() {
  mod_verify && return 0
  repair_torch
  if mod_verify; then
    return 0
  fi
  torch_cuda_diag   # still broken -> critical abort before launch, with cause
  return 1
}

mod_verify() {
  python - <<'PY' >/dev/null 2>&1
import torch
assert (torch.version.cuda or "").startswith("13") and torch.cuda.is_available()
PY
}
