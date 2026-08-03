# shellcheck shell=bash
# =============================================================================
#  mod: 20-torch-repair
#  requirements.txt / custom nodes / build deps can silently replace the CUDA
#  build of torch with a CPU-only one. This mod verifies and repairs it — at
#  build time (mod_apply, if ever added to container/build-mods.sh) AND before
#  every single launch (mod_prerun, which the entrypoint calls — the quick
#  guard against a custom node clobbering torch since the last start). A
#  broken torch means nothing on the GPU works, so mod_prerun's failure aborts
#  the launch loudly rather than degrading quietly.
# =============================================================================
# shellcheck disable=SC2034  # mod-contract flag; retained as docs (CLAUDE.md), no runner reads it now
MOD_CRITICAL=1
# shellcheck disable=SC2034  # mod-contract flag; retained as docs (CLAUDE.md), no runner reads it now
MOD_STREAM=1

mod_describe() {
  echo "torch CUDA 13 build verified/repaired (install-time + pre-launch guard)"
}

mod_apply() {
  if mod_verify; then
    echo "present"
    return 0
  fi
  repair_torch
  if mod_verify; then
    echo "applied repaired CUDA torch"
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
