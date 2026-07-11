# shellcheck shell=bash
# =============================================================================
#  mod_common.sh — shared helpers for spark-comfyui mods
# =============================================================================
#  Sourced by each mods/<name>/run.sh AND by the main script itself at
#  startup, so both the mod system and cmd_run/cmd_doctor/cmd_rollback share
#  one copy of the GB10 venv-package helpers (need_nvcc, sage_kernel_ok,
#  onnx_gpu_ok, repair_torch, etc.) instead of two.
#
#  A mod's run.sh must define these shell functions:
#    mod_describe   -> one-line human description (echo)
#    mod_apply      -> apply the mod; echo a short status; return 0 on success
#    mod_verify     -> return 0 if the mod is currently active, 1 otherwise
#  and may optionally define:
#    mod_prerun     -> runs before every `run` (not just install/update).
#                      Absence is a silent no-op. Only used by mods that need
#                      a cheap pre-launch guard (e.g. 20-torch-repair).
#
#  A mod's run.sh may optionally set these top-level variables:
#    MOD_CRITICAL=1 -> a nonzero exit from mod_apply/mod_prerun aborts the
#                      whole script (die) instead of being reported as a soft
#                      "skipped:error". Use for steps whose failure means the
#                      install/launch is genuinely broken (torch has no CUDA,
#                      the GPU kernel doesn't work) — NOT for optional source
#                      patches, which should stay soft so one broken mod
#                      never takes down an otherwise-working install.
#    MOD_STREAM=1   -> mod_apply/mod_prerun output streams live to the
#                      terminal instead of being buffered until it finishes.
#                      Required for anything that can take more than a few
#                      seconds (a build, a large download) — the buffered
#                      default would otherwise hide all progress until the
#                      subshell exits. A streamed mod MUST report its status
#                      via `mod_export STATUS=<word>` (see below) instead of
#                      an echoed last line, since stdout is no longer
#                      captured.
#
#  It may rely on these environment variables, exported by the main script:
#    INSTALL_DIR    -> ComfyUI checkout root
#    VENV_DIR       -> python virtualenv
#    MOD_DIR        -> this mod's own directory (for supporting files)
#    MOD_STATE_FILE -> path a mod can append KEY=value lines to via
#                      mod_export; read back into the caller's scope after
#                      the mod returns (used to hand state like SAGE_ACTION/
#                      ORT_STATE back to cmd_update's summary).
#    MOD_MARKER     -> the canonical "# spark-comfyui:<tag>" marker string
#
#  NOT a standalone library: this file is a sourced fragment that assumes the
#  sourcing shell (spark-comfyui.sh, or a mod subshell inheriting from it)
#  already provides the print helpers `log`/`warn`/`die` and the globals
#  `INSTALL_DIR`, `VENV_DIR`, `SAGE_SRC`, `SAGE_REF`, `SAGE_MARKER`,
#  `TORCH_INDEX`, `ORT_WHEEL_URL`. Sourcing it anywhere else (tests, other
#  scripts) requires stubbing those first.
# =============================================================================

# Marker embedded in patched files so apply/verify are idempotent.
mod_marker() { echo "# spark-comfyui:${1:?mod_marker needs a tag}"; }

# A mod writes KEY=value pairs here to hand state back to its caller (the
# main script reads $MOD_STATE_FILE back after the mod's subshell returns).
# Streamed mods (MOD_STREAM=1) also use this for their final status:
#   mod_export STATUS=applied
mod_export() {
  [[ -n "${MOD_STATE_FILE:-}" ]] || return 0
  echo "$1" >> "$MOD_STATE_FILE"
}

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

# =============================================================================
#  GB10 venv-package helpers (shared by the main script AND the mods that
#  wrap them: 05-setuptools-compat, 20-torch-repair, 40-sageattention,
#  50-onnxruntime-gpu). Logic is unchanged from when these lived directly in
#  spark-comfyui.sh — only their location and how they're invoked in sequence
#  changed. cmd_run's shadow-detection, cmd_rollback, and cmd_doctor's
#  diagnostics all call these directly too, same as before.
# =============================================================================

# sm_121 requires ptxas from CUDA >= 13.0. Version parse is the reliable
# test: CUDA 13.0's 'ptxas --help' does NOT enumerate sm_121 in its help
# text despite fully supporting it, so help-grepping gives false positives.
ptxas_ge_13() {
  local ver
  ver="$(ptxas --version 2>/dev/null | grep -o 'release [0-9]*\.[0-9]*' | head -1 | awk '{print $2}')"
  [[ -n "$ver" && "${ver%%.*}" -ge 13 ]]
}

need_nvcc() {
  if ! command -v nvcc >/dev/null 2>&1; then
    if [[ -x /usr/local/cuda/bin/nvcc ]]; then
      export PATH="/usr/local/cuda/bin:$PATH"
      export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
    else
      die "nvcc not found — SageAttention must be compiled from source.
Install the CUDA 13 toolkit (e.g. sudo apt-get install -y cuda-toolkit-13-0),
then re-run. Completed steps are skipped on re-run."
    fi
  fi
  export CUDA_HOME="${CUDA_HOME:-$(dirname "$(dirname "$(command -v nvcc)")")}"
  # The GB10 needs ptxas from CUDA >= 13.0. NOTE: don't grep 'ptxas --help'
  # for sm_121 — CUDA 13.0's help text doesn't enumerate it despite fully
  # supporting it (confirmed false positive). Parse the release version.
  if command -v ptxas >/dev/null 2>&1 && ! ptxas_ge_13; then
    warn "ptxas on PATH is older than CUDA 13.0 — it cannot target sm_121.
  offender: $(command -v ptxas)  ($(ptxas --version 2>/dev/null | tail -1))
  nvcc in use: $(command -v nvcc)  ($(nvcc --version 2>/dev/null | grep -o 'release [0-9.]*'))
This is the most common cause of 'no kernel image' on GB10. If nvcc above is
release 13.x the build will use its own sibling ptxas and likely still
succeed (verification will confirm). Otherwise put CUDA 13 first on PATH:
  export PATH=/usr/local/cuda-13.0/bin:\$PATH"
  fi
}

# Detect the SageAttention "pip shadowing" drift: a later `pip install
# sageattention` or a custom node dep can silently overwrite the local
# sm_121 build with a PyPI wheel that has no GB10 kernel — reintroducing the
# exact "no kernel image" failure invisibly. Returns 0 if the live kernel
# still runs, 1 if it's broken/shadowed. Cheap enough to gate launches on.
sage_kernel_ok() {
  python - <<'PY' >/dev/null 2>&1
import torch
from sageattention import sageattn
q = torch.randn(1, 8, 1024, 128, dtype=torch.float16, device="cuda")
o = sageattn(q, q, q, tensor_layout="HND")
torch.cuda.synchronize()
assert o.shape == q.shape and torch.isfinite(o).all()
PY
}

# comfy-kitchen NVFP4 live gate. ComfyUI auto-selects comfy-kitchen's
# fastest backend per call and quietly uses the pure-PyTorch 'eager' path
# when the native CUDA backend can't serve it — quantized (NVFP4/FP8)
# models keep working, just massively slower, with nothing surfaced.
# use_backend() genuinely enforces (raises BackendNotFoundError instead of
# falling back — verified live on GB10, 2026-07), so success under forcing
# proves the CUDA backend's kernels actually ran. The cosine check against
# a bf16 reference guards against garbage output, not just crashes (NVFP4
# is coarse; healthy runs measure ~0.99).
kitchen_nvfp4_ok() {
  python - <<'PY' >/dev/null 2>&1
import torch
import comfy_kitchen as ck
M, N, K = 128, 256, 512
a = torch.randn(M, K, dtype=torch.bfloat16, device="cuda")
b = torch.randn(N, K, dtype=torch.bfloat16, device="cuda")
FP4_MAX, FP8_MAX = 6.0, 448.0
sa = (a.abs().amax().float() / (FP4_MAX * FP8_MAX)).clamp(min=1e-8)
sb = (b.abs().amax().float() / (FP4_MAX * FP8_MAX)).clamp(min=1e-8)
with ck.use_backend("cuda"):
    qa, bsa = ck.quantize_nvfp4(a, sa)
    qb, bsb = ck.quantize_nvfp4(b, sb)
    y = ck.scaled_mm_nvfp4(qa, qb, sa, sb, bsa, bsb, out_dtype=torch.bfloat16)
torch.cuda.synchronize()
assert torch.isfinite(y).all()
ref = a.float() @ b.float().T
cos = torch.nn.functional.cosine_similarity(y.float().flatten(), ref.flatten(), dim=0)
assert cos > 0.98, f"cosine {cos.item():.4f}"
PY
}

# Count per-call SageAttention runtime fallbacks in a ComfyUI log. ComfyUI
# catches SageAttention exceptions per call and silently uses PyTorch
# attention instead (comfy/ldm/modules/attention.py), so the build-time
# kernel test can't see these. Prints "<total> <benign>", where benign =
# 'Unsupported head_dim' cases (a model-architecture limit, not a fault);
# total > benign means something real is broken (e.g. Triton's JIT shim
# failing from a missing python3.X-dev — up to ~18x slower sampling).
sage_fallback_counts() {
  local log="$1" total benign
  total="$(grep -c 'Error running sage attention' "$log" 2>/dev/null || true)"
  benign="$(grep -c 'Error running sage attention.*Unsupported head_dim' "$log" 2>/dev/null || true)"
  echo "${total:-0} ${benign:-0}"
}

# ONNX Runtime GPU check. get_available_providers() is the RELIABLE detector;
# startup-log GPU-discovery warnings are misleading and can appear even when
# the GPU provider works fine.
onnx_gpu_ok() {
  python - <<'PY' >/dev/null 2>&1
import onnxruntime as ort
assert "CUDAExecutionProvider" in ort.get_available_providers()
PY
}

# DWPose / ControlNet preprocessors run on onnxruntime. PyPI ships no GPU
# wheel for aarch64+cu13, so without the community sm_121 wheel they silently
# fall back to CPU — a large hidden slowdown. Also guards the shadow trap:
# a later 'pip install onnxruntime' (e.g. pulled in by a custom node)
# overwrites the GPU wheel via the shared import path with no pip conflict.
ensure_onnx_gpu() {
  ORT_STATE="unknown"
  local pyver
  pyver="$(python -c 'import sys; print(f"{sys.version_info.major}{sys.version_info.minor}")')"
  if [[ "$pyver" != "312" ]]; then
    warn "GPU onnxruntime wheel is cp312; this venv is Python ${pyver:0:1}.${pyver:1}.
Skipping — preprocessor nodes (DWPose etc.) will use CPU onnxruntime."
    ORT_STATE="skipped (non-3.12 venv)"
    return 0
  fi
  if onnx_gpu_ok; then
    echo "onnxruntime: OK — GPU provider live"
    ORT_STATE="GPU provider live"
    return 0
  fi
  log "Installing sm_121 GPU onnxruntime (community wheel)"
  # Remove PyPI CPU dists that shadow the same import path first.
  pip uninstall -y onnxruntime onnxruntime-gpu >/dev/null 2>&1 || true
  pip install "$ORT_WHEEL_URL"
  if onnx_gpu_ok; then
    echo "onnxruntime CUDAExecutionProvider: live (preprocessors on GPU)"
    ORT_STATE="GPU provider live (installed)"
  else
    warn "onnxruntime installed but CUDA provider is NOT available — DWPose
etc. will fall back to CPU. Ensure cuDNN 9.x is installed system-wide
(DGX OS ships it; otherwise: sudo apt-get install -y libcudnn9-cuda-13)."
    # shellcheck disable=SC2034  # read by mods/50-onnxruntime-gpu/run.sh
    ORT_STATE="CPU FALLBACK — see warning"
  fi
}

# requirements.txt / custom nodes / build deps can silently replace the CUDA
# build of torch with a CPU-only one. Verify and repair.
# torch pins a setuptools upper bound (e.g. <82 for torch 2.12); a blanket
# 'pip install -U setuptools' — or a custom node's requirements — can break
# it, which then breaks source builds (SageAttention uses torch's setuptools
# machinery). Read torch's OWN declared constraint from its metadata so this
# stays correct across torch versions, and upgrade/downgrade within it.
# No-op when already conformant; harmless "latest" upgrade if torch absent.
ensure_setuptools_compat() {
  local spec
  spec="$(python - <<'PY'
try:
    import importlib.metadata as md
    for r in (md.requires("torch") or []):
        r = r.split(";")[0].strip()
        if r.startswith("setuptools"):
            print(r); break
    else:
        print("setuptools")
except Exception:
    print("setuptools")
PY
)"
  pip install --upgrade "$spec" >/dev/null
}

repair_torch() {
  local cuda_ok
  cuda_ok="$(python - <<'PY'
import torch
print("ok" if (torch.version.cuda or "").startswith("13") and torch.cuda.is_available() else "bad")
PY
)"
  if [[ "$cuda_ok" != "ok" ]]; then
    warn "torch lost CUDA 13 support — reinstalling cu130 wheels"
    pip install --force-reinstall torch torchvision torchaudio --index-url "$TORCH_INDEX"
  fi
}

# Build SageAttention natively for GB10 and verify with a LIVE kernel test.
# A broken build silently falls back to PyTorch attention inside ComfyUI,
# so we fail loudly instead of shipping a silently-degraded install.
build_and_verify_sage() {
  log "Building SageAttention natively for GB10 (can take 10-30 min)"
  rm -f "$SAGE_MARKER"
  need_nvcc
  pip install ninja packaging >/dev/null

  # Pinned to $SAGE_REF, not thu-ml/SageAttention's default branch — see the
  # comment where SAGE_REF is defined in spark-comfyui.sh for why.
  if [[ -d "$SAGE_SRC/.git" ]]; then
    git -C "$SAGE_SRC" fetch -q origin
  else
    git clone -q https://github.com/thu-ml/SageAttention.git "$SAGE_SRC"
    git -C "$SAGE_SRC" fetch -q origin
  fi
  git -C "$SAGE_SRC" checkout -q "$SAGE_REF"

  # GB10 identifies as sm_121, which torch/many toolchains don't list as a
  # target yet. The correct, field-tested recipe (Triton #10331, NVIDIA
  # forums) is NATIVE sm_121 PLUS PTX: the "+PTX" is essential — it embeds
  # PTX so the driver can JIT if a matching cubin is ever missing. Building
  # for "12.0" alone produces sm_120 cubins with NO PTX fallback, which on
  # GB10 fails at runtime with "no kernel image is available". We also point
  # the build at CUDA 13's sm_121-aware ptxas.
  need_nvcc
  export TRITON_PTXAS_PATH="${CUDA_HOME:-/usr/local/cuda}/bin/ptxas"
  ( cd "$SAGE_SRC" && \
    TORCH_CUDA_ARCH_LIST="12.1+PTX" MAX_JOBS="$(nproc)" \
    pip install --no-build-isolation --no-deps . ) \
    || die "SageAttention compilation failed. Ensure python3-dev and the
CUDA 13 toolkit (with sm_121-aware ptxas) are installed, then re-run."

  repair_torch

  # Verify across MULTIPLE real diffusion-shaped inputs, not one tiny tensor.
  # The earlier single-shape test could pass while ComfyUI's actual shapes
  # hit the "no kernel image" path. This exercises head_dim 64 and 128 and
  # a large token count, and forces a real GPU sync to surface async errors.
  log "Verifying SageAttention on realistic shapes (no silent fallback)"
  if CUDA_LAUNCH_BLOCKING=1 python - <<'PY'
import torch
from sageattention import sageattn
shapes = [(2,10,4096,64), (1,24,4608,128), (1,16,8192,64)]
for b,h,n,d in shapes:
    q = torch.randn(b,h,n,d, dtype=torch.float16, device="cuda")
    o = sageattn(q, q, q, tensor_layout="HND")
    torch.cuda.synchronize()
    assert o.shape == q.shape, f"bad shape for {(b,h,n,d)}"
    assert torch.isfinite(o).all(), f"non-finite output for {(b,h,n,d)}"
print("SageAttention verified on", len(shapes), "shapes: OK")
PY
  then
    touch "$SAGE_MARKER"
  else
    die "SageAttention compiled but FAILED the runtime kernel test.
If you see 'no kernel image is available', the build produced no sm_121-
compatible kernel. Confirm: nvcc is from CUDA 13 (nvcc --version) and
ptxas is >= 13.0 (ptxas --version). Then re-run."
  fi
}
