# shellcheck shell=bash
# =============================================================================
#  mod_common.sh — shared helpers for spark-comfyui mods
# =============================================================================
#  Sourced by each mods/<name>/run.sh, by container/build-mods.sh during the
#  image build, and by the entrypoint and doctor at run time, so one copy of
#  the GB10 helpers (sage_kernel_ok, onnx_gpu_ok, kitchen_nvfp4_ok,
#  repair_torch, torch_cuda_diag, the patch helpers) serves all of them.
#  The host script sources it too, for sage_fallback_counts in doctor.
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
#  MOD_CRITICAL / MOD_STREAM are vestigial: they were read by the native
#  runner, which was deleted with the container cut. Mod 20 still declares
#  them purely as documentation of intent. Nothing reads them; do not build
#  new behavior on them.
#
#  It may rely on these environment variables, set by whoever runs the mod
#  (container/build-mods.sh at image build, the entrypoint at launch):
#    INSTALL_DIR    -> ComfyUI checkout root
#    VENV_DIR       -> python virtualenv
#    MOD_DIR        -> this mod's own directory (for supporting files)
#
#  NOT a standalone library: this file is a sourced fragment that assumes the
#  sourcing shell already provides the print helpers `log`/`warn`/`die` and
#  the globals `INSTALL_DIR`, `VENV_DIR`, `TORCH_INDEX` (repair_torch) and
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

# =============================================================================
#  GB10 helpers, shared by the mods, container/build-mods.sh, the entrypoint
#  and doctor. The SageAttention COMPILE is not here: it is a Dockerfile
#  stage (container/Dockerfile, `FROM torch AS sage`), pinned to SAGE_REF and
#  built with TORCH_CUDA_ARCH_LIST=12.1+PTX. What lives here is the live
#  kernel gate that decides whether that build is usable on this GPU, which
#  is what golden rule 3 actually requires.
# =============================================================================

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
#
# NOT CURRENTLY WIRED. The Dockerfile installs the pinned wheel at build
# time, and doctor DETECTS the shadow trap via onnx_gpu_ok, but nothing
# REPAIRS it the way mod 20 repairs torch on every start. This is the repair
# half, kept ready for that decision (see the roadmap in CLAUDE.md). It is
# the one function here with no caller; that is deliberate, not an oversight.
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
    # shellcheck disable=SC2034  # status for a caller; see the wiring note above
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

# Printed when a torch CUDA check fails. torch.cuda.is_available() swallows
# the underlying driver error; torch.cuda.init() raises it. Naming the real
# cause turns a field report into a one-post diagnosis (a bare
# AssertionError on a GX10 with a pre-CUDA-13 driver cost a forum
# round-trip, 2026-07-19). Always exits 0: the caller decides failure.
torch_cuda_diag() {
  python - <<'PY' 2>&1
import sys
try:
    import torch
except Exception as e:
    print(f"  diag: torch import failed: {e}")
    sys.exit(0)
cv = torch.version.cuda or ""
if not cv.startswith("13"):
    build = f"CUDA {cv}" if cv else "no CUDA (a CPU-only build)"
    print(f"  diag: torch {torch.__version__} is compiled for {build}, not CUDA 13.")
    print("        A package re-pinned torch; 'update' reinstalls the cu130 wheels.")
    sys.exit(0)
try:
    torch.cuda.init()
except Exception as e:
    msg = str(e).strip().replace("\n", "\n        ")
    print("  diag: the cu130 wheel is installed but CUDA failed to initialize:")
    print(f"        {msg}")
    print("  diag: that is a host problem, not a venv problem. Check nvidia-smi")
    print("        (torch cu130 needs an r580+ driver reporting CUDA 13) and")
    print("        reboot after any driver update.")
    sys.exit(0)
print(f"  diag: CUDA initializes fine ({torch.cuda.get_device_name(0)});")
print("        the failure above lies elsewhere.")
PY
}
