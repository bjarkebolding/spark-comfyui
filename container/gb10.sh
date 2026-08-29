# shellcheck shell=bash
# =============================================================================
#  gb10.sh — the GB10 probes, gates and guards
# =============================================================================
#  Everything here answers one question about THIS machine's GPU stack: does
#  the thing actually work right now. That is golden rule 3 in library form.
#  Two kinds live here and the difference matters at the call site:
#
#    *_ok       a probe. Returns 0/1, changes nothing, cheap enough to gate a
#               launch on. sage_kernel_ok, onnx_gpu_ok, torch_cuda_ok,
#               kitchen_nvfp4_ok.
#    ensure_*   a guard. Probes, REPAIRS if broken, probes again.
#               ensure_onnx_gpu warns on failure, ensure_torch_cuda returns
#               non-zero so the caller can abort. That asymmetry is
#               deliberate: CPU onnxruntime is a slow server, a torch without
#               CUDA is no server at all.
#
#  Sourced INSIDE THE IMAGE only, by container/entrypoint.sh at every launch
#  and by doctor's in-container gate block. The host script does not source it
#  and must not start: every function here calls the venv's python and expects
#  torch, sageattention, onnxruntime and comfy_kitchen to be importable, none
#  of which exist on the host.
#
#  It lived in mods/_lib/mod_common.sh until 2026-08-08. That was three times
#  the size of the mod contract it shared a file with, in a directory that
#  means build-time image modifications, and none of it has anything to do
#  with mods. mod_common.sh is the contract now, and only the contract.
#
#  NOT standalone: a sourced fragment that assumes the sourcing shell already
#  provides log/warn/die, and the globals TORCH_INDEX (ensure_torch_cuda) and
#  ORT_WHEEL_URL (ensure_onnx_gpu). The runtime image sets both as ENV.
#
#  The SageAttention COMPILE is not here: it is a Dockerfile stage
#  (`FROM torch AS sage`) pinned to SAGE_REF. What lives here is the live
#  kernel gate that decides whether that build is usable on this GPU, which
#  is the part a build with no GPU cannot do.
# =============================================================================

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

# Comfy Kitchen INT8 attention, the same live-kernel treatment as sage above:
# comfy_kitchen.int8_attention_is_available() is a capability probe, not proof
# that a kernel runs, so this calls ComfyUI's own wrapper on a realistic
# diffusion shape and checks the result. Only consulted when the caller asked
# for this backend (SPARK_ATTENTION=ck); the default path never pays for it.
ck_attention_ok() {
  python - <<'PY' >/dev/null 2>&1
import sys
sys.path.insert(0, "/opt/ComfyUI")
import torch, comfy_kitchen
assert comfy_kitchen.int8_attention_is_available()
import comfy.ldm.modules.attention as A
b, h, s, d = 1, 24, 4096, 64
q = torch.randn(b, s, h * d, dtype=torch.bfloat16, device="cuda")
o = A.attention_comfy_kitchen_int8(q, q, q, h)
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

# ONNX Runtime GPU check: a REAL session on a REAL model, never a provider
# string. get_available_providers() lists what was COMPILED IN, so it names
# CUDAExecutionProvider even in a process where the provider cannot load
# (measured 2026-08-29: unresolvable libcublasLt.so.13, session silently on
# CPU at 5x the time, old string check passed anyway). Naming CUDA as the
# only provider does not make ORT raise either, it falls back and reports it
# in get_providers(), which is why the assert below reads the SESSION.
#
# No torch import is needed: the runtime image puts torch's bundled CUDA
# libraries on LD_LIBRARY_PATH, so ORT resolves them on its own. That is
# what makes this gate cheap enough to run on every start, and it is also
# why a node that reaches onnxruntime without importing torch no longer
# lands silently on CPU.
#
# The model is embedded rather than built with the onnx package, which is not
# in the image and must not be added for a gate. 193 bytes, one 3x3 conv:
# the cuDNN path the DWPose and ControlNet preprocessors actually use.
onnx_gpu_ok() {
  python - <<'PY' >/dev/null 2>&1
import base64, numpy as np
import onnxruntime as ort
MODEL = base64.b64decode(
    "CAo6tgEKOQoBWAoBVxIBWSIEQ29udioVCgxrZXJuZWxfc2hhcGVAA0ADoAEHKhEKBHBhZH"
    "NAAUABQAFAAaABBxIKc3BhcmtfZ2F0ZSozCAEIAQgDCAMQAUIBV0okAAAAPwAAAD8AAAA/"
    "AAAAPwAAAD8AAAA/AAAAPwAAAD8AAAA/WhsKAVgSFgoUCAESEAoCCAEKAggBCgIICAoCCA"
    "hiGwoBWRIWChQIARIQCgIIAQoCCAEKAggICgIICEIECgAQEg=="
)
s = ort.InferenceSession(MODEL, providers=["CUDAExecutionProvider"])
assert s.get_providers()[0] == "CUDAExecutionProvider", s.get_providers()
y = s.run(None, {"X": np.ones((1, 1, 8, 8), dtype=np.float32)})[0]
assert np.isfinite(y).all()
assert abs(float(y[0, 0, 4, 4]) - 4.5) < 1e-4, float(y[0, 0, 4, 4])
PY
}

# DWPose / ControlNet preprocessors run on onnxruntime. Without the GPU wheel
# they silently fall back to CPU — a large hidden slowdown. Also guards the
# shadow trap:
# a later 'pip install onnxruntime' (e.g. pulled in by a custom node)
# overwrites the GPU wheel via the shared import path with no pip conflict.
#
# Called by container/entrypoint.sh on every start, right after the torch
# guard and for the same reason: node installs are when shadowing happens.
# The Dockerfile installs the pinned wheel at build time, doctor detects
# drift, and this repairs it, which makes onnx symmetric with torch (mod 20).
# The healthy path builds one tiny CUDA session; only a genuinely shadowed
# install pays the reinstall. The caller treats a failure as a warn,
# never a die: CPU onnxruntime is a slowdown, not a broken server.
ensure_onnx_gpu() {
  local pyver
  pyver="$(python -c 'import sys; print(f"{sys.version_info.major}{sys.version_info.minor}")')"
  if [[ "$pyver" != "312" ]]; then
    warn "GPU onnxruntime wheel is cp312; this venv is Python ${pyver:0:1}.${pyver:1}.
Skipping — preprocessor nodes (DWPose etc.) will use CPU onnxruntime."
    return 0
  fi
  if onnx_gpu_ok; then
    echo "onnxruntime: OK — GPU provider live"
    return 0
  fi
  log "Installing GPU onnxruntime"
  # Remove PyPI CPU dists that shadow the same import path first.
  pip uninstall -y onnxruntime onnxruntime-gpu >/dev/null 2>&1 || true
  pip install "$ORT_WHEEL_URL"
  if onnx_gpu_ok; then
    echo "onnxruntime CUDAExecutionProvider: live (preprocessors on GPU)"
  else
    warn "onnxruntime installed but CUDA provider is NOT available — DWPose
etc. will fall back to CPU. Ensure cuDNN 9.x is installed system-wide
(DGX OS ships it; otherwise: sudo apt-get install -y libcudnn9-cuda-13)."
  fi
}

# Exits non-zero unless torch is a CUDA 13 build that can see the GPU, and on
# success prints one line describing what it found. That line is not decoration:
# the entrypoint prints a "==> Torch CUDA guard" header for this step, and a
# header followed by nothing reads as "it did nothing" or "it hung" (asked about
# from a real new-box run, 2026-08-09). Every other step in the pipeline says
# something when it succeeds; this one was the only silent exception, inherited
# from the mod era when a mod runner printed the status on the mod's behalf.
# It is also the exact line worth having in a bug report.
torch_cuda_ok() {
  python - <<'PY' 2>/dev/null
import torch
assert (torch.version.cuda or "").startswith("13") and torch.cuda.is_available()
print(f"torch {torch.__version__} | CUDA {torch.version.cuda} | {torch.cuda.get_device_name(0)}")
PY
}

# The launch-path twin of ensure_onnx_gpu above, and it exists for the same
# reason: a custom node's requirements can pull a CPU-only torch over the cu130
# wheels through the shared import path, with no pip conflict to notice. The
# entrypoint calls both right after the installs that cause the problem.
#
# The one difference is what a failure means. CPU onnxruntime is a slowdown, so
# that guard warns; a torch that cannot see the GPU means nothing works at all,
# so this one returns non-zero and the caller aborts the launch. torch_cuda_diag
# runs first, because torch.cuda.is_available() swallows the underlying driver
# error and the caller's message would otherwise name no cause.
#
# This was mods/20-torch-repair until 2026-08-08. It never used the mod
# contract for anything: no MOD_DIR, no supporting files, and mod_apply had no
# caller in any runner. It is a function here, next to the guard its own
# comments already cited as the model.
ensure_torch_cuda() {
  local info
  if info="$(torch_cuda_ok)"; then
    echo "torch: OK — $info"
    return 0
  fi
  warn "torch lost CUDA 13 support — reinstalling cu130 wheels"
  pip install --force-reinstall torch torchvision torchaudio --index-url "$TORCH_INDEX"
  if info="$(torch_cuda_ok)"; then
    echo "torch: OK — $info (repaired)"
    return 0
  fi
  torch_cuda_diag
  return 1
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
