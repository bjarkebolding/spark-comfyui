# shellcheck shell=bash
# =============================================================================
#  mod: setuptools-compat
#  Keeps setuptools within torch's own declared constraint (e.g. <82 for
#  torch 2.12) — a blanket 'pip install -U setuptools', or a custom node's
#  requirements, can break that pin, which then breaks source builds
#  (SageAttention uses torch's setuptools machinery at build time). Reads the
#  constraint from torch's own metadata so it stays correct across versions.
#
#  Not a source patch — a venv-package repair, which is why it has no
#  transform.py. It runs in the image build, from container/build-mods.sh,
#  which the Dockerfile invokes AFTER ComfyUI's requirements and the onnx
#  wheel. So this repairs a pin that has already been moved rather than
#  preventing the move; the header claimed the opposite until 2026-08-08, a
#  leftover from the native install order. The outcome is the same either way
#  (mod_verify decides, not the timing), but do not rely on the old reading:
#  anything that needs to run BEFORE the requirements has to move up in the
#  Dockerfile, not into this pass.
# =============================================================================
# Lived in mods/_lib/mod_common.sh until 2026-08-08 with exactly one caller,
# this mod, so it was never shared anything.
#
# torch pins a setuptools upper bound (e.g. <82 for torch 2.12); a blanket
# 'pip install -U setuptools' — or a custom node's requirements — can break
# it, which then breaks source builds (SageAttention uses torch's setuptools
# machinery). Read torch's OWN declared constraint from its metadata so this
# stays correct across torch versions, and upgrade/downgrade within it.
# No-op when already conformant; harmless "latest" upgrade if torch absent.
#
# (The two lines that used to head this comment described repair_torch, which
# sat below it in the old shared file. They travelled with the wrong function
# for a while; they are gone rather than moved, because repair_torch is now
# folded into ensure_torch_cuda in container/gb10.sh.)
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

mod_describe() {
  echo "setuptools pinned within torch's declared constraint"
}

mod_apply() {
  if mod_verify; then
    echo "present"
    return 0
  fi
  ensure_setuptools_compat
  if mod_verify; then
    echo "applied"
  else
    echo "skipped:still-out-of-spec"
  fi
}

mod_verify() {
  python - <<'PY' >/dev/null 2>&1
import importlib.metadata as md
from packaging.requirements import Requirement
from packaging.version import Version
st = Version(md.version("setuptools"))
for r in (md.requires("torch") or []):
    req = Requirement(r.split(";")[0].strip())
    if req.name == "setuptools":
        assert st in req.specifier
        break
PY
}
