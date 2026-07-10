# shellcheck shell=bash
# =============================================================================
#  mod: 05-setuptools-compat
#  Keeps setuptools within torch's own declared constraint (e.g. <82 for
#  torch 2.12) — a blanket 'pip install -U setuptools', or a custom node's
#  requirements, can break that pin, which then breaks source builds
#  (SageAttention uses torch's setuptools machinery at build time). Reads the
#  constraint from torch's own metadata so it stays correct across versions.
#  Not a source patch — a venv-package repair. Soft: if it can't determine or
#  fix the pin, generation still works most of the time; only source builds
#  (40-sageattention) are at risk, and that mod is critical on its own.
# =============================================================================
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
