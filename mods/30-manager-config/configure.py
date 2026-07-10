#!/usr/bin/env python3
"""Configure ComfyUI-Manager for the DGX Spark: relax the security gating
that a 0.0.0.0 listener otherwise trips, enable uv, and install a native
torch-downgrade guard (downgrade_blacklist).

Deliberately does NOT write pip_auto_fix.list: Manager's prestartup_script.py
calls fix_broken() unconditionally on every launch, which parses that file's
version pins with its own home-grown StrictVersion — a naive `int()`-per-
`.`-segment parser that cannot handle a PEP 440 local version segment. Any
CUDA-specific torch build reports one (e.g. "2.13.0+cu130"), so a pin here
crashes on every single startup (caught internally, logged as [ERROR], not
fatal, but permanent noise for something that can never be fixed by
reformatting our side — Manager also parses the *installed* version the same
broken way when checking for drift). downgrade_blacklist below is a
completely separate Manager mechanism unaffected by this parser, and the
20-torch-repair mod already verifies/repairs torch — via real
torch.cuda.is_available() checks, not string parsing — both at install/
update time and before every `run`. Belt-and-suspenders via pip_auto_fix.list
isn't needed and actively hurts here.

Modes (argv[1]):
  apply   -> write config; print a short status; exit 0
  verify  -> exit 0 if network_mode == personal_cloud, else 1

Env: INSTALL_DIR (required).
"""
import configparser
import os
import sys

mode = sys.argv[1] if len(sys.argv) > 1 else "apply"
install_dir = os.environ["INSTALL_DIR"]
mgr_dir = os.path.join(install_dir, "user", "__manager")
path = os.path.join(mgr_dir, "config.ini")

# Always asserted (launcher depends on it); the rest are set-if-absent so
# hand edits are respected on re-runs.
ALWAYS = {"network_mode": "personal_cloud"}
DEFAULTS = {
    "security_level": "normal",
    "use_uv": "True",
    "file_logging": "True",
    "downgrade_blacklist": "torch,torchvision,torchaudio",
}


def load():
    cfg = configparser.ConfigParser()
    if os.path.isfile(path):
        cfg.read(path)
    if not cfg.has_section("default"):
        cfg.add_section("default")
    return cfg


if mode == "verify":
    cfg = load()
    ok = cfg.get("default", "network_mode", fallback="") == "personal_cloud"
    sys.exit(0 if ok else 1)

# apply
cfg = load()
before = dict(cfg["default"])
for k, v in ALWAYS.items():
    cfg.set("default", k, v)
for k, v in DEFAULTS.items():
    if not cfg.has_option("default", k):
        cfg.set("default", k, v)

changed = [k for k in list(ALWAYS) + list(DEFAULTS)
           if before.get(k) != cfg.get("default", k)]
os.makedirs(mgr_dir, exist_ok=True)
with open(path, "w") as f:
    cfg.write(f)

# Remove a stale pip_auto_fix.list from before this fix (2026-07) — its
# torch-trio pins crash Manager's own version parser on every launch. See
# the module docstring. downgrade_blacklist (set above) and the
# 20-torch-repair mod are the real protection; nothing replaces this file.
fix = os.path.join(mgr_dir, "pip_auto_fix.list")
removed_stale = os.path.isfile(fix)
if removed_stale:
    os.remove(fix)

# Status protocol (see mods/README.md): first word is applied|present|
# skipped, matching every other mod — only "applied" if this run actually
# changed something (config write or stale-file cleanup).
did_something = bool(changed) or removed_stale
msgs = ["applied" if did_something else "present",
        "config " + (f"updated ({', '.join(sorted(changed))})" if changed else "OK")]
if removed_stale:
    msgs.append("removed stale pip_auto_fix.list (crashed Manager's version parser)")

print(msgs[0] + " " + "; ".join(msgs[1:]))
