#!/usr/bin/env python3
"""Inject a unified-memory-aware early return into ComfyUI's get_free_memory.

On GB10 unified memory, torch.cuda.mem_get_info under-reports free memory when
another CUDA process is resident, causing needless offload and 5-15x slower
sampling. Host-available memory is the correct free-pool figure. Prefer
comfy.system_memory.virtual_memory_available(), which upstream made
cgroup-aware on 2026-08-27, and fall back to raw psutil if that module ever
moves; we run in a container, so the cgroup view is the honest one.

Reads source on stdin, writes patched source to stdout. No-op (echoes input)
if the anchor can't be found, so the caller can detect "anchor not found".
The marker string is passed via the MARKER env var.
"""
import os
import re
import sys

MARKER = os.environ["MARKER"]  # e.g. "# spark-comfyui:mem_unified"
src = sys.stdin.read()

if "def get_free_memory" not in src:
    sys.stdout.write(src); sys.exit(0)

# Ensure psutil is importable at module scope (belt-and-suspenders; the
# injected code also imports it locally).
if "import psutil" not in src:
    src = re.sub(r"(\nimport torch\b)", r"\nimport psutil\1", src, count=1)

pat = re.compile(r"(def get_free_memory\(dev=None, torch_free_too=False\):\n)")
if not pat.search(src):
    pat = re.compile(r"(def get_free_memory\([^\)]*\):\n)")
    if not pat.search(src):
        sys.stdout.write(src); sys.exit(0)

inject = (
    f"    {MARKER} — on GB10 unified memory, CUDA free queries under-report\n"
    "    # when another CUDA process is resident; use host-available memory\n"
    "    # as the truth for the shared pool.\n"
    "    try:\n"
    "        try:\n"
    "            # cgroup-aware since upstream 2026-08-27; we run in a\n"
    "            # container, so a raw psutil read would report the HOST\n"
    "            # pool and overshoot under a docker --memory limit.\n"
    "            _avail = comfy.system_memory.virtual_memory_available()\n"
    "        except Exception:\n"
    "            import psutil as _ps\n"
    "            _avail = _ps.virtual_memory().available\n"
    "        _dev = dev if dev is not None else 'cuda'\n"
    "        if 'cpu' not in str(_dev):\n"
    "            if torch_free_too:\n"
    "                return (_avail, _avail)\n"
    "            return _avail\n"
    "    except Exception:\n"
    "        pass\n"
)
sys.stdout.write(pat.sub(lambda m: m.group(1) + inject, src, count=1))
