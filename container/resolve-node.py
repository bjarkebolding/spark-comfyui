#!/usr/bin/env python3
"""resolve-node.py — turn a bare Comfy Registry node id into id@version.

Prints the version to stdout and exits 0, or exits non-zero and prints nothing.
Called by container/entrypoint.sh on the cold-cache path ONLY, as a cheap
alternative to `cm-cli update-cache`.

Why this exists. cm-cli resolves a BARE id against ComfyUI-Manager's registry
cache. On a cold box that cache is empty, and manager_core forces a non-blocking
reload when Manager is a pip package (which it is here), so the id falls through
to 'unknown' and cm-cli reads the catalogue bundled in the pip package instead.
That catalogue does not carry every registry node, which is the 2026-08-05 field
bug. The entrypoint's answer was `cm-cli update-cache`, which walks the whole
registry 30 nodes per page with a 0.5s sleep between pages: 169 requests and
84.5s of pure sleep, measured at 140s on this box.

A PINNED id needs none of that. resolve_node_spec returns immediately for
id@version without ever touching the cache, and cnr_install then makes a single
GET /nodes/{id}/install?version=X. So the only missing piece on a cold box is
the version string, and that is one HTTP request.

Deliberately narrow:
  - Bare ids only. A URL or an already-pinned id is not our problem.
  - No `latest_version` in the response means NEVER guess. 1445 of 5057 registry
    nodes have none; upstream defaults those to 'nightly' and we must not
    silently pick something different. Exit non-zero and let the caller fall
    back to the slow path that was always going to run.
  - Every failure is silent and non-zero. This is a fast path, not a gate: the
    caller must be free to ignore it entirely. The on-disk presence check
    remains the only thing that decides whether a node installed.
"""
import json
import re
import sys
import urllib.request

API = "https://api.comfy.org/nodes/%s"
TIMEOUT = 15

if len(sys.argv) != 2:
    sys.exit(2)
node_id = sys.argv[1].strip()

# The id lands in a URL. Registry ids are lowercase alphanumeric with dashes,
# dots and underscores; anything else is not ours to resolve.
if not re.fullmatch(r"[A-Za-z0-9._-]+", node_id):
    sys.exit(2)

try:
    req = urllib.request.Request(API % node_id,
                                 headers={"Accept": "application/json"})
    with urllib.request.urlopen(req, timeout=TIMEOUT) as r:
        data = json.loads(r.read())
except Exception:
    sys.exit(1)

if not isinstance(data, dict):
    sys.exit(1)
version = (data.get("latest_version") or {}).get("version")
if not isinstance(version, str) or not version.strip():
    sys.exit(1)          # no latest_version: fall back, do not invent one

print(version.strip())
