"""Validate a recipe.json and flatten it to tagged TSV for the shell.

Called by lib/recipes.sh as:
    recipe_read.py PATH                          validate, emit TSV
    recipe_read.py PATH --extract-workflow OUT   validate, then write the graph

A recipe is ONE self-contained .json file with the ComfyUI workflow embedded
under "workflow", so it can be pasted into a gist or an issue and installed
from there. The graph is far too big for the TSV, so the shell asks for it
separately once it is ready to write it out.

Emits one tagged, tab-separated row per line:
    META\t<key>\t<value>
    NODE\t<node-id>
    FLAG\t<run-flag>
    MODEL\t<file>\t<dest>\t<url>\t<size>\t<sha256>

This is the security boundary of the recipe feature. A recipe writes files
into the models tree and fetches URLs, so every field is checked here before
any caller acts: `dest` must be relative with no '..', `file` must be a bare
filename, sha256 exactly 64 hex, size a positive int, url http(s). A
malformed recipe has to fail before the first byte, not halfway through 40 GB.
Exits non-zero with "recipe error: ..." on stderr.
"""
import json, re, sys

path = sys.argv[1]


def die(msg):
    print("recipe error: %s" % msg, file=sys.stderr)
    sys.exit(1)


try:
    with open(path) as fh:
        r = json.load(fh)
except (OSError, ValueError) as e:
    die("%s: %s" % (path, e))

if not isinstance(r, dict):
    die("%s: top level must be a JSON object" % path)
if r.get("schema") != 1:
    die("%s: unsupported schema %r (this tool understands 1)" % (path, r.get("schema")))


def text(key, required=False):
    v = r.get(key, "")
    if not isinstance(v, str):
        die("%s must be a string" % key)
    if required and not v.strip():
        die("%s is required" % key)
    return v.replace("\t", " ").replace("\n", " ")


out = [("META", "name", text("name", True)),
       ("META", "description", text("description")),
       ("META", "notes", text("notes")),
       ("META", "run_note", text("run_note"))]

# The workflow graph is EMBEDDED, so a recipe is one self-contained file that
# can be pasted into a gist or an issue. It is a JSON object, never a path:
# there is nothing beside the recipe to point at.
wf = r.get("workflow")
if wf is not None:
    if not isinstance(wf, dict):
        die("workflow must be the embedded workflow object, not %s"
            % type(wf).__name__)
    if not wf.get("nodes"):
        die("workflow has no 'nodes' — that is not a ComfyUI workflow")
    out.append(("META", "workflow", "%d nodes" % len(wf["nodes"])))

for key, tag in (("nodes", "NODE"), ("run_flags", "FLAG")):
    v = r.get(key, [])
    if not isinstance(v, list):
        die("%s must be a list" % key)
    for item in v:
        if not isinstance(item, str) or not item.strip():
            die("%s entries must be non-empty strings" % key)
        if "\t" in item or "\n" in item:
            die("%s entry contains a tab or newline: %r" % (key, item))
        out.append((tag, item.strip()))

models = r.get("models", [])
if not isinstance(models, list):
    die("models must be a list")
HEX = re.compile(r"^[0-9a-f]{64}$")
seen = set()
for m in models:
    if not isinstance(m, dict):
        die("each entry in models must be an object")
    for k in ("file", "dest", "url", "size", "sha256"):
        if k not in m:
            die("model %r is missing '%s'" % (m.get("file", "?"), k))
    f = m["file"]
    if not isinstance(f, str) or not f or "/" in f or f in (".", ".."):
        die("model file must be a bare filename: %r" % (f,))
    d = m["dest"]
    if not isinstance(d, str) or not d.strip():
        die("model %s: dest must be a non-empty string" % f)
    if d.startswith("/") or d.startswith("~"):
        die("model %s: dest must be relative to the models dir: %r" % (f, d))
    if ".." in d.split("/"):
        die("model %s: dest may not contain '..': %r" % (f, d))
    u = m["url"]
    if not isinstance(u, str) or not u.startswith(("https://", "http://")):
        die("model %s: url must be http(s) — a captured recipe with a blank "
            "url has to be filled in before it can install" % f)
    sz = m["size"]
    if isinstance(sz, bool) or not isinstance(sz, int) or sz <= 0:
        die("model %s: size must be a positive integer (bytes)" % f)
    h = m["sha256"]
    if not isinstance(h, str) or not HEX.match(h.strip().lower()):
        die("model %s: sha256 must be 64 hex characters" % f)
    key = "%s/%s" % (d.strip("/"), f)
    if key in seen:
        die("model listed twice: %s" % key)
    seen.add(key)
    out.append(("MODEL", f, d.strip("/"), u, str(sz), h.strip().lower()))

# Extraction mode. Reached only after everything above has validated, so
# writing the graph out implies the whole recipe is sound. The graph is far
# too big for the TSV the shell reads, hence a separate ask.
if len(sys.argv) > 3 and sys.argv[2] == "--extract-workflow":
    if wf is None:
        die("recipe carries no workflow to extract")
    with open(sys.argv[3], "w") as fh:
        json.dump(wf, fh, separators=(",", ":"))
    sys.exit(0)

for row in out:
    print("\t".join(row))
