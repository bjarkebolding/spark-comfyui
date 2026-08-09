"""Author a recipe.json from a workflow that already runs on this box.

Called by lib/recipes.sh as: recipe_capture.py WORKFLOW OUTFILE
with MODELS_DIR, RECIPE_NAME, CATALOG_FILE and RECIPES_DIR in the environment.

Scans the workflow for model filenames, locates each under the models tree to
learn its destination, and records size and sha256 from the local file. URLs
come from two places: recipes already in RECIPES_DIR, matched on sha256 so a
url is only reused for provably identical bytes, and failing that
ComfyUI-Manager's model-list.json matched on filename. Anything still blank is
reported loudly, because a recipe with a blank url is refused by
recipe_read.py.

CATALOG_FILE is a path, not the JSON itself: the catalogue is ~3 MB and
passing it through the environment overflows the exec argument limit.
"""
import hashlib, json, os, re, sys

workflow, outfile = sys.argv[1], sys.argv[2]
models_dir = os.environ["MODELS_DIR"]
catalog_file = os.environ.get("CATALOG_FILE", "")

MODEL_RE = re.compile(r"\.(safetensors|ckpt|pth|sft|gguf|onnx)$", re.I)

with open(workflow) as fh:
    wf = json.load(fh)

# Model references hide in widgets_values, which is an untyped list whose
# shape varies per node, so scan every string rather than guess a schema.
names = []
def walk(v):
    if isinstance(v, str):
        if MODEL_RE.search(v) and v not in names:
            names.append(v)
    elif isinstance(v, list):
        for x in v:
            walk(x)
    elif isinstance(v, dict):
        for x in v.values():
            walk(x)
for node in wf.get("nodes", []):
    walk(node.get("widgets_values"))

catalog = {}
if catalog_file and os.path.isfile(catalog_file):
    try:
        with open(catalog_file) as fh:
            for m in json.load(fh).get("models", []):
                catalog.setdefault(m.get("filename", ""), m)
    except (OSError, ValueError):
        pass

# Recipes already in the repo are themselves a catalogue, and a better one:
# their urls were verified by whoever wrote them. Keyed by sha256, not
# filename, so it only ever reuses a url for provably identical bytes.
known_by_sha = {}
recipes_dir = os.environ.get("RECIPES_DIR", "")
if recipes_dir and os.path.isdir(recipes_dir):
    for entry in sorted(os.listdir(recipes_dir)):
        if not entry.endswith(".json"):
            continue
        rj = os.path.join(recipes_dir, entry)
        if not os.path.isfile(rj):
            continue
        try:
            with open(rj) as fh:
                for m in json.load(fh).get("models", []):
                    if m.get("url") and m.get("sha256"):
                        known_by_sha.setdefault(m["sha256"].lower(), m["url"])
        except (OSError, ValueError):
            continue

# Index the models tree once: a workflow names files, never paths.
on_disk = {}
for root, _dirs, files in os.walk(models_dir):
    for fn in files:
        on_disk.setdefault(fn, os.path.join(root, fn))

models, unresolved, blank = [], [], 0
for fn in names:
    base = os.path.basename(fn)
    path = on_disk.get(base)
    if not path:
        unresolved.append(fn)
        continue
    rel = os.path.relpath(os.path.dirname(path), models_dir)
    size = os.path.getsize(path)
    print("  hashing %s (%.1f GB)" % (base, size / 1e9), flush=True)
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(16 << 20), b""):
            h.update(chunk)
    digest = h.hexdigest()
    # An existing recipe wins over the Manager catalogue: it is matched on
    # sha256, so it is the same bytes, whereas the catalogue matches on a
    # filename that could mean anything.
    url = known_by_sha.get(digest) or catalog.get(base, {}).get("url", "")
    if not url:
        blank += 1
    models.append({"file": base, "dest": rel.replace(os.sep, "/"),
                   "url": url, "size": size, "sha256": digest})

# The graph is embedded so a recipe is ONE shareable file. It goes LAST and
# stays COMPACT: pretty-printed it is 47 KB and 1735 lines, which would bury
# the ~40 lines a human actually reads under the graph. ComfyUI itself stores
# it minified, so this matches.
PLACEHOLDER = "@@SPARK_WORKFLOW@@"
recipe = {
    "schema": 1,
    "name": os.environ["RECIPE_NAME"],
    "description": "",
    "notes": "",
    "nodes": [],
    "run_flags": [],
    "run_note": "",
    "models": models,
    "workflow": PLACEHOLDER,
}
text = json.dumps(recipe, indent=2)
text = text.replace('"%s"' % PLACEHOLDER, json.dumps(wf, separators=(",", ":")))
with open(outfile, "w") as fh:
    fh.write(text + "\n")

print("  %d model(s) resolved on disk" % len(models))
for u in unresolved:
    print("  \033[1;33m[warn]\033[0m not found under the models dir: %s" % u)
if blank:
    print("  \033[1;33m[warn]\033[0m %d model(s) have no url — the recipe CANNOT "
          "install until you fill them in" % blank)
    print("          for a HuggingFace file the sha256 above should match "
          "lfs.sha256 from")
    print("          https://huggingface.co/api/models/<repo>?blobs=true")
