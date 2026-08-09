# recipes/

A recipe is everything a workflow needs in order to run, as data: the workflow
graph, the custom nodes it wants, and every model with a destination, size and
sha256.

It exists because a ComfyUI workflow names models by bare filename. No path,
no URL, no hash. The destination folder is only implied by the loader node
type, and that inference fails outright on subgraphs, where several models can
sit inside one node with nothing saying which belongs in `diffusion_models/`
and which in `vae/`. A recipe states it.

Recipes are TRACKED in this repo, unlike `comfyui-nodes.list` and
`comfyui-patches.list`. A recipe is meant to be shared. It is the executable
form of a forum post, and one file can be pasted into a gist or an issue.

## Using one

```
./spark-comfyui.sh recipe list                    what is available
./spark-comfyui.sh recipe show NAME               models, nodes and flags
./spark-comfyui.sh recipe check NAME              what is missing, downloads nothing
./spark-comfyui.sh recipe check NAME --verify     also hash every local file
./spark-comfyui.sh recipe install NAME
```

`check` and `install` are the same engine, so what `check` tells you is
exactly what `install` would do. `install` adds the actions.

`install` downloads what is missing into the resolved `models` mount, merges
the recipe's `nodes` into `comfyui-nodes.list`, and writes the workflow to
`user/default/workflows/NAME.json`. An existing workflow of the same name that
differs is saved aside as `NAME.json.bak` first.

Useful flags: `--yes` skips the download confirmation, needed when stdin is
not a terminal. `--force` re-downloads a model whose size or hash disagrees
with the recipe.

Downloads resume. A file is fetched to `<name>.part`, verified against its
sha256, and only then moved into place, so a partial or corrupt file is never
visible to a loader. Re-run `install` to continue an interrupted fetch.

A preflight runs before anything downloads. One HEAD request per missing
model classifies it as gated, rotted, unreachable, or a size that disagrees
with the recipe. `check` reports; `install` refuses to start and downloads
nothing.

A model already on disk that does not match is reported and LEFT ALONE.
Models are expensive and a recipe disagreeing is not grounds to delete one.
Use `--force` to opt in.

## Gated downloads

`HF_TOKEN` and `CIVITAI_TOKEN` are read from the environment and sent as a
bearer header:

```
HF_TOKEN=hf_... ./spark-comfyui.sh recipe install NAME
```

Never put a token in a recipe. Recipes are committed, so a credential must not
be able to live in one.

Most HuggingFace files are public, so an unset token is not warned about. The
preflight names the variable only when a server actually answers 401 or 403.

## Authoring one

Start from a workflow that already runs on this box:

```
./spark-comfyui.sh recipe capture WORKFLOW [--name NAME]
```

`WORKFLOW` is a saved workflow's name as it appears in the ComfyUI sidebar, or
a path to a `.json` file. The name defaults to the workflow's.

`capture` scans the graph for model filenames, locates each one under the
models dir, and records `dest`, `size` and `sha256` from the file itself. It
fills `url` from two sources: recipes already in this directory, matched on
sha256 so the bytes are known to be the same, then ComfyUI-Manager's model
catalogue, matched on filename. Anything it cannot resolve is left blank and
reported, and the recipe cannot install until you fill those in.

`capture` never overwrites an existing recipe.

After capturing, fill in by hand: any blank `url`, plus `description`,
`notes`, `nodes`, `run_flags` and `run_note`, which are always left empty.

### Finding a url and its sha256

Both sites that matter publish the hash without a download, which is what
makes a required sha256 reasonable.

HuggingFace, `lfs.sha256` from:

```
https://huggingface.co/api/models/<repo>?blobs=true
```

CivitAI, `files[].hashes.SHA256`, which is uppercase and needs lowering.

This also gives a way to PROVE a source rather than guess it. The shipped
`image_krea2_turbo_t2i` recipe's URLs were found by matching local hashes
against `Comfy-Org/Krea-2`.

## File format

One self-contained `.json` file per recipe, named `<name>.json`. The name
lands in a path, so it must be a plain name with no slashes.

```json
{
  "schema": 1,
  "name": "Krea-2 Turbo text-to-image",
  "description": "one or two lines on what this produces",
  "notes": "sizes, sources, anything a reader needs before committing the disk",
  "nodes": [],
  "run_flags": [],
  "run_note": "",
  "models": [
    {
      "file": "krea2_darkbrush.safetensors",
      "dest": "loras",
      "url": "https://huggingface.co/Comfy-Org/Krea-2/resolve/main/loras/krea2_darkbrush.safetensors",
      "size": 469291992,
      "sha256": "f47c4316dd93af66e0518c93b582f459571d4925b519133770c73a52cd5db7c6"
    }
  ],
  "workflow": { "nodes": [] }
}
```

| Field | Required | Meaning |
|---|---|---|
| `schema` | yes | must be `1` |
| `name` | yes | human title, not the filename |
| `description` | no | what the workflow produces |
| `notes` | no | JSON has no comments, so prose goes here |
| `nodes` | no | node-list entries, merged into `comfyui-nodes.list` on install |
| `run_flags` | no | launch flags this workflow needs |
| `run_note` | no | why those flags are needed |
| `models` | no | one object per model, all five keys required |
| `workflow` | no | the embedded graph object, never a path |

Model entry rules, all enforced before any I/O:

- `file` is a bare filename. No slashes, no `.` or `..`.
- `dest` is relative to the models dir. No leading `/` or `~`, no `..`.
- `url` is http or https.
- `size` is a positive integer, in bytes.
- `sha256` is exactly 64 hex characters.
- No two entries may resolve to the same `dest/file`.

`run_flags` are printed as advice and never applied. Those flags are global,
and one workflow's requirement is not a reason to change how every other
workflow launches.

`nodes` are not installed directly. They are merged into
`comfyui-nodes.list` and the entrypoint does the work on the next start, so
there is one source of truth for the installed node set.

### Why the graph is last and compact

The metadata is written with `indent=2`, the graph is spliced in minified on
the final line. Measured on the shipped `image_krea2_turbo_t2i`: 35394 bytes
total over 40 lines, of which the graph is a single 33550 byte line and the
metadata is 1844 bytes. Pretty-printing the whole thing gives 73005 bytes over
2394 lines, burying the 39 lines a human actually reads. ComfyUI stores
workflows minified anyway, so this matches.

Read it back out with the workflow on its own:

```
python3 lib/recipe_read.py recipes/NAME.json --extract-workflow /tmp/wf.json
```

That runs the full validation first, so a successful extract also means the
recipe is sound.

## Validation

`lib/recipe_read.py` is the security boundary. A recipe writes files into the
models tree and fetches URLs, so every field is checked before the first byte
is downloaded. A malformed recipe fails immediately, not halfway through
40 GB. Errors print as `recipe error: ...` and exit non-zero.

A recipe that fails validation still shows up in `recipe list`, marked
`INVALID`, rather than silently disappearing.
