# mods/

A mod is a **build-time modification to the image**. That is the whole
definition, and it is the only kind there is.

Each subdirectory is one mod: self-contained, idempotent, and applied by
`container/build-mods.sh` during the image build. A mod that fails to apply or
fails its own verification fails the build.

```
mods/
  _lib/mod_common.sh        # the contract's helpers (not a mod; the _ skips it)
  setuptools-compat/
    run.sh                  # the contract
  unified-memory-free/
    run.sh
    transform.py            # supporting file(s) (optional)
```

`mod_common.sh` holds the contract and nothing else: `mod_marker`,
`py_patch_file`, `py_marker_present`. The GB10 probes and guards that used to
share the file (`sage_kernel_ok`, `ensure_torch_cuda`, `ensure_onnx_gpu`,
`kitchen_nvfp4_ok`, `torch_cuda_diag`) are `container/gb10.sh`, sourced by the
entrypoint and by doctor's in-container gates. They were three quarters of the
file and had nothing to do with mods.

Adding one means dropping in `mods/<name>/` **and** naming it in
`container/build-mods.sh`. The list there is explicit on purpose, not a glob
over `mods/*/`: a half-copied checkout has to fail loudly rather than quietly
bake an image with fewer patches in it. Silently skipping
`unified-memory-free` would ship the `get_free_memory` offload cliff back into
the image and say nothing, which is exactly what golden rule 5 forbids. Same
reasoning as `lib/` being sourced by name in the entry point.

## There are no launch-time mods

There used to be two, and neither was really a mod.

`20-torch-repair` implemented half the contract (`mod_prerun` only), had no
`MOD_DIR` and no supporting files, and its `mod_apply` had no caller in any
runner. Meanwhile `ensure_onnx_gpu`, a structurally identical launch-time
guard that the entrypoint runs three lines later, was a plain shared
function. Two of the same thing, built two different ways, for
reasons that were purely historical. It is `ensure_torch_cuda` in
`container/gb10.sh` now, next to its twin.

`30-manager-config` never implemented the contract at all: no `run.sh`, and
the entrypoint always ran its Python directly. It is
`container/manager-config.py` now, beside the entrypoint it serves, the same
way `build-patches.sh` sits beside the build.

Both moved on 2026-08-08. If you need something to happen at launch, add a
step to `container/entrypoint.sh`, where it will sit beside the six that are
already there: Manager config, the node list, custom-node requirements, the
torch guard, the onnx guard and the SageAttention gate, with the launch itself
as step seven. Do not reintroduce a mod hook for it.

## The numeric prefixes are gone

Directories were `05-`, `10-`, `20-`, `30-` until 2026-08-08. The numbers were
real once, when a deleted native-era pass globbed `mods/` and applied
everything in filename order. After the container cut there was no such pass,
so the numbers encoded a sequence the system no longer had:

- The build runner iterates a **set**. `setuptools-compat` and
  `unified-memory-free` are independent and their order does not matter.
- The launch-time pair ran inside the entrypoint's pipeline, as steps 1 and 4
  of seven, so their relative order was set by that pipeline. It ran `30`
  before `20`, against the numbers.

They were not renumbered, because no correct numbering exists for a thing with
no order. They were dropped.

## The contract

Every `run.sh` is sourced, not executed, with `_lib/mod_common.sh` already
loaded and `INSTALL_DIR`, `VENV_DIR` and `MOD_DIR` set. It defines three shell
functions:

| Function | Returns | Purpose |
|---|---|---|
| `mod_describe` | echoes one line | Human description, printed by the runner |
| `mod_apply` | echoes a status word, returns 0 | Applies the mod idempotently |
| `mod_verify` | exit 0 = active, 1 = not | Is the mod currently in effect |

`mod_apply` echoes one of `applied`, `present` or `skipped:<reason>` as its
first token. The helpers in `mod_common.sh` do that for you.

## Writing a source patch

Most mods edit one ComfyUI Python file. Use the helpers:

```sh
MOD_TAG="my_fix"
MOD_FILE="comfy/somefile.py"
mod_describe() { echo "what this does"; }
mod_apply()    { py_patch_file "$MOD_FILE" "$MOD_TAG" "$MOD_DIR/transform.py"; }
mod_verify()   { py_marker_present "$MOD_FILE" "$MOD_TAG"; }
```

`MOD_TAG` is the marker written into the patched file, and it is independent
of the directory name, which is why renaming these directories was safe.

`transform.py` reads the source on stdin, writes the patched source to stdout,
and **must echo the input unchanged if it cannot find its anchor**. That is how
"upstream moved the code" is detected: it surfaces as
`skipped:anchor-not-found`, which fails the image build loudly.  The marker
string arrives via `$MARKER`.

`py_patch_file` handles the rest: the idempotency check, a `.spark-orig` backup
refreshed on every apply, and a post-write `ast.parse` guard that reverts the
file if the patch would have produced invalid Python.

Test a transform against a realistic fixture and confirm the result still
parses before shipping it. That is a golden rule, not a suggestion.

## A note on timing

`container/build-mods.sh` runs late in the Dockerfile, after ComfyUI's
requirements and the onnx wheel are installed. A mod therefore **repairs**
state that earlier steps have already set; it cannot pre-empt them. If
something must happen before the requirements install, it belongs in the
Dockerfile above that step, not in a mod.
