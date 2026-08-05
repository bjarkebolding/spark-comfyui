# mods/

Each subdirectory is a self-contained, idempotent modification. Mods run in
filename order, so the numeric prefix (`05-`, `10-`, `20-`, `30-`) controls
sequence.

Mods run **inside the image**, in one of two places. There is no host-side mod
pass.

| Where | Runner | Mods | Why there |
|---|---|---|---|
| Image build | `container/build-mods.sh` | `05-setuptools-compat`, `10-unified-memory-free` | Need neither a GPU nor user content, so they bake into the image. A failed apply or verify fails the build. |
| Every launch | `container/entrypoint.sh` | `20-torch-repair` (via `mod_prerun`), `30-manager-config` (via `configure.py`) | Need the GPU or the bind-mounted user directory, neither of which exists at build time. |

Adding a build-time mod means dropping in `mods/NN-name/` **and** adding it to
the list in `container/build-mods.sh`. Adding launch-time behavior means
editing `container/entrypoint.sh`.

## Anatomy of a mod

```
mods/
  _lib/mod_common.sh        # shared helpers (not a mod; the leading _ skips it)
  05-setuptools-compat/
    run.sh                  # the contract, for mods a contract runner invokes
  10-unified-memory-free/
    run.sh
    transform.py            # supporting file(s) (optional)
  30-manager-config/
    configure.py            # no run.sh: the entrypoint calls this directly
```

`run.sh` is required only for mods a contract runner invokes, which today
means `container/build-mods.sh` (05, 10) and the entrypoint's `mod_prerun`
call (20). `30-manager-config` is not one: the entrypoint runs
`configure.py apply` directly, because the config it writes lives under the
bind-mounted user directory and has to be re-asserted at run time. It carried
a `run.sh` until 2026-08-05 purely because the deleted native host-side pass
globbed every mod and called `mod_apply` on all of them. Same story as mods
40 and 50 (see CLAUDE.md): native-era orchestration, retired once the
container did the job.

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

It may optionally define a fourth:

| Function | Returns | Purpose |
|---|---|---|
| `mod_prerun` | returns 0/1 | Runs before every launch, from the entrypoint. Absence is a silent no-op. Only for a cheap pre-launch guard, as in `20-torch-repair`. |

## Writing a source patch

Most source-patch mods edit one ComfyUI Python file. Use the helpers:

```sh
MOD_TAG="my_fix"
MOD_FILE="comfy/somefile.py"
mod_describe() { echo "what this does"; }
mod_apply()    { py_patch_file "$MOD_FILE" "$MOD_TAG" "$MOD_DIR/transform.py"; }
mod_verify()   { py_marker_present "$MOD_FILE" "$MOD_TAG"; }
```

`transform.py` reads the source on stdin, writes the patched source to stdout,
and **must echo the input unchanged if it cannot find its anchor**. That is how
"upstream moved the code" is detected: it surfaces as
`skipped:anchor-not-found`, which fails the image build loudly. The marker
string arrives via `$MARKER`.

`py_patch_file` handles the rest: the idempotency check, a `.spark-orig` backup
refreshed on every apply, and a post-write `ast.parse` guard that reverts the
file if the patch would have produced invalid Python.

Test a transform against a realistic fixture and confirm the result still
parses before shipping it. That is a golden rule, not a suggestion.

## Vestigial flags

`MOD_CRITICAL` and `MOD_STREAM` were read by the native runner, which was
deleted with the container cut. `20-torch-repair` still declares them as
documentation of intent. Nothing reads them. Do not build new behavior on
them.
