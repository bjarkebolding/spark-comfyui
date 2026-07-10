# mods/

Each subdirectory is a self-contained modification that `spark-comfyui.sh`
discovers, applies, and verifies. Mods are run in **filename order**, so the
numeric prefix (`05-`, `10-`, `20-`, …) controls sequence — this matters for
real dependencies, e.g. setuptools pinned correctly (`05`) before torch is
verified (`20`) before SageAttention builds from source (`40`).

Two flavors live here side by side under one contract: **source-patch mods**
that edit ComfyUI's own Python (`10-unified-memory-free`) or its config tree
(`30-manager-config`), and **venv-package mods** that install/verify/repair
things in the virtualenv (`05-setuptools-compat`, `20-torch-repair`,
`40-sageattention`, `50-onnxruntime-gpu`). The latter use the contract
extensions below (`MOD_CRITICAL`, `MOD_STREAM`, `mod_export`, `mod_prerun`) —
source-patch mods don't need any of them.

## Anatomy of a mod

```
mods/
  _lib/mod_common.sh        # shared helpers (not a mod; the leading _ skips it)
  05-setuptools-compat/
    run.sh                  # the contract (required)
  10-unified-memory-free/
    run.sh
    transform.py            # supporting file(s) (optional)
```

## The contract

Every `run.sh` is sourced (not executed) by the main script with `_lib/mod_common.sh`
already loaded and these variables exported: `INSTALL_DIR`, `VENV_DIR`, and
`MOD_DIR` (the mod's own directory). It must define three shell functions:

| Function | Returns | Purpose |
|---|---|---|
| `mod_describe` | echoes one line | Human description, shown in `doctor`/summaries |
| `mod_apply` | echoes a status word, returns 0 | Applies the mod idempotently |
| `mod_verify` | exit 0 = active, 1 = not | Checks whether the mod is currently in effect |

`mod_apply` should echo one of: `applied`, `present` (already there),
or `skipped:<reason>` — the helpers in `mod_common.sh` do this for you.
(Streamed mods report status differently — see below.)

It may optionally define a fourth function:

| Function | Returns | Purpose |
|---|---|---|
| `mod_prerun` | returns 0/1 | Runs before **every** `run`, not just install/update. Absence is a silent no-op — only define this if your mod needs a cheap pre-launch guard. |

## Writing a source patch

Most source-patch mods edit a ComfyUI Python file. Use the helpers:

```sh
MOD_TAG="my_fix"
MOD_FILE="comfy/somefile.py"
mod_describe() { echo "what this does"; }
mod_apply()    { py_patch_file "$MOD_FILE" "$MOD_TAG" "$MOD_DIR/transform.py"; }
mod_verify()   { py_marker_present "$MOD_FILE" "$MOD_TAG"; }
```

`transform.py` reads the source on stdin, writes the patched source to stdout,
and **must echo the input unchanged if it can't find its anchor** (that's how
"upstream changed" is detected). The marker string arrives via `$MARKER`.

`py_patch_file` handles everything else: the idempotency check, a one-time
`.spark-orig` backup, and a post-write `ast.parse` guard that reverts the file
if the patch would have produced invalid Python. Patches are re-applied after
every `git pull`, so they self-heal across ComfyUI updates.

## Writing a venv-package mod (critical / streaming / stateful)

Building SageAttention, repairing a shadowed torch, etc. don't fit the
default contract as-is: they can take minutes (buffering their output until
`mod_apply` returns would hide all progress), their failure genuinely breaks
the install (should abort loudly, not degrade to a soft skip), and their
caller sometimes needs a value back (e.g. the update summary's "SageAttention:
rebuilt & verified" line). Three opt-in additions cover this:

```sh
MOD_CRITICAL=1   # a nonzero exit from mod_apply/mod_prerun aborts the whole
                 # script (die) instead of being reported as skipped:error.
                 # Use for steps whose failure means the install/launch is
                 # genuinely broken. Leave unset for anything where a failure
                 # should just mean "this optimization is inactive" — that's
                 # the right default; most mods should NOT set this.

MOD_STREAM=1     # mod_apply/mod_prerun output streams live to the terminal
                 # instead of being buffered until it finishes. Set this for
                 # anything that can take more than a few seconds.
```

A streamed mod can't rely on its echoed stdout for status (stdout isn't
captured) — report status via `mod_export` instead, and use it for any other
value the caller needs:

```sh
mod_apply() {
  ...
  mod_export "STATUS=applied rebuilt & verified"
  mod_export "SAGE_ACTION=rebuilt & verified"   # read back by cmd_update
}
```

`mod_export KEY=value` appends to `$MOD_STATE_FILE`; the runner reads it back
after the mod returns and sets each `KEY` as a global in the calling
function's scope, whether or not the mod is streamed — only the `STATUS=`
key's meaning is streaming-specific (it replaces the echoed-stdout status
line; non-streamed mods keep using the classic echoed-first-token protocol
from "The contract" above). `50-onnxruntime-gpu` exports `ORT_STATE` this
way, and is itself streamed too (its wheel download can take a while).

## Disabling mods

Set `SPARK_SOURCE_PATCHES=0` in the environment to skip all mods, including
the pre-launch `mod_prerun` guard.
