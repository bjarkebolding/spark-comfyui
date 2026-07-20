# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with
code in this repository.

This file orients Claude Code (or any contributor) to the project so work can
continue without re-deriving the history. Read it fully before editing.

Style rule for every markdown file in this repo, including release notes:
plain and direct. No dash punctuation in prose (no em or en dashes), no
rhetorical flourishes, short sentences. Hyphens in compound words, flags,
dates and list markers are fine. Console captures stay verbatim even when
program output contains dashes.

## What this is

`spark-comfyui` is a single-entry-point bash tool that installs, runs,
updates, and maintains [ComfyUI](https://github.com/Comfy-Org/ComfyUI) on the
**NVIDIA DGX Spark (GB10 Grace Blackwell)**. The Spark is unusual hardware:
aarch64 Grace CPU, an sm_121 Blackwell GPU most toolchains don't target yet,
and 128 GB of unified CPU/GPU memory. A generic ComfyUI install either fails
or runs in silently degraded states. This tool makes the whole lifecycle
automatic and self-healing.

Author/owner: GitHub `bjarkebolding`. Target repo name: `spark-comfyui`.
Hardware in use: DGX Spark, hostname `sparky`, install root `~/comfyui-spark/`.
Published: https://github.com/bjarkebolding/spark-comfyui.
Current version: **2026.07.19** (MIT licensed, shellcheck-clean).

## Versioning and releasing

CalVer as of 2026-07-13: `YYYY.MM.DD`, plus `.N` for a second
behavior-changing release the same day. Semver was dropped because push
cadence made it meaningless (pushing to main IS releasing). A version's only
job is to stamp which behavior state a bug report ran. The 1.4.0 to 2026.x
transition sorts correctly under `sort -V`, and `doctor`'s update probe is
git-ancestry-based, so the format is cosmetic to tooling.

**Self-update pulls main HEAD, so pushing to main IS releasing. Always bump
VERSION in the same push.** Docs-only pushes need no bump.

## Golden rules (do not regress these)

1. **shellcheck-clean is non-negotiable.** Run `shellcheck -S warning
   spark-comfyui.sh mods/*/run.sh mods/_lib/mod_common.sh` before every commit.
   Mod `run.sh` files are sourced fragments and carry `# shellcheck shell=bash`.
2. **The main script must stay relocatable.** All paths derive from
   `BASE_DIR="$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")"`. Never hardcode
   `$HOME` or absolute install paths (the only legit `$HOME` uses are the
   systemd user unit dir and `loginctl`). `mods/` must sit next to the script.
3. **Every optimization is backed by a functional gate, not a heuristic.**
   Warnings guide; a live test decides. Example: SageAttention is only enabled
   after a real multi-shape GPU kernel run passes, never on the strength of a
   version string or a `--help` grep. Two prior bugs came from trusting
   heuristics: `TORCH_CUDA_ARCH_LIST="12.0"` and grepping `ptxas --help` for
   `sm_121`.
4. **Idempotent and self-healing.** `install` and `update` are safe to re-run;
   they skip or refresh, never duplicate or break. Anything that patches the
   ComfyUI tree must re-apply cleanly after `git pull`.
5. **Fail loud on real problems, quiet when healthy.** No silent degradation,
   and no false alarms: benign platform noise (e.g. the aarch64
   `nvidia-*-cu13 is not supported on this platform` line from `pip check`) is
   filtered, not surfaced as a failure. This rule governs the `--watch`
   dashboard too: rows render only when they carry information (see the watch
   section below).
6. **Test patches against fixtures before shipping.** Source patches edit
   upstream Python; always dry-run the transform on a realistic fixture and
   confirm the result still `ast.parse`s.

## Repository layout

```
spark-comfyui.sh          # The entry point: install/run/update lifecycle +
                          # dispatch, ~1700 lines. GB10 venv-package logic
                          # (torch, SageAttention, onnx) lives in
                          # mods/_lib/mod_common.sh, not here.
mods/                     # discovered, applied and self-healed automatically
  _lib/mod_common.sh      # shared helpers: py_patch_file, py_marker_present,
                          # mod_export, and the GB10 venv-package functions
                          # (need_nvcc, sage_kernel_ok, onnx_gpu_ok,
                          # ensure_onnx_gpu, ensure_setuptools_compat,
                          # repair_torch, torch_cuda_diag,
                          # build_and_verify_sage). Sourced by
                          # the main script itself at startup, not just mods.
  05-setuptools-compat/   # setuptools pinned within torch's own constraint
    run.sh
  10-unified-memory-free/ # get_free_memory() -> host-available unified pool
    run.sh transform.py
  20-torch-repair/        # torch CUDA verified/repaired; also mod_prerun
    run.sh                # (runs before every `run`, not just install/update)
  30-manager-config/      # ComfyUI-Manager config.ini (network_mode, uv,
                          # downgrade_blacklist). NOT pip_auto_fix.list, see
                          # configure.py's docstring (crashed Manager's own
                          # version parser on every launch, retired 2026-07).
    run.sh configure.py
  40-sageattention/       # native sm_121 build + live kernel verification
    run.sh
  50-onnxruntime-gpu/     # community sm_121 GPU wheel for preprocessors
    run.sh
  README.md               # how to author a mod (incl. MOD_CRITICAL/MOD_STREAM)
README.md LICENSE .gitignore CLAUDE.md
```

The script installs ComfyUI, the venv, and SageAttention next to itself
(`ComfyUI/`, `comfyui-env/`, `SageAttention/`), and `backup` writes to
`backups/`, all gitignored.

## Commands (dispatch at the bottom of the script)

`install` · `run [args]` · `stop` · `update [--torch|--rollback]` · `doctor` ·
`status [--watch [SEC]]` · `tune [--clock-cap MHZ] [--persist]` ·
`backup [--with-output] [FILE]` · `restore FILE` · `reset [--yes]` ·
`service` · `--version`. Hidden back-compat aliases: `verify` (doctor), `monitor`
(status --watch), `rollback` (update --rollback), `bench` (removed, prints
A/B guidance).

Mental model, also printed in `--help`: install once, then run; update now
and then; something feels wrong, run doctor. `run` also runs a cheap
pre-launch mod pass (`apply_prerun_mods`, currently just `20-torch-repair`'s
guard against a custom node having swapped in CPU torch since the last
launch). `update` self-updates the tool itself first (`self_update`: ff-only
git pull of BASE_DIR when upstream is strictly ahead, then re-execs the fresh
script exactly once; bash must never keep executing a file that changed under
it).

## The mod system (most important architecture)

A "mod" is a self-contained, idempotent, self-healing unit that
`spark-comfyui.sh` applies during install/update and verifies in `doctor`.
Two flavors share one contract: **source-patch mods** that edit ComfyUI's own
Python (`10`) or config tree (`30`), and **venv-package mods** that
install/verify/repair virtualenv state (`05`, `20`, `40`, `50`). The four
functions that used to live directly in the main script
(`ensure_setuptools_compat`, `repair_torch`, `build_and_verify_sage`,
`ensure_onnx_gpu`) now live in `mods/_lib/mod_common.sh`, wrapped by thin
`mods/NN-name/run.sh` files. Every other direct call site of those functions
(`cmd_run`'s SageAttention shadow-check, `cmd_rollback`, `cmd_doctor`'s
diagnostics) still calls them directly. Only install/update/pre-run
orchestration goes through the mod pass.

Each `mods/<name>/run.sh` is **sourced** (not executed) in a subshell with
`_lib/mod_common.sh` preloaded and `INSTALL_DIR`, `VENV_DIR`, `MOD_DIR`
exported. It must define three shell functions:

- `mod_describe`: echoes a one-line description
- `mod_apply`: applies idempotently, echoes a status, returns 0
- `mod_verify`: exit 0 if currently active, 1 if not

and may optionally define a fourth:

- `mod_prerun`: runs before every `run`, not just install/update. Absence
  is a silent no-op; only `20-torch-repair` defines it.

**Status protocol** (parsed by `_invoke_mod`/`apply_source_patches` in the
main script): the first token of `mod_apply`'s output is the class
(`applied`, `present`, or `skipped`) and any remainder is human detail shown
to the user. Example: `applied config updated (network_mode, use_uv)`.

**Extended contract for venv-package mods**: three opt-in top-level
declarations a `run.sh` can set, all unused by the two source-patch mods.

- `MOD_CRITICAL=1`: a nonzero exit from `mod_apply`/`mod_prerun` aborts the
  whole script (`die`) instead of being swallowed into `skipped:error`. Set
  by `20` and `40` (torch/Sage failure means the install is genuinely
  broken). NOT set by `05`, `10`, `30`, `50` (their failure just means one
  optimization is inactive, generation still works).
- `MOD_STREAM=1`: output streams live to the terminal instead of being
  buffered until `mod_apply` returns (buffering would hide all progress
  during SageAttention's 10-30 min build or onnxruntime's ~220 MB wheel
  download). Set by `20`, `40`, `50`. A streamed mod reports status via
  `mod_export STATUS=<word>` instead of an echoed line, since stdout is no
  longer captured.
- `mod_export KEY=value`: appends to `$MOD_STATE_FILE`; the runner reads it
  back after the mod returns and sets `KEY` as a global in the caller's
  scope. This is how `cmd_update`'s summary gets `SAGE_ACTION` (from `40`)
  and `ORT_STATE` (from `50`), instead of the mod's function setting them as
  plain globals directly.

`cmd_update` also exports `SPARK_TORCH_UPGRADED` (the `--torch` flag) before
the mods pass runs, so `40-sageattention`'s `mod_apply` can use it as an
extra forced-rebuild trigger alongside its own marker/git-rev-drift check.

For Python-source patches use the helpers in `mod_common.sh`.
`py_patch_file <rel> <tag> <transform.py>` handles the marker/idempotency
check, a `<file>.spark-orig` backup (refreshed on every apply so the revert
below never restores stale upstream code), and an `ast.parse` guard that
**reverts the file if the patch would produce invalid Python**. The
`transform.py` reads source on stdin, writes patched source to stdout, and
MUST echo input unchanged when its anchor isn't found. That is how "upstream
moved the code" is detected and reported as `skipped:anchor-not-found`. The
marker string arrives via `$MARKER`.

Mods run in **filename order** (numeric prefix; this is what gives `05`
setuptools before `20` torch before `40` Sage). `_`-prefixed dirs are skipped.
Disable all mods, including the pre-run guard, with `SPARK_SOURCE_PATCHES=0`.

Patched files sit MODIFIED in ComfyUI's working tree between updates. That
is why `sync_comfyui` reverts every tracked modified file carrying the
`# spark-comfyui:` marker before its branch switch and ff-only merge (the
mods pass re-applies them right after): without the revert, the first
upstream commit touching a patched file kills `update` on git's "local
changes would be overwritten" error (reproduced 2026-07-18). User edits
without a marker are deliberately left alone. `cmd_rollback` runs
`apply_source_patches` after its hard reset for the same reason: the reset
wipes the patches.

Adding a source-patch mod: drop a new `mods/NN-name/` dir; no main-script
edits needed. A venv-package mod needs its underlying logic added to
`mod_common.sh` too if it doesn't already live there.

## status --watch

Everything lives in `cmd_status` plus helpers (`_watch_row`, `_watch_hdr`,
`_watch_comfy`, `_series_nonzero`/`_series_any`). These decisions are
field-verified; don't re-derive them.

**Two outputs per tick, different jobs.** The append-only
`thermal_monitor.log` line is the primary output: it survives the silent
hard-reboots this exists to diagnose, and it carries **every field
unconditionally** (`GPU PWR SM UTIL RAM SWAP CPU CACHE LOAD IO RSS CGPU OGPU
PST EVT GEN ACT ITS LAT Q HIT`). The dashboard is the live tty view of the
same samples; when stdout isn't a tty, plain log lines are emitted instead.
Some fields are log-only by design (RSS, the CGPU/OGPU per-process GPU
split, CACHE, LOAD, raw EVT hex): they matter in a post-mortem, not on
screen.

**Quiet-when-healthy rendering** (golden rule 5 applied to the dashboard).
Always-on rows: GPU temp/power/sm-clk/util, unified memory, CPU, disk I/O.
That is seven rows idle. Conditional rows render only with a story:
`throttle` (any slowdown bit in the window), `swap` (only if swap exists at
all), `gen`/`it/s`/`latency`/`queue`/`hit rate` (only with data). Ring
buffers always advance, so a row appears with its window history intact.
Visibility tests use `_series_nonzero`/`_series_any`. They exist because of
a bash gotcha: `${arr[*]//pat/}` substitutes per-element and re-adds the
joining spaces, so it can never yield an empty string; join to a scalar
first.

**Renderer** (`_watch_row`, one awk per row): per-glyph heat colors by
absolute thresholds. Temp 70/80, power 60/80 (red power IS the GB10
overcurrent zone), unified memory 85%/95% of the pool. Unthresholded rows
render in one accent color. Trend arrows are dead-banded to 5% of the window
span. Colors wrap the padded value text so escape bytes never skew columns.
Bar glyphs are split into an awk array (not substr) for mawk/gawk parity.
Section rules use `sed`, not `tr` (tr is byte-oriented and shreds the
multibyte `─`). The window width adapts to the terminal.

**Generation telemetry** (`_watch_comfy`, one python call per tick) polls
three stock HTTP endpoints: `/history`, `/queue`, `/internal/logs/raw`.
HTTP polling is BY DESIGN, not a websocket: per-step `progress_state` ws
events go only to the prompt's owning client (`comfy_execution/progress.py`),
so a passive `/ws` listener would never see other clients' step progress.
Live `it/s` is the newest tqdm rate scraped from the server's terminal ring
buffer, only while a prompt is in flight. `hit rate` is the node count in
the `execution_cached` message divided by the prompt's node count. An
errored gen is labeled. A gen in flight at crash time points to overcurrent.

**Latency** (submission to saved): the loop timestamps queue ids on first
sight; when one finishes, latency = now minus first-seen. The latency check
MUST run before the seen[] populate/prune. On the very tick a gen finishes
the queue is already empty and idle, and pruning first would wipe the
timestamp. Only covers jobs submitted while the watch runs.

**Session A/B summary** (the `session:` line): every gen that finishes
while the watch runs is recorded exactly once via fin_id-change accounting.
Whatever fin_id says on the FIRST tick is only a baseline, so history
predating the watch never skews a bench. Distilled to `N gens · first Xs ·
steady ~Ys (lo–hi) · ~Z it/s · E errored`. First carries the model load,
steady excludes it, mirroring the 2026-07-13 bench methodology. Stats reset
per launch: one watch session per A/B condition gives one comparable line.
The GENERATION section header names the attention backend
(SageAttention/SDPA), the usual A/B dimension.

**Dead/dropped gauges. Don't re-add:** `utilization.memory` reads a
constant 0 on GB10 even at 90 W (probed 2026-07-14; a dead gauge in a
diagnostic log misleads). `sw-power-cap` (0x4) is benign and constant on
GB10 even at idle; only the four HW/thermal-slowdown bits
(0x08/0x20/0x40/0x80) alarm, and the raw hex stays in the log's EVT field.
A dedicated pstate row duplicated the sm-clk story (P0 generating, P8 idle);
it is a tag on the sm-clk row now. GB10 nvidia-smi N/A fields: `clocks.mem`,
`fan.speed`, `temperature.memory`, `power.limit` (and `nvidia-smi -pl` does
not work, hence `tune --clock-cap`).

## reset / backup / restore

Three commands, one shared definition of "user content": the readonly
`USER_CONTENT` array near the top of the script (`models user input output
custom_nodes extra_model_paths.yaml`). `reset` iterates it directly.
`backup`/`restore` cover the same set but handle each entry with its own
rules (manifests, filters), so they do not loop over the array; keep the
three in sync by hand when the set changes.

**reset (`cmd_reset`)**: wipe `INSTALL_DIR`, `VENV_DIR`, `SAGE_SRC` and rerun
`cmd_install`, preserving user content via a hold directory. The hold dir is
a sibling of the install dir (`.spark-reset-hold`), so every `mv` is a
same-filesystem rename: instant even for 74 GB of models. Protocol, in order:

1. Confirmation: the user must type the word `reset`; `--yes` skips it and
   is required when stdin is not a terminal.
2. Hold move: each `USER_CONTENT` entry is `mv`'d into the hold dir.
   No-overwrite rule: if an entry already exists in the hold dir (left by an
   interrupted earlier reset) AND in the install dir, `die`. Guessing which
   copy is the user's is not this script's call.
3. Wipe guard: before any `rm -rf`, re-check that no `USER_CONTENT` entry
   remains under `INSTALL_DIR`. This is the line between "reset" and
   "deleted the models after a failed mv".
4. `cd "$BASE_DIR"` (the wipe may delete the caller's cwd, which would break
   git/pip inside `cmd_install`), wipe, write `wiped` to `$hold_dir/.phase`,
   then `cmd_install`.
5. Move everything back (the fresh-from-git skeleton dirs lose to the
   user's copies), then `git checkout -q --` each `USER_CONTENT` entry: the
   held dirs carry stock tracked files from the old checkout
   (`custom_nodes/websocket_image_save.py`, `models/configs/*`) and putting
   them back at the fresh HEAD keeps the tree clean for `update`'s ff-only
   merge. `checkout --` never touches untracked user files. Finally remove
   `.phase` and `rmdir` the hold dir.

Resume cases (re-running converges): interrupted before the wipe (no
`.phase`), the hold move just continues, with the no-overwrite rule catching
real collisions; interrupted anywhere after the wipe (`.phase` says
`wiped`), `cmd_install` reruns unconditionally. A `.git` dir is no proof the
interrupted install finished (clone is its first step), so there is
deliberately no skip-install shortcut; `cmd_install` is idempotent and
skips or refreshes whatever did complete. The hold dir lives inside the
repo checkout on a default install, so `.gitignore` lists
`.spark-reset-hold/` (a `git clean -fd` must never see held models as
junk).

**backup (`cmd_backup`)**: writes to `FILE` or a timestamped default under
`backups/` next to the script. Only the small generated entries (meta,
manifests, plain-node copies, config files) are staged in a mktemp dir
(cleaned by an EXIT trap on failure); `user/`, `input/` and `output/` are
tarred straight from the live tree with `--exclude` for logs and caches. tar
runs with `--ignore-failed-read`, and exit status 1 (a file changed or
vanished mid-read) is a warning, not a failure; that tolerance is what makes
"safe while ComfyUI is serving" true. Archive format (top-level entries,
`format=1`):

- `meta`: `format=1`, tool version, ISO date, hostname, ComfyUI commit.
- `user/`: `INSTALL_DIR/user` minus `*.log` files and `__pycache__`.
- `input/` (only if non-empty); `output/` only with `--with-output`.
- `comfyui-patches.list` and `extra_model_paths.yaml` if present.
- `custom-nodes.manifest`: one `name<TAB>origin-url<TAB>commit` line per
  git-cloned node. Entries tracked by ComfyUI's own git (checked with
  `git ls-files`) are excluded, so stock files like
  `websocket_image_save.py` never leak into the manifest.
- `custom_nodes_plain/`: full copies of non-git nodes (minus `__pycache__`).
- `models.manifest`: `bytes<TAB>relpath` for every file under `models/`,
  sorted. Models are NEVER archived; the live install's backup was 3.7M
  against 74 GB of models. Safe while ComfyUI is serving.

**restore (`cmd_restore`)**: order matters. Unpack to a mktemp stage (EXIT
trap cleans it on failure) and check `format=1`; `cmd_install` if
`INSTALL_DIR/.git` or the venv is missing (self-heals a half-gutted
machine); `cmd_stop` unconditionally (it reports "not running" itself); merge `user/`, `input/`, `output/` (`cp -a` over the live
tree); restore the two config files, saving a differing live copy aside as
`.bak` first; custom nodes (plain copies, then manifest clones with a
detached checkout of the pinned commit; every freshly restored node's
`requirements.txt` is pip-installed, plain copies included; a checkout miss
and a failed per-node `pip install -r` are warnings, not failed restores;
clone and pip run with stdin from `/dev/null` so a prompting clone cannot
eat the manifest lines the loop is reading); `apply_source_patches`, because node
pip installs can clobber torch and the mods pass re-verifies it; finally
diff `models.manifest` against disk and list every missing file with its
size. Idempotent: restoring onto a healthy install reports everything
present.

**doctor**: an info line (not pass/fail) in the self section names the
newest `backups/spark-backup-*.tgz` and its age in days, or prints
`Backup: none in backups/ (run: spark-comfyui.sh backup)`. It only knows
about the default `backups/` dir; backups taken with an explicit FILE
argument elsewhere are invisible to it, which the wording reflects.

**Source guard**: just before the dispatch block,
`if [[ "${BASH_SOURCE[0]}" != "$0" ]]; then return 0; fi`. Sourcing the
script defines all functions and returns before dispatch, so test harnesses
can source it and stub commands. Caveats: sourcing still runs the top-level
code (env defaults, sourcing `mod_common.sh`), and `USER_CONTENT` is
`readonly`, so sourcing twice in one shell errors.

## GB10 domain knowledge (do not relitigate)

- **PyTorch**: cu130 aarch64 wheels. Install BEFORE ComfyUI requirements so
  nothing pulls CPU-only torch. The `sm_121 exceeds torch's max` startup
  warning is expected and harmless (PTX JITs).
- **SageAttention**: build from source with
  `TORCH_CUDA_ARCH_LIST="12.1+PTX"` (native sm_121 cubin + PTX fallback).
  `"12.0"` alone produced sm_120 cubins with no PTX and failed with
  `no kernel image`. Needs CUDA 13's `ptxas` (check by version >= 13, NOT by
  grepping --help). Mandatory; gated on a live multi-shape kernel test.
  10-30 min build. `build_and_verify_sage` (mod `40-sageattention`,
  `MOD_CRITICAL=1 MOD_STREAM=1`: build output streams live, failure aborts
  the script). **Pinned via `SAGE_REF`** (not tracking upstream's default
  branch). The live kernel test checks shape/finiteness, not visual
  correctness, so it can't catch a regression like 3.x's mosaic artifacts on
  GB10; the pin is the actual gate, per golden rule 3. Bumping it is a
  deliberate decision (edit the default in spark-comfyui.sh, or override
  `SAGE_REF`), not something that happens on its own. Currently pinned to a
  commit 38 past the `v2.2.0` tag, field-verified on GB10 sm_121.
- **Flash Attention**: FA3 can't target sm_121 at all. FA2 2.8.3 can be
  compiled (`TORCH_CUDA_ARCH_LIST="12.0"`, ~2h) but loses to SDPA on
  Blackwell. Not installed; only worth it if a custom node hard-imports
  `flash_attn`.
- **onnxruntime**: no PyPI aarch64+cu13 GPU wheel; a community sm_121 wheel
  is installed (URL in `ORT_WHEEL_URL`). Detect via
  `get_available_providers()`, not startup logs. A later PyPI `onnxruntime`
  install shadows it (same import path, no pip conflict); `update`/`doctor`
  re-detect and repair.
- **Unified memory**: swap ON plus heavy load means a silent whole-box freeze
  (no OOM kill). `tune` disables swap. `--gpu-only`/`--highvram` HURT here.
  bf16 flags are the native fast path (`SPARK_BF16=0` to disable).
- **`--high-ram` (upstream PR #14437, merged 2026-06): A/B-tested 2026-07-13,
  REJECTED.** Full 2x2 matrix (pinning x high-ram), Krea-2 Turbo template,
  1024², 8 steps, fresh launch per condition. All three of the flag's code
  paths gate on pinned memory (`ensure_pin_budget`, `pinned_hostbuf_size`,
  `handle_pin` in `comfy/ops.py`) and `pin_memory()` no-ops under our
  `--disable-pinned-memory`, so on the production config it reduces to
  forcing `--cache-classic`. Measured: no change (<=3%, within noise). In
  its real mode (pinning enabled) it was strictly worse on GB10: first-gen
  19.4s vs 18.4s, LoRA-repatch 18.9s vs 15.0s (+26%), steady 13.5s vs 12.8s,
  and ~17 GB extra RAM pinned (uncapped `pinned_hostbuf_size` is 2x model
  size, which would also starve co-resident vLLM). Its designed benefit
  (forcing residency vs pagefile) is moot here: `tune` already disables
  swap. The same bench re-validated `--disable-pinned-memory` against the
  reworked 2026-06 pinning code: pinning-enabled-without-high-ram was
  identical to baseline within noise (12.69s vs 12.76s steady), so the flag
  stays. No cost, and it inoculates against the uncapped-pin path.
- **get_free_memory cliff** (mod 10): `cudaMemGetInfo` under-reports free mem
  when another CUDA process (vLLM) is resident, causing needless offload and
  5-15x slower sampling. Fixed by reading
  `psutil.virtual_memory().available`. The owner DOES run co-resident vLLM,
  so this matters in practice.
- **Overcurrent reboots**: some units hard-reboot (no logs) on the ~85W power
  spike at denoise start. `tune --clock-cap 2100` caps clocks (nvidia-smi -pl
  is N/A on GB10). `status --watch` captures the pre-crash telemetry.
- **NVRTC**: aarch64 cu130 wheels bundle none, so torch uses the system CUDA
  13 lib, which is the GOOD state. `doctor` distinguishes the four cases.
- **Dependency pins**: torch pins `setuptools<82`; never blindly upgrade
  setuptools. `ensure_setuptools_compat` (wrapped by mod
  `05-setuptools-compat`) reads torch's own constraint. `doctor` runs
  `pip check` (filtering the benign cusparselt platform line).
- **ComfyUI-Manager's pip_auto_fix.list is incompatible with CUDA torch
  builds** (found 2026-07): `prestartup_script.py` calls `fix_broken()`
  unconditionally on every launch, which parses pinned versions with its own
  `StrictVersion`, a naive per-`.`-segment `int()` parser that crashes on
  any PEP 440 local version segment (e.g. torch's own `2.13.0+cu130`). It
  crashes the same way on the installed version during drift comparison, so
  no pin format on our side avoids it. Don't relitigate reformatting the
  pin. `mods/30-manager-config` no longer writes this file (and deletes a
  stale one from before this fix); `downgrade_blacklist` (independent
  mechanism, unaffected) plus `20-torch-repair`'s real
  `torch.cuda.is_available()` checks are the actual protection.

## Containerization (container-dev branch, EXPERIMENTAL, the roadmap)

The native path above becomes legacy once this matures. Local-only until
ready: nothing container-related is pushed to GitHub yet, and container-dev
deliberately has no upstream so a bare `git push` errors. Development home
is `~/projects/spark-comfyui` (remotes: `origin` GitHub, `live` the old
checkout at `~/spark-comfyui`, which is still the production install).

Split: the image holds everything reproducible (ComfyUI at a pinned commit,
venv with cu130 torch, native sm_121 SageAttention at SAGE_REF, the
sha256-pinned onnxruntime wheel, build-time mods 05+10 applied via
container/build-mods.sh reusing the mods/ contract). The USER_CONTENT set is
bind-mounted by `container run` from this checkout's ComfyUI/ dirs. A named
volume `spark-comfyui-cache` at /home/comfy/.cache carries pip downloads and
compiled sm_121 kernels across container recreation (CUDA_CACHE_PATH points
into it); the mountpoint must exist in the image owned by comfy (uid 1000)
or docker creates it root-owned and pip/uv fail.

`container/entrypoint.sh` is the runtime half of the mod system, every
start: custom-node requirements reinstall (container layer is ephemeral;
runs BEFORE the torch guard because node installs are what clobber torch),
mod 20 prerun with torch_cuda_diag, the live SageAttention kernel gate
(builds have no GPU, so golden rule 3 lives here; failure refuses to
launch), mod 30 onto the mounted user/ dir, then main.py with the native
flag set. Hardening on `container run`: non-root, cap-drop ALL,
no-new-privileges, GPU only, --rm (stateless), 1 GB shm.

Commands: `container build | run | stop | update [--rollback] | status |
doctor | shell`. build resolves upstream master to a SHA and passes it as
COMFY_SHA (the docker cache key: a ComfyUI bump rebuilds only from the
clone layer; torch/Sage layers stay cached). update self-updates the tool
first (SELF_UPDATE_RESUME makes self_update's re-exec land back in
`container update`), rebuilds, and keeps the replaced image as :previous;
--rollback swaps :latest and :previous (toggles). status is
quiet-when-healthy; its one warning is a running container whose image is
no longer :latest. doctor runs the four live gates (torch+diag, sage
kernel, onnx provider, kitchen NVFP4) inside a throwaway --gpus container.

Field-learned docker gotchas (do not re-derive):
- The containerd image store garbage-collects a tagless image INSTANTLY.
  cmd_container_update therefore holds `:pre-update` on the old image
  through the build and promotes it to :previous only on a real change.
- buildx provenance attestations stamp each build's manifest with the build
  time, so identical cached builds got different "image IDs" until
  `--provenance=false`. Without it, update's changed-vs-current comparison
  always says changed.
- Base image pinned to the CUDA 13.0 patch line (13.0.3): the r580 driver
  and cu130 torch wheels are 13.0-era, and a 13.1+ ptxas can emit PTX the
  driver JIT rejects. Bumping past 13.0.x is a deliberate pin change with a
  field test, same policy as SAGE_REF.
- The stock GLSL nodes (comfy_extras/nodes_glsl.py) dlopen bundled ANGLE
  libs that link libX11/libXext/libxcb even headless; those apt packages
  are in the image for that reason alone.
- .dockerignore whitelists the build context (script, mods/, container/):
  BASE_DIR on a live install holds 100+ GB next to the Dockerfile.

Phase status: phase 1 (build/run/stop/shell) and phase 2 (update/rollback/
status/doctor) implemented and field-verified on the GB10. Phase 3 open:
image slimming (multi-stage), port-binding choice (127.0.0.1 vs LAN),
restore parity for container-only installs, README/docs, release plan.
backup already works against a container install as-is (it never touches
the venv); restore works but its self-heal path still runs the NATIVE
cmd_install.

## Env var overrides

- `INSTALL_DIR`, `VENV_DIR`, `SAGE_SRC`, `REPO_URL`, `PORT`, `TORCH_INDEX`,
  `ORT_WHEEL_URL`, `PATCH_LIST`, `MODS_DIR`, `PIP_RETRIES`,
  `PIP_DEFAULT_TIMEOUT`
- `SAGE_REF`: pinned SageAttention commit, see GB10 domain knowledge
- `SPARK_BF16` (default 1)
- `SPARK_STATIC_VRAM` (default 0). Caveat: with SageAttention this can hang a
  second sampling pass on GB10, open ComfyUI issue #13920.
- `SPARK_SOURCE_PATCHES` (default 1). 0 disables all six mods, not just
  source patches.
- `SPARK_SELF_UPDATE` (default 1). 0 stops `update` from git-pulling the
  spark-comfyui repo itself.

## Patch list (separate from mods)

`comfyui-patches.list` next to the script lets users merge upstream
PRs/branches on top of master onto a rebuilt-fresh `spark-patched` branch
each update. Format: `pr:<N>` | `branch:<name>` | `remote:<url> <branch>`.
A commented template is seeded on first install. As of mid-2026 the big
Spark PRs are merged upstream, so the default (empty) list means plain
master tracking, which is optimal.

## Development workflow that has worked

Tight empirical loop: propose a change, the user runs it on the real Spark,
the real error comes back, diagnose from the actual output, fix, re-test.
The hardware is the source of truth; do not trust assumptions about GB10
behavior without a test or a cited field report. When adding source patches,
dry-run the transform on a fixture and `ast.parse` the result first.

## Likely next tasks / open threads

- Watch for ComfyUI refactors that move `get_free_memory`. Mod 10 will report
  `skipped:anchor-not-found` in `doctor` if so, and the transform's
  regex/anchor needs updating.
- Mod `20-no-double-vram` was retired (2026-07): upstream moved the
  `weight = weight.to(device=device_to)` line it patched out of
  `comfy/utils.py` entirely, into `comfy/model_management.py`'s `cast_to()` /
  `cast_to_device()`. Tracing the new load path (`comfy/ops.py:362`) shows the
  common load-time cast already calls `cast_to(..., copy=weight_has_function,
  ...)`, which is falsy for a plain weight with no LoRA/hook. Upstream now
  avoids the transient double-allocation itself. If a future profile shows
  it's still happening, the fix would target `cast_to`/`cast_to_device` in
  `model_management.py`, not the old `utils.py` anchor. The `20` prefix was
  later reused for the unrelated `20-torch-repair` mod (2026-07, same
  refactor that moved venv-package steps into mods/). No connection between
  the two beyond the number.
- The community ONNX wheel URL and the "PRs already merged" note in the patch
  template will age; refresh periodically. The URL carries a `#sha256=` pin
  since 2026.07.18 (pip verifies it before installing); a URL refresh must
  update the hash in the same edit, sourced from the HF API's lfs.oid and
  confirmed against a real download.
- **NVFP4 (verified 2026-07, no mod needed)**: `comfy-kitchen` is a stock
  ComfyUI dependency now (`ComfyUI/requirements.txt` pins it directly,
  currently 0.2.18), not something spark-comfyui.sh adds. Its NVFP4 kernels
  need SM >= 10.0 (Blackwell); GB10's sm_121 qualifies. Its fast native
  `cuda` backend (vs the slow pure-PyTorch `eager` fallback) needs torch
  compiled with CUDA >= 13, see `comfy/quant_ops.py`'s
  `ck.registry.disable("cuda")` gate. That is exactly what
  `20-torch-repair`/the cu130 install already guarantee, so NVFP4 rides for
  free on the existing torch guarantee. Confirmed live via
  `comfy_kitchen.list_backends()`: `cuda` backend `available: True` with
  `quantize_nvfp4`/`scaled_mm_nvfp4` etc. in its capabilities. Since
  2026-07-11 this is enforced by a real forced-kernel gate in `doctor`
  (`kitchen_nvfp4_ok`), not just the registry listing. One lever we
  deliberately don't touch: `--enable-triton-backend` (off by default in
  `comfy/quant_ops.py`, not passed by `cmd_run`) would additionally enable
  Kitchen's Triton backend as a third NVFP4 path. Unevaluated on GB10, don't
  adopt without a field test.
- Possible future mod candidates from community forums (evaluate, don't adopt
  blindly): SageAttention 3 (mosaic artifacts on Spark; as of the 2026-07
  research round this is corroborated in SageAttention's own tracker, issues
  #321 [GB10, the original mosaic report], #334, #340, #357 [active
  2026-07-09], across GB10/RTX 5060 Ti/5070 Ti, all open. Stay on the
  `SAGE_REF` pin; bumping past 2.2.x is a deliberate pin change, not
  automatic). The stardust7700 UMA fork (whole-fork merge via the patch
  list, opt-in flags).
- Watch list from the 2026-07 research round (re-check periodically):
  - SageAttention PR #372 (unmerged): Sage2 causal-mask fix for unequal
    q/kv lengths, the one real 2.2.x-line fix pending upstream. If it
    merges, that is the trigger to consider a new `SAGE_REF`. Note upstream
    `main` hasn't moved since our pin (we ARE the tip as of 2026-07-10).
  - `comfy-aimdo` (new ComfyUI pip dep, multi-threaded model loader built
    with DGX Spark in mind, PR #13802): touches the memory-loading path.
    On next install/update, confirm mod 10's anchor still applies.
  - `comfy-kitchen[cublas]` extra: RESOLVED 2026-07-11 by live test. The
    plain wheel's CUDA backend passes a forced NVFP4 quantize+matmul on GB10
    (cosine ~0.99 vs bf16 reference), so the extra is NOT needed on a stock
    install. `doctor` now gates this permanently (`kitchen_nvfp4_ok` in
    mod_common.sh forces the cuda backend via ck.use_backend, which
    verifiably raises instead of falling back). Also: the forum's
    "comfy-kitchen 0.2.61" claim is NOT a real version; latest is 0.2.18,
    already pinned by ComfyUI's requirements.txt.
  - ComfyUI issue #13920 (open): GB10 hang on second sampling pass with
    `--disable-dynamic-vram` + SageAttention (bisected upstream to commit
    1ac78180). Documented as a SPARK_STATIC_VRAM caveat in the env-var
    overrides section above (README no longer carries an env-var table,
    slimmed 2026-07). If it's fixed upstream, remove the caveat.
- Consider a GitHub issue template that asks for `spark-comfyui.sh doctor`
  output. For this project that one command is nearly a complete bug report.

## Release history (one line per release; details live in the GitHub Releases)

- **v1.0.0** (2026-07-10): first public release.
- **v1.1.0** (2026-07-10): backup-revert bug fix; runtime-fallback and
  stuck-clock doctor checks; `TRITON_PTXAS_PATH`; mod-state allowlist.
- **v1.2.0** (2026-07-11): NVFP4 live doctor gate; self-updating `update`.
- **v1.3.0** (2026-07-11): version banner on every invocation; doctor
  self-version and update-pending probe; status version line.
- **v1.4.0** (2026-07-13): first `status --watch` sparkline dashboard
  (temp/power/clock/util/RAM/CPU, interval arg, dated log lines,
  python-not-wrapper pid detection, plain-line fallback when not a tty).
- **2026.07.13.1** (2026-07-13): switch to CalVer; no functional change.
- **2026.07.14**: watch v2. Sectioned dashboard, per-glyph heat colors,
  trend arrows, adaptive window. Added P-state and decoded clock-event
  flags, page cache, load, disk I/O, per-process GPU memory, and generation
  telemetry (`_watch_comfy`). Log line gained the CACHE through HIT fields.
  Probed and dropped `utilization.memory` (dead on GB10). Dropped the static
  overcurrent hint lines (README troubleshooting carries that now).
- **2026.07.15**: watch declutter. Quiet-when-healthy conditional rows,
  pstate folded into sm-clk, page-cache/load-1m rows cut, UNIFIED MEMORY and
  SYSTEM sections merged. Added `_series_nonzero`/`_series_any`.
- **2026.07.15.1**: PROCESSES section replaced by GENERATION (header names
  the attention backend) plus the per-session A/B `session:` summary line.
  rss/gpu-self/co-res became log-only. Field-verified with 3 live Krea-2
  gens (first 15.6s, steady ~12.6s; pre-watch gens correctly not counted).
- **2026.07.16**: reset/backup/restore lifecycle commands (regenerate an
  install or move to a new Spark without losing user content; models
  manifested, never archived), the doctor backup info line, and the source
  guard before dispatch for test harnesses.
- **2026.07.16.1**: restore also pip-installs the requirements of plain
  (non-git) custom nodes, so Manager registry installs work on a
  fresh-machine restore.
- **2026.07.18**: hardening round, two passes in one release. Pass 1:
  onnxruntime wheel pinned by sha256 (pip verifies the URL fragment; hash
  confirmed against a live download), patch-list fetches prompt-proofed
  (GIT_TERMINAL_PROMPT=0 plus detached stdin, the restore-loop fix applied
  to the older loop), thermal log rotates to .1 at 50 MB on watch start
  (field-tested both directions), restore rejects path-traversal node names
  from tampered manifests, update fetches get timeouts (300s ComfyUI, 30s
  self) and a clean offline die, service unit gains
  Wants=network-online.target. Pass 2: sync_comfyui reverts marker-patched
  files before the ff-merge so update survives upstream touching a patched
  file (failure reproduced first, then regression-tested), tune --persist
  carries over a previously persisted clock cap instead of silently
  dropping it, rollback re-applies the mods pass after its hard reset,
  configure.py self-heals a corrupt config.ini on apply (verify stays
  read-only), restore tolerates corrupt manifest size fields and installs
  when the venv is missing, tune validates --clock-cap as 300-4000 MHz up
  front.
- **2026.07.19**: torch_cuda_diag. Every torch CUDA check failure (doctor,
  mod 20 apply and prerun) now prints the real torch.cuda.init() exception
  plus a host-vs-venv hint, instead of a bare AssertionError. Prompted by a
  GX10 field report where the correct cu130 wheel was installed but the
  host driver could not initialize CUDA; doctor's failure text no longer
  guesses "a custom node re-pinned it". All four diag branches exercised
  live on the GB10.

## Release checklist (repeat per release)

1. `chmod +x spark-comfyui.sh` stays committed (mode bit `100755` in the index).
2. `git add spark-comfyui.sh mods/ README.md LICENSE .gitignore CLAUDE.md`
3. Set `VERSION=` in spark-comfyui.sh to today's date, `YYYY.MM.DD`
   zero-padded (append `.N` if a behavior-changing release already went out
   today). Also update the header comment, the "Current version" line above,
   and add a Release-history line. Tag `v<VERSION>` to match `--version`.
4. README clone URL already set to `github.com/bjarkebolding/spark-comfyui`.
   The version strings inside README console captures are illustrative;
   refresh them when a capture is retaken, not on every release.
5. `gh release create v<VERSION> --title v<VERSION> --notes "..."`. Every
   version has a GitHub Release, not just a tag (v1.4.0 nearly shipped
   without one). Notes style: one-line theme, then `##` sections with the
   user-visible changes; end with the upgrade line. Console captures where
   they help. Release notes follow the same markdown style rule as the rest
   of the repo.
