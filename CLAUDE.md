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
**NVIDIA DGX Spark (GB10 Grace Blackwell)** — since the container cut,
entirely as a hardened docker container. The Spark is unusual hardware:
aarch64 Grace CPU, an sm_121 Blackwell GPU most toolchains don't target yet,
and 128 GB of unified CPU/GPU memory. A generic ComfyUI install either fails
or runs in silently degraded states. This tool makes the whole lifecycle
automatic, reproducible (the image), and confined (custom-node code cannot
touch the host).

Author/owner: GitHub `bjarkebolding`. Target repo name: `spark-comfyui`.
Hardware in use: DGX Spark, hostname `sparky`. Development home:
`~/projects/spark-comfyui` (sole remote: `origin` GitHub; the old native
install was deleted 2026-07-20 and content starts from a blank slate).
Published: https://github.com/bjarkebolding/spark-comfyui.
Current version: **2026.07.20.2**. No legacy branch: the last native
release is reachable as the v2026.07.19 tag, and the migration tooling
lives in the v2026.07.20 tag only. MIT licensed, shellcheck-clean.

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
   they skip or refresh, never duplicate or break. Since the cut this is
   largely structural: the image is immutable, patches bake into it from a
   fresh clone every build, and the entrypoint re-verifies the stack on
   every start.
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
spark-comfyui.sh          # The entry point: ~1750 lines. Host-side
                          # lifecycle (docker orchestration, mounts,
                          # backup/restore, status/watch, tune).
container/
  Dockerfile              # 4 named stages: base / torch / sage / final.
                          # Everything reproducible bakes in here.
  entrypoint.sh           # runtime half of the mod system (see below)
  build-mods.sh           # build-time half (mods 05 + 10)
  build-patches.sh        # merges comfyui-patches.list in-build
mods/                     # mod contract + shared helper library
  _lib/mod_common.sh      # py_patch_file, torch_cuda_diag, sage_kernel_ok,
                          # onnx_gpu_ok, kitchen_nvfp4_ok, repair_torch...
                          # sourced by the host script AND inside the image
  05-setuptools-compat/   # build-time: setuptools within torch's pin
  10-unified-memory-free/ # build-time: get_free_memory() unified pool
  20-torch-repair/        # entrypoint: torch guard (mod_prerun + diag)
  30-manager-config/      # entrypoint: Manager config.ini on user mount
  40-sageattention/       # vestigial run.sh; logic in Dockerfile + gates
  50-onnxruntime-gpu/     # vestigial run.sh; logic in Dockerfile + gates
  README.md
data/                     # ALL user content (gitignored), bind-mounted
spark-mounts.conf         # mount overrides (gitignored, seeded template)
comfyui-patches.list      # patch list (gitignored, seeded template)
README.md LICENSE .gitignore .dockerignore CLAUDE.md
```

The docker image (`spark-comfyui:latest`, ~22 GB) and the cache volume
(`spark-comfyui-cache`: pip downloads + compiled sm_121 kernels) live in
docker's storage, not the repo dir. `backup` writes to `backups/`
(gitignored).

## Commands (dispatch at the bottom of the script)

`install` · `run [args]` · `service [--disable]` · `stop` ·
`update [--torch|--rollback]` · `doctor` · `status [--watch [SEC]]` ·
`tune [--clock-cap MHZ] [--persist]` · `backup [--with-output] [FILE]` ·
`restore FILE` · `reset [--yes]` · `shell` · `--version`. Hidden
aliases: the `container <verb>` spellings from the dev phase, `verify`
(doctor), `monitor` (status --watch), `rollback` (update --rollback).

Mental model, also printed in `--help`: install once (image build), then
run or service; update now and then (cached rebuild + `:previous`
rollback point); something feels wrong, run doctor. `update` self-updates
the tool first (`self_update`: ff-only pull, re-exec once;
`SELF_UPDATE_RESUME` makes the re-exec land back in the container
update). `install`, `update` and `restore` refuse to run over a
native-era layout (`check_legacy_layout`: a ComfyUI checkout where data/
should be) and point at the v2026.07.20 tag, which is where the migrate
tooling lives, or the v2026.07.19 tag to stay native. The legacy mount
fallback, the migrate command and the legacy branch were removed in
2026.07.20.1; without the gate, install would create an empty data/ and
silently shadow a native user's content.

## The mod system (most important architecture)

A "mod" is a self-contained, idempotent unit under `mods/NN-name/`. Since
the cut, mods run in two places, both INSIDE the image lifecycle; there is
no host-side mod pass anymore:

- **Image build** (`container/build-mods.sh`): runs `05-setuptools-compat`
  and `10-unified-memory-free` — everything that needs neither a GPU nor
  user content. A failed apply or verify fails the build.
- **Entrypoint** (`container/entrypoint.sh`, every start): custom-node
  requirements install (before the torch guard, because node installs are
  what clobber torch), `20-torch-repair`'s `mod_prerun` with
  `torch_cuda_diag`, the live SageAttention kernel gate (`sage_kernel_ok`
  directly — golden rule 3 lives here since builds have no GPU; failure
  refuses to launch), then `30-manager-config` onto the mounted user dir.

The authoring contract is unchanged: each `run.sh` is a sourced fragment
(`# shellcheck shell=bash`) defining `mod_describe` / `mod_apply` /
`mod_verify` (+ optional `mod_prerun`), with `mods/_lib/mod_common.sh`
preloaded and `INSTALL_DIR`, `VENV_DIR`, `MOD_DIR` set. Status protocol:
`mod_apply`'s first output token is `applied`/`present`/`skipped`.
`py_patch_file <rel> <tag> <transform.py>` still does marker idempotency,
a `.spark-orig` backup refreshed per apply, and the `ast.parse` guard
that reverts a patch producing invalid Python; transforms echo input
unchanged when their anchor is missing (`skipped:anchor-not-found` is how
"upstream moved the code" surfaces — the build fails and names it).

Vestigial but retained: `MOD_CRITICAL`/`MOD_STREAM`/`mod_export` were
consumed by the deleted native runner; the declarations remain in the
run.sh files as documentation and for a possible future runner, and
`mods/40-sageattention`/`50-onnxruntime-gpu` keep their run.sh even
though the image build installs Sage/onnx in Dockerfile steps — their
underlying functions (`sage_kernel_ok`, `onnx_gpu_ok`, `repair_torch`,
`torch_cuda_diag`, `kitchen_nvfp4_ok`) live in `mod_common.sh` and are
what the entrypoint and doctor gates call.

Patched files never sit modified in a host tree anymore: the patch is
baked into the image, and every image rebuild starts from a fresh clone.
The whole sync/revert dance the native updater needed is gone.

Adding a build-time source-patch mod: drop `mods/NN-name/` AND add it to
the list in `container/build-mods.sh`. Adding entrypoint behavior: edit
`container/entrypoint.sh`.

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

One shared definition of "user content": the readonly `USER_CONTENT` array
(`models user input output custom_nodes extra_model_paths.yaml`), resolved
to host paths by `resolve_mounts` (a spark-mounts.conf key wins, else
`DATA_DIR/<entry>`), read per entry with `_mount_path`. Per-entry
overrides are fully supported since 2026.07.20.2: archives always carry
entry names (user/, input/, output/ enter the tar through stage-dir
symlinks with -h, which also dereferences symlinks inside them), and
restore merges each entry into its resolved path.

**reset (`cmd_container_reset`)**: content is outside the image by design,
so reset only removes what is reproducible: the container, every image
tag, the cache volume, then rebuilds with `--no-cache`. `data/` is never
touched.

**backup (`cmd_backup`)**: small tgz of meta, manifests, plain-node
copies, config files; `user/`, `input/`, `output/` tarred from the live
tree with `--ignore-failed-read` (exit 1 tolerated: safe while serving).
The ComfyUI commit in `meta` comes from the image label
`org.spark-comfyui.comfy-sha`. Archive format unchanged (`format=1`), so
native-era backups restore fine.

**restore (`cmd_restore`)**: unpack + `format=1` check; the legacy gate
(see Commands) fires before anything else; build the image if missing;
`container stop`; merge `user/`/`input/`/`output/` into data/; restore
config files (live copies saved aside as `.bak`); custom nodes (plain
copies, then manifest clones with detached checkout, prompt-proofed);
models manifest diffed against disk with sizes. NO pip step and no mod
pass: the entrypoint installs every node's requirements and verifies
torch on each start, so a restore is content-only by construction.

**migrate**: removed in 2026.07.20.1. The tooling (content moves by
git-trackedness-aware rename, stock-skeleton handling, systemd
retirement) lives permanently in the v2026.07.20 tag, which is exactly
where `check_legacy_layout` sends native-era users.

**doctor**: an info line names the newest `backups/spark-backup-*.tgz`
and its age, or that none exists. Only knows the default `backups/` dir.

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
- **NVRTC**: current cu130 wheels bundle it (pip nvidia-cuda-nvrtc,
  verified in-image 2026-07-20; earlier aarch64 wheels bundled none). The
  slim runtime image relies on the bundled one and carries no system CUDA
  beyond a copied ptxas (Triton's JIT needs a CUDA-13 ptxas, triton#10331).
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

## Containerization (THE architecture since the cut)

The architecture below was designed and executed as phases 1-3d on
container-dev during 2026-07-19/20 (the planning document,
container/ROADMAP.md, was deleted after execution; the phase log lives in
the git history and the release history below). The last native release
is reachable as the v2026.07.19 tag.

Split: the image holds everything reproducible (ComfyUI at a pinned commit,
venv with cu130 torch, native sm_121 SageAttention at SAGE_REF, the
sha256-pinned onnxruntime wheel, build-time mods 05+10 applied via
container/build-mods.sh reusing the mods/ contract). The USER_CONTENT set is
bind-mounted per the resolution contract (a spark-mounts.conf key wins,
else data/; `resolve_mounts` implements it, `status` prints it). A named
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

The build resolves upstream master to a SHA and passes it as COMFY_SHA
(the docker cache key: a ComfyUI bump rebuilds only from the clone layer;
torch/Sage stages stay cached; `--torch` busts the named torch stage via
--no-cache-filter). update keeps the replaced image as :previous;
--rollback swaps :latest and :previous (toggles; a tag swap, atomic).
status is quiet-when-healthy; its one warning is a running container
whose image is no longer :latest. doctor runs the four live gates
(torch+diag, sage kernel, onnx provider, kitchen NVFP4) inside a
throwaway --gpus container from the exact image run uses.

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

Phase status: 1 through 3d are COMPLETE and field-verified on the GB10
(2026-07-19/20); the git log carries the per-phase detail. The cutover gate (3c) passed decisively: same
Krea-2 Turbo workflow and seeds, fresh launch per condition, native
first=29.8s steady=13.59s vs container first=28.9s steady=13.61s (0.15%
steady delta against a 3% threshold), and all four seed-matched output
pairs BIT-IDENTICAL (mean abs pixel diff 0, max 0). Bench method: the
production workflow extracted from an output PNG's embedded API prompt,
POSTed to /prompt with a fixed seed sequence, watch running. The cut (3d)
removed the native path (972 lines), gated the upgrade cliff
(check_legacy_layout in install/update), created the local `legacy`
branch at v2026.07.19, and rewrote README and this file. RELEASED as
v2026.07.20 (2026-07-20), consolidated in v2026.07.20.1. Phase 4 progress
(2026-07-20): image slimming DONE — a final `runtime` stage on plain
ubuntu:24.04 (venv + ComfyUI + spark scripts + a copied ptxas; NVRTC
comes from the pip wheels; build tools kept for sdist node deps; the
SageAttention build tree not copied) halved the image, 22.4 to 11.3 GB,
all four GPU gates and a live launch green. Per-entry mount-override
support in backup/restore DONE (_mount_path replaced content_root).
Remaining phase 4, all needing a decision or host changes: GHCR prebuilt
images (CI + supply chain), rootless docker, read-only rootfs.

## Env var overrides

- `DATA_DIR` (content root, default `data/` next to the script),
  `MOUNTS_CONF` (default `spark-mounts.conf`), `CONTAINER_IMAGE`,
  `CONTAINER_NAME` (both default `spark-comfyui`)
- `REPO_URL`, `PORT`, `TORCH_INDEX`, `ORT_WHEEL_URL`, `PATCH_LIST`,
  `MODS_DIR` (all become build args or run-time wiring)
- `SAGE_REF`: pinned SageAttention commit, see GB10 domain knowledge
- `SPARK_BF16` (default 1), `SPARK_STATIC_VRAM` (default 0, ComfyUI issue
  #13920 caveat): passed into the container by `run`/`service`
- `SPARK_SELF_UPDATE` (default 1). 0 stops `update` from git-pulling the
  spark-comfyui repo itself.
- Gone since the cut: `SPARK_SOURCE_PATCHES` (the host-side mod pass it
  toggled no longer exists; mods apply at image build and in the
  entrypoint), `PIP_RETRIES`/`PIP_DEFAULT_TIMEOUT` as user knobs (pip runs
  inside the build/container). Gone since 2026.07.20.1: `INSTALL_DIR`,
  `VENV_DIR`, `SAGE_SRC` (legacy-tree locations; the migrate tooling that
  used them lives in the v2026.07.20 tag).

## Patch list (separate from mods)

`comfyui-patches.list` next to the script (template seeded by `install`)
merges PRs/branches on top of the pinned upstream commit INSIDE the image
build (`container/build-patches.sh`, on a `spark-patched` branch). Format:
`pr:<N>` | `branch:<name>` | `remote:<url> <branch>`. A changed list busts
exactly the clone-down layers; a merge conflict fails the build loudly. As
of mid-2026 the big Spark PRs are merged upstream, so the default (empty)
list means plain master tracking, which is optimal.

## Development workflow that has worked

Tight empirical loop: propose a change, the user runs it on the real Spark,
the real error comes back, diagnose from the actual output, fix, re-test.
The hardware is the source of truth; do not trust assumptions about GB10
behavior without a test or a cited field report. When adding source patches,
dry-run the transform on a fixture and `ast.parse` the result first.

## Likely next tasks / open threads

- Watch for ComfyUI refactors that move `get_free_memory`. Mod 10 reports
  `skipped:anchor-not-found` if so, which since the cut FAILS THE IMAGE
  BUILD (build-mods.sh dies on a failed verify) — loud by design; the
  transform's regex/anchor then needs updating.
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
- **2026.07.20**: THE CONTAINER RELEASE. spark-comfyui is container-only:
  the whole stack (ComfyUI at a pinned commit, cu130 torch, native sm_121
  SageAttention, sha256-pinned GPU onnxruntime, the mods) bakes into a
  docker image; content lives in data/ with spark-mounts.conf overrides;
  custom-node code runs confined (non-root, cap-drop ALL,
  no-new-privileges). update rebuilds on cached layers and keeps
  :previous for atomic rollback; service uses a docker restart policy;
  reset is content-safe by construction; doctor runs the live GPU gates
  in a throwaway container. New migrate command; install/update refuse a
  legacy layout with instructions; branch legacy = v2026.07.19 for
  stay-native users. Cutover gate: native steady 13.59s vs container
  13.61s on the production Krea-2 workflow, seed-matched outputs
  bit-identical. Developed as phases 1-3d on container-dev, each
  field-verified on the GB10; details in the phase commits.
- **2026.07.20.1**: legacy cleanup. The migrate command, the legacy mount
  fallback, the legacy content-root support in backup/restore, the native
  systemd branch in stop, and the INSTALL_DIR/VENV_DIR/SAGE_SRC vars are
  removed (145 lines), and the legacy branch is deleted from GitHub (the
  last native release stays reachable as the v2026.07.19 tag).
  check_legacy_layout stays as a slim detector that refuses
  install/update/restore over a native-era layout and points at the
  v2026.07.20 tag (where migrate lives permanently) or the v2026.07.19
  tag — without it, install would create an empty data/ and silently
  shadow a native user's content. Suite rewritten (15 assertions), fresh
  blank-slate launch field-verified.
- **2026.07.20.2**: phase 4 pair. Slim runtime image: final stage on
  plain ubuntu:24.04 with the venv, ComfyUI, the spark scripts and a
  copied CUDA-13 ptxas (NVRTC now ships in the cu130 pip wheels; build
  tools kept so sdist node deps still compile; the SageAttention build
  tree stays behind). 22.4 GB to 11.3 GB, all four GPU gates plus a live
  launch verified on the slim image. And per-entry spark-mounts.conf
  overrides are now honored by backup/restore: _mount_path per entry,
  archives keyed by entry name via symlink-staged tar -h, restore merges
  into resolved paths; the single-root refusal is gone.

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
