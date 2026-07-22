# CLAUDE.md

This file orients Claude Code (or any contributor) to the project:
architecture, design decisions, GB10 domain knowledge, and the roadmap.
It is not a changelog. Release history lives in the git tags and the
commit log; results of specific benchmarks live in the commits that ran
them.

Style rule for every markdown file in this repo, including release notes:
plain and direct. No dash punctuation in prose, no rhetorical flourishes,
short sentences. Hyphens in compound words, flags, dates and list markers
are fine. Console captures stay verbatim.

## What this is

`spark-comfyui` is a single-entry-point bash tool that installs, runs,
updates, and maintains [ComfyUI](https://github.com/Comfy-Org/ComfyUI)
on the **NVIDIA DGX Spark (GB10 Grace Blackwell)**, entirely as a
hardened docker container. The Spark is unusual hardware: aarch64 Grace
CPU, an sm_121 Blackwell GPU most toolchains don't target yet, and
128 GB of unified CPU/GPU memory. A generic ComfyUI install either fails
or runs in silently degraded states. This tool makes the whole lifecycle
automatic, reproducible (the image), and confined (custom-node code
cannot touch the host).

Author/owner: GitHub `bjarkebolding`. Development home:
`~/projects/spark-comfyui` on host `sparky` (sole remote: `origin`
GitHub). Published: https://github.com/bjarkebolding/spark-comfyui.
Current version: **2026.07.22**. Only the newest tag and GitHub
Release are kept; older tags and release pages were removed on
2026-07-20 (the git history is the archive). MIT licensed,
shellcheck-clean.

## Versioning and releasing

CalVer: `YYYY.MM.DD`, plus `.N` for a second behavior-changing release
the same day. A version's only job is to stamp which behavior state a
bug report ran; `doctor`'s update probe is git-ancestry-based, so the
format is cosmetic to tooling.

**Self-update pulls main HEAD, so pushing to main IS releasing. Always
bump VERSION in the same push.** Docs-only pushes need no bump. Only the
newest GitHub Release stays published; creating a new one includes
deleting the one it supersedes.

## Golden rules (do not regress these)

1. **shellcheck-clean is non-negotiable.** Run `shellcheck -S warning
   spark-comfyui.sh mods/*/run.sh mods/_lib/mod_common.sh container/*.sh`
   before every commit. Mod `run.sh` files are sourced fragments and
   carry `# shellcheck shell=bash`.
2. **The main script must stay relocatable.** All paths derive from
   `BASE_DIR`. Never hardcode `$HOME` or absolute install paths. `mods/`
   and `container/` must sit next to the script.
3. **Every optimization is backed by a functional gate, not a
   heuristic.** Warnings guide; a live test decides. SageAttention is
   only enabled after a real multi-shape GPU kernel run passes, never on
   a version string or a `--help` grep. Two prior bugs came from
   trusting heuristics: `TORCH_CUDA_ARCH_LIST="12.0"` and grepping
   `ptxas --help` for `sm_121`.
4. **Idempotent and self-healing.** `install` and `update` are safe to
   re-run. Structurally so since the container cut: the image is
   immutable, patches bake into it from a fresh clone every build, and
   the entrypoint re-verifies the stack on every start.
5. **Fail loud on real problems, quiet when healthy.** No silent
   degradation, and no false alarms: benign platform noise (e.g. the
   aarch64 cusparselt line from `pip check`) is filtered, not surfaced.
   This rule governs the `--watch` dashboard too: rows render only when
   they carry information.
6. **Test patches against fixtures before shipping.** Source patches
   edit upstream Python; dry-run the transform on a realistic fixture
   and confirm the result still `ast.parse`s.

## Repository layout

```
spark-comfyui.sh          # The entry point. Host-side lifecycle: docker
                          # orchestration, mounts, backup/restore,
                          # status/watch, tune.
container/
  Dockerfile              # named stages: base / torch / sage / final /
                          # runtime. Everything reproducible bakes here.
  entrypoint.sh           # runtime half of the mod system
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
data/                     # ALL user content (gitignored), bind-mounted
spark-mounts.conf         # mount overrides (gitignored, seeded template)
comfyui-patches.list      # patch list (gitignored, seeded template)
```

The docker image (`spark-comfyui:latest`) and the cache volume
(`spark-comfyui-cache`: pip downloads + compiled sm_121 kernels) live in
docker's storage. `backup` writes to `backups/` (gitignored).

## Commands (dispatch at the bottom of the script)

`install` · `run [args]` · `service [--disable]` · `stop` ·
`update [--torch|--rollback]` · `doctor` · `status [--watch [SEC]]` ·
`tune [--clock-cap MHZ] [--persist]` · `backup [--with-output] [FILE]` ·
`restore FILE` · `reset [--yes]` · `shell` · `--version`. Hidden
aliases: `verify`, `monitor`, `rollback`.

Mental model, also printed in `--help`: install once (image build), then
run or service; update now and then (cached rebuild plus a `:previous`
rollback point); something feels wrong, run doctor. `update`
self-updates the tool first (`self_update`: ff-only pull, re-exec once;
`SELF_UPDATE_RESUME` makes the re-exec land back in the container
update). `install`, `update` and `restore` refuse to run over a
native-era layout (`check_legacy_layout`: a ComfyUI checkout where
data/ should be) and print the move commands (five renames into data/)
inline. Without that gate, install would create an empty data/ and
silently shadow a native user's content.

## The container architecture

Split: the image holds everything reproducible (ComfyUI at a pinned
commit, venv with cu130 torch, native sm_121 SageAttention at SAGE_REF,
the sha256-pinned onnxruntime wheel, build-time mods). The USER_CONTENT
set (`models user input output custom_nodes extra_model_paths.yaml`) is
bind-mounted per the resolution contract: a spark-mounts.conf key wins,
else `DATA_DIR/<entry>`. `resolve_mounts` implements it, `_mount_path`
reads single entries, `status` prints the resolved table. Extra
`mount = HOST:CONTAINER[:ro]` lines are validated: host must exist (a
typo must not silently shadow a NAS with an empty dir), container path
must be under /opt/ComfyUI.

Dockerfile stages: `base` (apt, uid-1000 user) → `torch` (venv + cu130
wheels) → `sage` (SageAttention compile, no GPU needed) → `final`
(ComfyUI clone at COMFY_SHA, patch list, requirements, onnx wheel,
build mods) → `runtime` (the shipped image: plain ubuntu:24.04 with the
venv, ComfyUI, spark scripts, and a copied CUDA-13 ptxas; NVRTC comes
from the pip wheels; build tools kept so sdist node deps compile; the
SageAttention build tree is not copied). COMFY_SHA is the docker cache
key: a ComfyUI bump rebuilds only from the clone layer, `--torch` busts
the named torch stage via `--no-cache-filter`.

`container/entrypoint.sh` is the runtime half of the mod system, every
start: custom-node requirements install (BEFORE the torch guard, because
node installs are what clobber torch), mod 20 prerun with
torch_cuda_diag, the live SageAttention kernel gate (builds have no GPU,
so golden rule 3 lives here; failure refuses to launch), mod 30 onto the
mounted user dir, then main.py with the tuned flag set. Hardening on
`run`: non-root, cap-drop ALL, no-new-privileges, GPU only, --rm
(stateless), 1 GB shm. `service` is the same container detached with
`--restart unless-stopped`; no systemd.

`update` keeps the replaced image as `:previous`; `--rollback` swaps
`:latest` and `:previous` (a tag swap, atomic, toggles). `doctor` runs
the four live gates (torch+diag, sage kernel, onnx provider, kitchen
NVFP4) inside a throwaway `--gpus` container from the exact image `run`
uses. The native path was removed only after a live A/B against it on
the production hardware gated the cut (golden rule 3 applied to the
architecture itself).

Field-learned docker gotchas (do not re-derive):
- The containerd image store garbage-collects a tagless image INSTANTLY.
  `update` therefore holds `:pre-update` on the old image
  through the build and promotes it to `:previous` only on a real change.
- buildx provenance attestations stamp each build's manifest with the
  build time, so identical cached builds get different image IDs unless
  built with `--provenance=false`. Without it, update's
  changed-vs-current comparison always says changed.
- Base image pinned to the CUDA 13.0 patch line: the r580 driver and
  cu130 torch wheels are 13.0-era, and a 13.1+ ptxas can emit PTX the
  driver JIT rejects. Bumping past 13.0.x is a deliberate pin change
  with a field test, same policy as SAGE_REF.
- The stock GLSL nodes (comfy_extras/nodes_glsl.py) dlopen bundled ANGLE
  libs that link libX11/libXext/libxcb even headless; those apt packages
  are in the image for that reason alone.
- The cache volume mountpoint must exist in the image owned by the
  uid-1000 user, or docker creates it root-owned and pip/uv fail.
- .dockerignore whitelists the build context (script, mods/, container/,
  patch list): BASE_DIR on a live install holds user content next to the
  Dockerfile.

## The mod system

A "mod" is a self-contained, idempotent unit under `mods/NN-name/`. Mods
run in two places, both inside the image lifecycle; there is no
host-side mod pass:

- **Image build** (`container/build-mods.sh`): mods needing neither GPU
  nor user content (05, 10). A failed apply or verify fails the build.
- **Entrypoint**: everything needing the GPU or the mounted user dir
  (20 prerun, the sage gate, 30).

Authoring contract: each `run.sh` is a sourced fragment defining
`mod_describe` / `mod_apply` / `mod_verify` (+ optional `mod_prerun`),
with `mod_common.sh` preloaded and `INSTALL_DIR`, `VENV_DIR`, `MOD_DIR`
set. `mod_apply`'s first output token is `applied`/`present`/`skipped`.
`py_patch_file <rel> <tag> <transform.py>` does marker idempotency, a
`.spark-orig` backup refreshed per apply, and an `ast.parse` guard that
reverts a patch producing invalid Python. Transforms echo input
unchanged when their anchor is missing; `skipped:anchor-not-found` is
how "upstream moved the code" surfaces, and it fails the image build
loudly.

Vestigial but retained: `MOD_CRITICAL`/`MOD_STREAM`/`mod_export` were
consumed by the deleted native runner; the declarations remain as
documentation. Mods 40/50 keep their run.sh though the Dockerfile
installs Sage/onnx directly; their underlying functions live in
`mod_common.sh` and are what the entrypoint and doctor gates call.

Adding a build-time source-patch mod: drop `mods/NN-name/` AND add it to
the list in `container/build-mods.sh`. Adding entrypoint behavior: edit
`container/entrypoint.sh`.

## status --watch

Everything lives in `cmd_status` plus helpers (`_watch_row`,
`_watch_hdr`, `_watch_comfy`, `_series_nonzero`/`_series_any`). These
decisions are field-verified; don't re-derive them.

**Two outputs per tick, different jobs.** The append-only
`thermal_monitor.log` line is the primary output: it survives the silent
hard-reboots this exists to diagnose, and it carries every field
unconditionally. The dashboard is the live tty view of the same samples;
when stdout isn't a tty, plain log lines are emitted instead. Some
fields are log-only by design (RSS, the per-process GPU split, CACHE,
LOAD, raw EVT hex): they matter in a post-mortem, not on screen. The log
rotates to `.1` at 50 MB on watch start.

**Quiet-when-healthy rendering.** Always-on rows: GPU
temp/power/sm-clk/util, unified memory, CPU, disk I/O. Conditional rows
render only with a story: throttle (any slowdown bit in the window),
swap (only if swap exists at all), generation telemetry (only with
data). Ring buffers always advance, so a row appears with its window
history intact. Visibility tests use `_series_nonzero`/`_series_any`;
they exist because `${arr[*]//pat/}` substitutes per-element and re-adds
joining spaces, so it can never yield an empty string; join to a scalar
first.

**Renderer** (`_watch_row`, one awk per row): per-glyph heat colors by
absolute thresholds (temp 70/80, power 60/80 where red power IS the GB10
overcurrent zone, unified memory 85/95 percent of the pool). Trend
arrows are dead-banded to 5 percent of the window span. Colors wrap the
padded value text so escape bytes never skew columns. Bar glyphs are
split into an awk array (not substr) for mawk/gawk parity. Section rules
use `sed`, not `tr` (tr is byte-oriented and shreds multibyte rules).
The window width adapts to the terminal.

**Generation telemetry** (`_watch_comfy`, one python call per tick)
polls three stock HTTP endpoints: `/history`, `/queue`,
`/internal/logs/raw`. HTTP polling is BY DESIGN, not a websocket:
per-step progress events go only to the prompt's owning client, so a
passive `/ws` listener would never see other clients' progress. Live
it/s is the newest tqdm rate scraped from the server's log ring buffer,
only while a prompt is in flight. Hit rate is the `execution_cached`
node count over the prompt's node count.

**Latency** (submission to saved): the loop timestamps queue ids on
first sight; when one finishes, latency = now minus first-seen. The
latency check MUST run before the seen[] populate/prune: on the very
tick a gen finishes the queue is already empty, and pruning first would
wipe the timestamp. Only covers jobs submitted while the watch runs.

**Session A/B summary** (the `session:` line): every gen that finishes
under this watch is recorded exactly once via fin_id-change accounting;
whatever fin_id says on the FIRST tick is only a baseline, so history
predating the watch never skews a comparison. First gen carries the
model load, steady excludes it. Stats reset per launch: one watch
session per condition gives one comparable line. The GENERATION header
names the attention backend, the usual A/B dimension.

**Dead/dropped gauges, don't re-add:** `utilization.memory` reads a
constant 0 on GB10 even under full load. `sw-power-cap` (0x4) is benign
and constant on GB10; only the four HW/thermal-slowdown bits
(0x08/0x20/0x40/0x80) alarm. A dedicated pstate row duplicated the
sm-clk story; it is a tag on that row now. GB10 nvidia-smi N/A fields:
`clocks.mem`, `fan.speed`, `temperature.memory`, `power.limit` (and
`nvidia-smi -pl` does not work, hence `tune --clock-cap`).

## reset / backup / restore

One shared definition of "user content": the readonly `USER_CONTENT`
array, resolved per entry by `resolve_mounts`/`_mount_path`. Per-entry
overrides are fully supported: archives always carry entry names
(user/, input/, output/ enter the tar through stage-dir symlinks with
`-h`, which also dereferences symlinks inside them), and restore merges
each entry into its resolved path.

**reset**: content is outside the image by design, so reset removes only
what is reproducible (container, every image tag, the cache volume) and
rebuilds with `--no-cache`. `data/` is never touched.

**backup**: small tgz of meta, manifests, plain-node copies, config
files; content dirs tarred from the live tree with
`--ignore-failed-read` (exit 1 tolerated: safe while serving). Models
are manifested with sizes, never archived. The ComfyUI commit in `meta`
comes from the image label `org.spark-comfyui.comfy-sha`.

**restore**: format check, legacy gate, build the image if missing,
stop, merge content into resolved paths, config files saved aside as
`.bak` when they differ, custom nodes re-cloned at pinned commits
(prompt-proofed, path-traversal-checked names), models manifest diffed
against disk. NO pip step and no mod pass: the entrypoint installs
every node's requirements and verifies torch on each start, so a
restore is content-only by construction.

## GB10 domain knowledge (do not relitigate)

- **PyTorch**: cu130 aarch64 wheels. Install BEFORE ComfyUI requirements
  so nothing pulls CPU-only torch. The `sm_121 exceeds torch's max`
  startup warning is expected and harmless (PTX JITs).
- **SageAttention**: build from source with
  `TORCH_CUDA_ARCH_LIST="12.1+PTX"` (native sm_121 cubin + PTX
  fallback). `"12.0"` alone produced sm_120 cubins with no PTX and
  failed with `no kernel image`. Needs CUDA 13's ptxas (check by
  version, NOT by grepping --help). Mandatory; gated on a live
  multi-shape kernel test. The kernel test checks shape/finiteness, not
  visual correctness, so it cannot catch an output-quality regression;
  the **SAGE_REF pin** is the actual gate. Bumping it is a deliberate
  decision with a field test, never automatic (SageAttention 3.x showed
  visual artifacts on GB10; stay on the pinned 2.2.x line).
- **Flash Attention**: FA3 cannot target sm_121 at all. FA2 can be
  compiled but loses to SDPA on Blackwell. Not installed; only worth
  revisiting if a custom node hard-imports `flash_attn`.
- **onnxruntime**: no PyPI aarch64+cu13 GPU wheel; a community sm_121
  wheel is installed (URL in `ORT_WHEEL_URL`, sha256-pinned; pip
  verifies the fragment). Detect via `get_available_providers()`, not
  startup logs. A later PyPI `onnxruntime` install shadows it silently;
  `doctor` re-detects.
- **Unified memory**: swap ON plus heavy load means a silent whole-box
  freeze (no OOM kill). `tune` disables swap. `--gpu-only`/`--highvram`
  HURT here. bf16 flags are the native fast path (`SPARK_BF16=0` to
  disable).
- **`--high-ram`: evaluated on GB10, REJECTED.** All of its code paths
  gate on pinned memory, which our `--disable-pinned-memory` no-ops, so
  on the production config it reduces to forcing `--cache-classic` with
  no measurable effect. In its real mode (pinning enabled) it was
  slower here and pinned RAM uncapped (which would also starve a
  co-resident LLM). Its designed benefit (residency vs pagefile) is
  moot because `tune` disables swap. `--disable-pinned-memory` stays:
  no cost, inoculates against the uncapped-pin path.
- **get_free_memory cliff** (mod 10): `cudaMemGetInfo` under-reports
  free memory when another CUDA process is resident, causing needless
  offload and drastically slower sampling. Fixed by reading
  `psutil.virtual_memory().available`. Matters in practice with a
  co-resident vLLM.
- **Overcurrent reboots**: some units hard-reboot (no logs) on the power
  spike at denoise start. `tune --clock-cap 2100` caps clocks
  (nvidia-smi -pl is N/A on GB10). `status --watch` captures the
  pre-crash telemetry.
- **NVRTC**: current cu130 wheels bundle it (pip nvidia-cuda-nvrtc).
  The runtime image relies on the bundled one and carries no system
  CUDA beyond a copied ptxas (Triton's JIT needs a CUDA-13 ptxas,
  triton#10331).
- **Dependency pins**: torch pins setuptools; never blindly upgrade it.
  `ensure_setuptools_compat` (mod 05) reads torch's own constraint.
  `doctor` runs `pip check` (filtering the benign cusparselt line).
- **ComfyUI-Manager's pip_auto_fix.list is incompatible with CUDA torch
  builds**: `prestartup_script.py` parses pins with a naive
  per-segment `int()` parser that crashes on any PEP 440 local version
  (e.g. `+cu130`), including the installed version during drift checks,
  so no pin format avoids it. Don't relitigate reformatting the pin.
  Mod 30 never writes this file (and deletes a stale one);
  `downgrade_blacklist` plus mod 20's real `torch.cuda.is_available()`
  checks are the actual protection.
- **NVFP4**: `comfy-kitchen` is a stock ComfyUI dependency; its fast
  native `cuda` backend needs torch compiled with CUDA >= 13, which the
  cu130 install guarantees, so NVFP4 rides for free. Enforced by a real
  forced-kernel gate in `doctor` (`kitchen_nvfp4_ok`: forces the cuda
  backend, which verifiably raises instead of falling back). The
  `[cublas]` extra is NOT needed on a stock install (live-tested). One
  lever deliberately untouched: `--enable-triton-backend` (a third
  NVFP4 path, unevaluated on GB10; don't adopt without a field test).

## Env var overrides

- `DATA_DIR` (content root, default `data/`), `MOUNTS_CONF` (default
  `spark-mounts.conf`), `CONTAINER_IMAGE`, `CONTAINER_NAME` (both
  default `spark-comfyui`)
- `REPO_URL`, `PORT`, `TORCH_INDEX`, `ORT_WHEEL_URL`, `PATCH_LIST`,
  `MODS_DIR` (become build args or run-time wiring)
- `SAGE_REF`: pinned SageAttention commit, see GB10 domain knowledge
- `SPARK_BF16` (default 1), `SPARK_STATIC_VRAM` (default 0; see the
  ComfyUI issue #13920 caveat in the roadmap): passed into the container
  by `run`/`service`
- `SPARK_SELF_UPDATE` (default 1). 0 stops `update` from git-pulling
  the spark-comfyui repo itself.

## Patch list (separate from mods)

`comfyui-patches.list` next to the script (template seeded by `install`)
merges PRs/branches on top of the pinned upstream commit INSIDE the
image build (`container/build-patches.sh`, on a `spark-patched` branch).
Format: `pr:<N>` | `branch:<name>` | `remote:<url> <branch>`. A changed
list busts exactly the clone-down layers; a merge conflict fails the
build loudly. The default (empty) list means plain master tracking.

## Development workflow that has worked

Tight empirical loop: propose a change, run it on the real Spark, take
the real error, diagnose from actual output, fix, re-test. The hardware
is the source of truth; do not trust assumptions about GB10 behavior
without a test or a cited field report. When adding source patches,
dry-run the transform on a fixture and `ast.parse` the result first.

## Roadmap / open threads

- Phase 4 remainder, each needing a decision or host changes: GHCR
  prebuilt images (arm64 CI, supply-chain stance), rootless docker
  (host-level change affecting other containers), read-only rootfs
  (conflicts with the entrypoint's pip installs by design; would need a
  writable venv volume and trades away image-immutable starts).
- Watch for ComfyUI refactors that move `get_free_memory`. Mod 10
  reports `skipped:anchor-not-found` and the image build fails loudly;
  the transform's anchor then needs updating. Related: `comfy-aimdo`
  (multi-threaded model loader, upstream PR #13802) touches the
  memory-loading path; on updates confirm mod 10 still applies.
- Retired mod knowledge: `20-no-double-vram` was removed because
  upstream moved the patched line out of `comfy/utils.py` into
  `cast_to`/`cast_to_device` in `comfy/model_management.py` and now
  avoids the transient double-allocation itself. If a future profile
  shows it again, target those functions, not the old anchor. The `20`
  prefix was later reused by the unrelated `20-torch-repair`.
- The community ONNX wheel URL will age. A URL refresh must update the
  `#sha256=` fragment in the same edit, sourced from the HF API's
  lfs.oid and confirmed against a real download.
- SageAttention watch list: PR #372 (unmerged causal-mask fix, the one
  real 2.2.x-line fix pending) is the trigger to consider a new
  SAGE_REF. The 3.x artifact reports on GB10-class hardware remain open
  in upstream's tracker; stay pinned.
- ComfyUI issue #13920 (open): GB10 hang on second sampling pass with
  `--disable-dynamic-vram` + SageAttention. Documented as the
  SPARK_STATIC_VRAM caveat; if fixed upstream, remove the caveat.
- Consider a GitHub issue template asking for `doctor` output; for this
  project that one command is nearly a complete bug report.

## Release checklist (repeat per release)

1. `chmod +x spark-comfyui.sh` stays committed (mode bit `100755`).
2. shellcheck everything (golden rule 1).
3. Set `VERSION=` to today's date (append `.N` if a behavior-changing
   release already went out today). Update the header comment and the
   "Current version" line above. Tag `v<VERSION>` to match `--version`.
4. Push main + tag. `gh release create v<VERSION>` with notes in the
   repo markdown style: one-line theme, `##` sections, the one-command
   upgrade at the end. Then delete the release it supersedes; only the
   newest stays published.
