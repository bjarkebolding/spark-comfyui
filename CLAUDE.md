# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with
code in this repository.

This file orients Claude Code (or any contributor) to the project so work can
continue without re-deriving the history. Read it fully before editing.

## What this is

`spark-comfyui` is a single-entry-point bash tool that installs, runs, updates,
and maintains [ComfyUI](https://github.com/Comfy-Org/ComfyUI) on the **NVIDIA
DGX Spark (GB10 Grace Blackwell)**. The Spark is unusual hardware — aarch64
Grace CPU, an sm_121 Blackwell GPU most toolchains don't target yet, and 128 GB
of *unified* CPU/GPU memory — so a generic ComfyUI install either fails or runs
in silently degraded states. This tool's job is to make all of that boring and
self-healing.

Author/owner: GitHub `bjarkebolding`. Target repo name: `spark-comfyui`.
Hardware in use: DGX Spark, hostname `sparky`, install root `~/comfyui-spark/`.
Current version: **1.2.0** (MIT licensed, shellcheck-clean).
Published: https://github.com/bjarkebolding/spark-comfyui (v1.0.0 2026-07-10,
v1.1.0 same day — backup-revert bug fix, runtime-fallback + stuck-clock
doctor checks, TRITON_PTXAS_PATH, mod-state allowlist; v1.2.0 2026-07-11 —
NVFP4 live doctor gate, self-updating `update`).

## Golden rules (do not regress these)

1. **shellcheck-clean is non-negotiable.** Run `shellcheck -S warning
   spark-comfyui.sh mods/*/run.sh mods/_lib/mod_common.sh` before every commit.
   Mod `run.sh` files are *sourced fragments* and carry `# shellcheck shell=bash`.
2. **The main script must stay relocatable.** All paths derive from
   `BASE_DIR="$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")"`. Never hardcode
   `$HOME` or absolute install paths (the only legit `$HOME` uses are the
   systemd *user* unit dir and `loginctl`). `mods/` must sit next to the script.
3. **Every optimization is backed by a functional gate, not a heuristic.**
   Warnings guide; a live test decides. E.g. SageAttention is only enabled after
   a real multi-shape GPU kernel run passes — never on the strength of a version
   string or `--help` grep. (Two prior bugs came from trusting heuristics:
   `TORCH_CUDA_ARCH_LIST="12.0"` and grepping `ptxas --help` for `sm_121`.)
4. **Idempotent + self-healing.** `install` and `update` are safe to re-run;
   they skip or refresh, never duplicate or break. Anything that patches the
   ComfyUI tree must re-apply cleanly after `git pull`.
5. **Fail loud on real problems, quiet when healthy.** No silent degradation.
   But don't cry wolf: benign platform noise (e.g. the aarch64
   `nvidia-*-cu13 is not supported on this platform` line from `pip check`) is
   filtered, not surfaced as a failure.
6. **Test patches against fixtures before shipping.** Source patches edit
   upstream Python; always dry-run the transform on a realistic fixture and
   confirm the result still `ast.parse`s.

## Repository layout

```
spark-comfyui.sh          # THE entry point — vanilla install/run/update
                          # lifecycle + dispatch; ~1100 lines. GB10 venv-
                          # package logic (torch, SageAttention, onnx) lives
                          # in mods/_lib/mod_common.sh, not here.
mods/                     # discovered, applied & self-healed automatically
  _lib/mod_common.sh      # shared helpers: py_patch_file, py_marker_present,
                          # mod_export, AND the GB10 venv-package functions
                          # (need_nvcc, sage_kernel_ok, onnx_gpu_ok,
                          # ensure_onnx_gpu, ensure_setuptools_compat,
                          # repair_torch, build_and_verify_sage) — sourced by
                          # the main script itself at startup, not just mods.
  05-setuptools-compat/   # setuptools pinned within torch's own constraint
    run.sh
  10-unified-memory-free/ # get_free_memory() -> host-available unified pool
    run.sh transform.py
  20-torch-repair/        # torch CUDA verified/repaired; also mod_prerun
    run.sh                # (runs before every `run`, not just install/update)
  30-manager-config/      # ComfyUI-Manager config.ini (network_mode, uv,
                          # downgrade_blacklist) — NOT pip_auto_fix.list, see
                          # configure.py's docstring (crashed Manager's own
                          # version parser on every launch, retired 2026-07)
    run.sh configure.py
  40-sageattention/       # native sm_121 build + live kernel verification
    run.sh
  50-onnxruntime-gpu/     # community sm_121 GPU wheel for preprocessors
    run.sh
  README.md               # how to author a mod (incl. MOD_CRITICAL/MOD_STREAM)
README.md LICENSE .gitignore CLAUDE.md
```

The script installs ComfyUI, the venv, and SageAttention *next to itself*
(`ComfyUI/`, `comfyui-env/`, `SageAttention/`), all gitignored.

## Commands (dispatch at the bottom of the script)

`install` · `run [args]` · `stop` · `update [--torch|--rollback]` · `doctor` ·
`status [--watch]` · `tune [--clock-cap MHZ] [--persist]` · `service` ·
`--version`. Hidden back-compat aliases: `verify`→doctor, `monitor`→status
--watch, `rollback`→update --rollback, `bench`→removed (prints A/B guidance).

Mental model, also printed in `--help`: *install once → run → update now and
then; something feels wrong → doctor.* `run` also runs a cheap pre-launch mod
pass (`apply_prerun_mods` — currently just `20-torch-repair`'s guard against a
custom node having swapped in CPU torch since the last launch). `update`
self-updates the tool itself first (`self_update`: ff-only git pull of
BASE_DIR when upstream is strictly ahead, then re-execs the fresh script
exactly once — bash must never keep executing a file that changed under it).

## The mod system (most important architecture)

A "mod" is a self-contained, idempotent, self-healing unit that
`spark-comfyui.sh` applies during install/update and verifies in `doctor`.
Two flavors share one contract: **source-patch mods** that edit ComfyUI's own
Python (`10`) or config tree (`30`), and **venv-package mods** that
install/verify/repair virtualenv state (`05`, `20`, `40`, `50`) — the four
functions that used to live directly in the main script
(`ensure_setuptools_compat`, `repair_torch`, `build_and_verify_sage`,
`ensure_onnx_gpu`) now live in `mods/_lib/mod_common.sh`, wrapped by thin
`mods/NN-name/run.sh` files. Every other direct call site of those functions
(`cmd_run`'s SageAttention shadow-check, `cmd_rollback`, `cmd_doctor`'s
diagnostics) still calls them directly — only install/update/pre-run
*orchestration* goes through the mod pass now.

Each `mods/<name>/run.sh` is **sourced** (not executed) in a subshell with
`_lib/mod_common.sh` preloaded and `INSTALL_DIR`, `VENV_DIR`, `MOD_DIR`
exported. It must define three shell functions:

- `mod_describe` → echoes a one-line description
- `mod_apply`    → applies idempotently; echoes a status; returns 0
- `mod_verify`   → exit 0 if currently active, 1 if not

and may optionally define a fourth:

- `mod_prerun`   → runs before every `run`, not just install/update. Absence
  is a silent no-op; only `20-torch-repair` defines it.

**Status protocol** (parsed by `_invoke_mod`/`apply_source_patches` in the
main script): the first token of `mod_apply`'s output is the class —
`applied`, `present`, or `skipped` — and any remainder is human detail shown
to the user. Example: `applied config updated (network_mode, use_uv)`.

**Extended contract for venv-package mods** — three opt-in top-level
declarations a `run.sh` can set, all unused by the two source-patch mods:

- `MOD_CRITICAL=1` — a nonzero exit from `mod_apply`/`mod_prerun` aborts the
  whole script (`die`) instead of being swallowed into `skipped:error`. Set
  by `20`, `40` (torch/Sage failure = the install is genuinely broken); NOT
  set by `05`, `10`, `30`, `50` (their failure just means one optimization
  is inactive, generation still works).
- `MOD_STREAM=1` — output streams live to the terminal instead of being
  buffered until `mod_apply` returns (buffering would hide all progress
  during SageAttention's 10-30 min build or onnxruntime's ~220 MB wheel
  download). Set by `20`, `40`, `50`. A streamed mod reports status via
  `mod_export STATUS=<word>` instead of an echoed line, since stdout is no
  longer captured.
- `mod_export KEY=value` — appends to `$MOD_STATE_FILE`; the runner reads it
  back after the mod returns and sets `KEY` as a global in the caller's
  scope. This is how `cmd_update`'s summary gets `SAGE_ACTION` (from `40`)
  and `ORT_STATE` (from `50`) now, instead of the mod's function setting
  them as plain globals directly.

`cmd_update` also exports `SPARK_TORCH_UPGRADED` (the `--torch` flag) before
the mods pass runs, so `40-sageattention`'s `mod_apply` can use it as an
extra forced-rebuild trigger alongside its own marker/git-rev-drift check.

For Python-source patches use the helpers in `mod_common.sh`:
`py_patch_file <rel> <tag> <transform.py>` handles the marker/idempotency
check, a `<file>.spark-orig` backup (refreshed on every apply so the revert
below never restores stale upstream code), and — critically — an
`ast.parse` guard that **reverts the file if the patch would produce invalid
Python**. The `transform.py` reads source on stdin, writes patched source to
stdout, and MUST echo input unchanged when its anchor isn't found (that's how
"upstream moved the code" is detected → reported as `skipped:anchor-not-found`).
The marker string arrives via `$MARKER`.

Mods run in **filename order** (numeric prefix — this is what gives `05`
setuptools before `20` torch before `40` Sage). `_`-prefixed dirs are skipped.
Disable all mods, including the pre-run guard, with `SPARK_SOURCE_PATCHES=0`.

Adding a source-patch mod = drop a new `mods/NN-name/` dir; no main-script
edits needed. A venv-package mod needs its underlying logic added to
`mod_common.sh` too if it doesn't already live there.

## GB10 domain knowledge (hard-won; don't relitigate)

- **PyTorch**: cu130 aarch64 wheels. Install BEFORE ComfyUI requirements so
  nothing pulls CPU-only torch. The `sm_121 exceeds torch's max` startup
  warning is expected and harmless (PTX JITs).
- **SageAttention**: build from source with
  `TORCH_CUDA_ARCH_LIST="12.1+PTX"` (native sm_121 cubin + PTX fallback).
  `"12.0"` alone produced sm_120 cubins with no PTX → `no kernel image`.
  Needs CUDA 13's `ptxas` (check by *version* ≥13, NOT by grepping --help).
  Mandatory; gated on a live multi-shape kernel test. ~10-30 min build.
  `build_and_verify_sage` (mod `40-sageattention`, `MOD_CRITICAL=1
  MOD_STREAM=1` — build output streams live, failure aborts the script).
  **Pinned via `SAGE_REF`** (not tracking upstream's default branch) — the
  live kernel test checks shape/finiteness, not visual correctness, so it
  can't catch a regression like 3.x's mosaic artifacts on GB10; the pin is
  the actual gate here, per golden rule 3. Bumping it is a deliberate
  decision (edit the default in spark-comfyui.sh, or override `SAGE_REF`),
  not something that happens on its own. Currently pinned to a commit 38
  past the `v2.2.0` tag, field-verified on GB10 sm_121.
- **Flash Attention**: FA3 can't target sm_121 at all. FA2 2.8.3 *can* be
  compiled (`TORCH_CUDA_ARCH_LIST="12.0"`, ~2h) but loses to SDPA on Blackwell.
  Not installed; only worth it if a custom node hard-imports `flash_attn`.
- **onnxruntime**: no PyPI aarch64+cu13 GPU wheel; a community sm_121 wheel is
  installed (URL in `ORT_WHEEL_URL`). Detect via `get_available_providers()`,
  not startup logs. A later PyPI `onnxruntime` install shadows it (same import
  path, no pip conflict) → `update`/`doctor` re-detect and repair.
- **Unified memory**: swap ON + heavy load = silent whole-box freeze (no OOM
  kill). `tune` disables swap. `--gpu-only`/`--highvram` HURT here. bf16 flags
  are the native fast path (`SPARK_BF16=0` to disable).
- **get_free_memory cliff** (mod 10): `cudaMemGetInfo` under-reports free mem
  when another CUDA process (vLLM) is resident → needless offload → 5-15×
  slower sampling. Fixed by reading `psutil.virtual_memory().available`.
  The owner DOES run co-resident vLLM, so this matters in practice.
- **Overcurrent reboots**: some units hard-reboot (no logs) on the ~85W power
  spike at denoise start. `tune --clock-cap 2100` caps clocks (nvidia-smi -pl
  is N/A on GB10). `status --watch` captures the pre-crash telemetry.
- **NVRTC**: aarch64 cu130 wheels bundle none → torch uses system CUDA 13 lib,
  which is the GOOD state. `doctor` distinguishes the four cases.
- **Dependency pins**: torch pins `setuptools<82`; never blindly upgrade
  setuptools. `ensure_setuptools_compat` (wrapped by mod `05-setuptools-compat`)
  reads torch's own constraint. `doctor` runs `pip check` (filtering the
  benign cusparselt platform line).
- **ComfyUI-Manager's pip_auto_fix.list is incompatible with CUDA torch
  builds** (found 2026-07): `prestartup_script.py` calls `fix_broken()`
  unconditionally on every launch, which parses pinned versions with its own
  `StrictVersion` — a naive per-`.`-segment `int()` parser that crashes on
  any PEP 440 local version segment (e.g. torch's own `2.13.0+cu130`). It
  crashes the same way on the *installed* version during drift comparison,
  so no pin format on our side avoids it — don't relitigate reformatting the
  pin. `mods/30-manager-config` no longer writes this file (and deletes a
  stale one from before this fix); `downgrade_blacklist` (independent
  mechanism, unaffected) plus `20-torch-repair`'s real `torch.cuda.is_available()`
  checks are the actual protection.

## Env var overrides

`INSTALL_DIR`, `VENV_DIR`, `SAGE_SRC`, `SAGE_REF` (pinned SageAttention commit
— see GB10 domain knowledge), `REPO_URL`, `PORT`, `TORCH_INDEX`,
`ORT_WHEEL_URL`, `PATCH_LIST`, `MODS_DIR`, `SPARK_BF16` (1),
`SPARK_STATIC_VRAM` (0) — caveat: with SageAttention this can hang a second
sampling pass on GB10, open ComfyUI issue #13920, `SPARK_SOURCE_PATCHES` (1)
— disables *all six mods*, not just source patches, `SPARK_SELF_UPDATE` (1)
— set 0 to stop `update` from git-pulling the spark-comfyui repo itself,
`PIP_RETRIES`, `PIP_DEFAULT_TIMEOUT`.

## Patch list (separate from mods)

`comfyui-patches.list` next to the script lets users merge upstream PRs/branches
on top of master onto a rebuilt-fresh `spark-patched` branch each update.
Format: `pr:<N>` | `branch:<name>` | `remote:<url> <branch>`. A commented
template is seeded on first install. As of mid-2026 the big Spark PRs are merged
upstream, so the default (empty) list = plain master tracking, which is optimal.

## Development workflow that has worked

Tight empirical loop: propose change → user runs it on the real Spark → real
error comes back → diagnose from the actual output → fix → user re-tests. The
hardware is the source of truth; do not trust assumptions about GB10 behavior
without a test or a cited field report. When adding source patches, dry-run the
transform on a fixture and `ast.parse` the result first.

## Likely next tasks / open threads

- Watch for ComfyUI refactors that move `get_free_memory` — mod 10 will report
  `skipped:anchor-not-found` in `doctor` if so, and the transform's
  regex/anchor needs updating.
- Mod `20-no-double-vram` was retired (2026-07): upstream moved the
  `weight = weight.to(device=device_to)` line it patched out of
  `comfy/utils.py` entirely, into `comfy/model_management.py`'s `cast_to()` /
  `cast_to_device()`. Tracing the new load path (`comfy/ops.py:362`) shows the
  common load-time cast already calls `cast_to(..., copy=weight_has_function,
  ...)`, which is falsy for a plain weight with no LoRA/hook — i.e. upstream
  now avoids the transient double-allocation itself. If a future profile shows
  it's still happening, the fix would target `cast_to`/`cast_to_device` in
  `model_management.py`, not the old `utils.py` anchor. The `20` prefix was
  later reused for the unrelated `20-torch-repair` mod (2026-07, same
  refactor that moved venv-package steps into mods/) — no connection between
  the two beyond the number.
- The community ONNX wheel URL and the "PRs already merged" note in the patch
  template will age; refresh periodically.
- **NVFP4 (verified 2026-07, no mod needed)**: `comfy-kitchen` is a stock
  ComfyUI dependency now (`ComfyUI/requirements.txt` pins it directly,
  currently 0.2.18) — not something spark-comfyui.sh adds. Its NVFP4 kernels
  need SM ≥ 10.0 (Blackwell); GB10's sm_121 qualifies, and its fast native
  `cuda` backend (vs the slow pure-PyTorch `eager` fallback) needs torch
  compiled with CUDA ≥ 13 — see `comfy/quant_ops.py`'s `ck.registry.disable
  ("cuda")` gate. That's exactly what `20-torch-repair`/the cu130 install
  already guarantee, so NVFP4 rides for free on the existing torch guarantee;
  confirmed live via `comfy_kitchen.list_backends()`: `cuda` backend
  `available: True` with `quantize_nvfp4`/`scaled_mm_nvfp4` etc. in its
  capabilities — and since 2026-07-11 enforced by a real forced-kernel gate
  in `doctor` (`kitchen_nvfp4_ok`), not just the registry listing. One lever we deliberately don't touch: `--enable-triton-backend`
  (off by default in `comfy/quant_ops.py`, not passed by `cmd_run`) would
  additionally enable Kitchen's Triton backend as a third NVFP4 path —
  unevaluated on GB10, don't adopt without a field test.
- Possible future mod candidates from community forums (evaluate, don't adopt
  blindly): SageAttention 3 (mosaic artifacts on Spark — as of the 2026-07
  research round this is now corroborated in SageAttention's own tracker:
  issues #321 [GB10, the original mosaic report], #334, #340, #357 [active
  2026-07-09], across GB10/RTX 5060 Ti/5070 Ti, all open — stay on the
  `SAGE_REF` pin; bumping past 2.2.x is a deliberate pin change, not
  automatic), the stardust7700 UMA fork (whole-fork merge via the patch
  list, opt-in flags).
- Watch list from the 2026-07 research round (re-check periodically):
  - SageAttention PR #372 (unmerged): Sage2 causal-mask fix for unequal
    q/kv lengths — the one real 2.2.x-line fix pending upstream; if it
    merges, that's the trigger to consider a new `SAGE_REF`. Note upstream
    `main` hasn't moved since our pin (we ARE the tip as of 2026-07-10).
  - `comfy-aimdo` (new ComfyUI pip dep, multi-threaded model loader built
    with DGX Spark in mind, PR #13802): touches the memory-loading path —
    on next install/update, confirm mod 10's anchor still applies.
  - `comfy-kitchen[cublas]` extra — RESOLVED 2026-07-11 by live test: the
    plain wheel's CUDA backend passes a forced NVFP4 quantize+matmul on GB10
    (cosine ~0.99 vs bf16 reference), so the extra is NOT needed on a stock
    install. `doctor` now gates this permanently (`kitchen_nvfp4_ok` in
    mod_common.sh — forces the cuda backend via ck.use_backend, which
    verifiably raises instead of falling back). Also: the forum's
    "comfy-kitchen 0.2.61" claim is NOT a real version; latest is 0.2.18,
    already pinned by ComfyUI's requirements.txt.
  - ComfyUI issue #13920 (open): GB10 hang on second sampling pass with
    `--disable-dynamic-vram` + SageAttention (bisected upstream to commit
    1ac78180). Documented as a SPARK_STATIC_VRAM caveat in the env-var
    overrides section above (README no longer carries an env-var table —
    slimmed 2026-07); if it's fixed upstream, remove the caveat.
- Consider a GitHub issue template that asks for `spark-comfyui.sh doctor`
  output — for this project that one command is nearly a complete bug report.

## Release checklist (v1.0.0 shipped 2026-07-10; repeat per release)

1. `chmod +x spark-comfyui.sh` stays committed (mode bit `100755` in the index).
2. `git add spark-comfyui.sh mods/ README.md LICENSE .gitignore CLAUDE.md`
3. Bump `VERSION=` in spark-comfyui.sh (+ its header comment + the "Current
   version" line above), tag `v<VERSION>` to match `--version`.
4. README clone URL already set to `github.com/bjarkebolding/spark-comfyui`.
