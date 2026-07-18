# spark-comfyui

**One script to install, run, update, and maintain [ComfyUI](https://github.com/Comfy-Org/ComfyUI) on the NVIDIA DGX Spark (GB10 Grace Blackwell).**

## Quick start

Needs: a DGX Spark (GB10) on DGX OS (the NVIDIA driver, CUDA 13.x toolkit, and Python 3.12 all ship with it), plus ~15 GB of disk before models.

```bash
git clone https://github.com/bjarkebolding/spark-comfyui.git
cd spark-comfyui
./spark-comfyui.sh install          # ~30-60 min (SageAttention compiles from source)
./spark-comfyui.sh tune --persist   # recommended: system stability (swap off, persistence mode)
./spark-comfyui.sh run              # UI at http://<spark-ip>:8188
./spark-comfyui.sh update           # now and then: updates itself + ComfyUI + deps, self-heals
```

Everything installs under the directory the script lives in. Fully relocatable, idempotent, safe to re-run. `sudo` is only invoked when a system package is missing or a `tune` setting isn't already applied. Deep GB10 details live in [CLAUDE.md](CLAUDE.md) and [mods/README.md](mods/README.md).

## Commands

| Command | What it does |
|---|---|
| `install [--with-service]` | Full setup: PyTorch cu130, then ComfyUI + Manager deps, then the mods pass (setuptools pin, torch verify, Manager config, SageAttention built natively and kernel-verified, GPU onnxruntime). Re-running refreshes rather than breaks. |
| `run [args...]` | Starts ComfyUI with GB10-tuned flags and environment. Extra args pass through to `main.py`. Re-verifies the SageAttention kernel before every launch and auto-rebuilds if something broke it. |
| `stop` | Stops ComfyUI (systemd service or foreground process). |
| `update [--torch] [--rollback]` | Self-updates spark-comfyui itself first (git fast-forward, only when this repo has newer commits), then updates ComfyUI and dependencies, rebuilds SageAttention only when needed, repairs anything shadowed, and ends with a clear summary. `--torch` upgrades PyTorch (forces a Sage rebuild). `--rollback` returns to the pre-update revision. |
| `doctor` | Full health check. Verifies every optimization is present **and active**, and diagnoses the GB10 silent-drift traps: shadowed torch/SageAttention/onnxruntime, silent attention fallbacks, dead quantization (NVFP4) backend, stale toolchain, swap, stuck clocks. Every failure names its fix. |
| `status [--watch [SEC]]` | One-page glance: process, GPU temp/power/memory, versions, branch, config. `--watch` opens a live dashboard (every 5s or `SEC`): heat-colored sparkline timeseries for GPU, memory and system health, rows that appear only when they carry information (throttle flags, swap, generation telemetry from ComfyUI's own API), and a `session:` summary made for A/B testing (gen count, first vs steady duration, mean it/s). Every sample lands in `thermal_monitor.log`, including per-process GPU memory, so co-resident LLMs show up there. That log is the post-mortem evidence for silent hard-reboots. |
| `tune [--clock-cap MHZ] [--persist]` | System stability: disables swap, sets GPU persistence mode, optional clock cap. `--persist` makes it survive reboots via systemd. |
| `backup [--with-output] [FILE]` | Writes a small tgz of the hand-made state: workflows and settings, inputs, config files, the custom-node set (git nodes as pinned manifest entries, plain nodes copied whole), and a manifest of every model file. Models are never archived; a real backup was 3.7M against 74 GB of models. `--with-output` also archives generated images. Safe while ComfyUI runs. |
| `restore FILE` | Rebuilds from a backup: installs first if ComfyUI is missing, merges user state back, re-clones custom nodes at their pinned commits, reruns the mod pass, then lists exactly which model files are missing and their sizes. Idempotent; restoring onto a healthy install reports everything present. |
| `reset [--yes]` | Deletes and reinstalls ComfyUI, the venv and SageAttention (including the 10-30 min SageAttention build) while preserving models, workflows, settings, inputs, outputs and custom nodes. Asks you to type `reset` before touching anything; `--yes` skips the prompt. An interrupted reset resumes safely on rerun. |
| `service` | Installs and starts a systemd user service (auto-start, restart-on-failure, survives logout). |

`./spark-comfyui.sh help` prints the full reference; `--version` prints the version.

## What it looks like

A routine `update`. The tool self-updates first, ComfyUI moves forward, every mod re-verifies itself, and the summary says whether a restart is needed:

```console
$ ./spark-comfyui.sh update
                          __                              ____            _
   _________  ____ ______/ /__      _________  ____ ___  / __/_  ____  __(_)
  / ___/ __ \/ __ `/ ___/ //_/_____/ ___/ __ \/ __ `__ \/ /_/ / / / / / / /
 (__  ) /_/ / /_/ / /  / ,< /_____/ /__/ /_/ / / / / / / __/ /_/ / /_/ / /
/____/ .___/\__,_/_/  /_/|_|      \___/\____/_/ /_/ /_/_/  \__, /\__,_/_/
    /_/                                                   /____/
  v2026.07.13.1 — ComfyUI on the NVIDIA DGX Spark (GB10 Grace Blackwell)

==> Checking ComfyUI for updates
ComfyUI master updated: 3f8a12c4 -> b96e02d1
b96e02d1 Bump frontend version
77410cf3 Fix mask composite when destination has alpha
5c19aa08 Speed up latent preview interval

==> Refreshing python dependencies
[... pip output ...]

==> Applying mods (idempotent, self-healing)
  = 05-setuptools-compat (already active)
  = 10-unified-memory-free (already active)
  = 20-torch-repair (already active)
  = 30-manager-config (already active) — config OK
SageAttention: OK — verified, no rebuild needed
  = 40-sageattention (already active) — verified, no rebuild needed
onnxruntime: OK — GPU provider live
  = 50-onnxruntime-gpu (already active) — GPU provider live

==> Update summary
  ComfyUI:        updated -> b96e02d1
  Patches:        none
  SageAttention:  verified (no rebuild needed)
  Mods:           active: 05-setuptools-compat 10-unified-memory-free 20-torch-repair 30-manager-config 40-sageattention 50-onnxruntime-gpu
  torch:          2.13.0+cu130 (pins enforced by Manager)
  onnxruntime:    GPU provider live

Changes applied — restart to pick them up: ./spark-comfyui.sh run
```

A healthy `doctor` (banner cropped). Every optimization is checked **live**, not by version strings, and any failure names its exact fix:

```console
$ ./spark-comfyui.sh doctor

== spark-comfyui (self) ==
  [info] git revision d6d7ab4
  [info] Backup: spark-backup-20260716-074352.tgz (today)
  [info] up to date with the published repo

== PyTorch / GPU (CPU-shadow check) ==
  torch 2.13.0+cu130 | compiled CUDA 13.0
  device: NVIDIA GB10 | sm_121
  [PASS] torch is the cu130 CUDA build and sees the GPU
  [PASS] pip dependency graph is consistent
  [info] (ignored benign aarch64 platform-metadata notice for nvidia-*-cu13)

== SageAttention (pip-shadow + kernel-image check) ==
  [PASS] install-time verification marker present
  [PASS] live sm_121 kernel runs (local build intact, not shadowed)
  [info] embedded cubin: sm_121     embedded PTX: sm_121
  [PASS] extension has an sm_121 path (native cubin and/or PTX fallback)
  [info] sageattention distribution origin: local
  [PASS] python3.12-dev present (Triton can JIT — no silent per-call fallback)
  [PASS] no runtime SageAttention fallbacks in ComfyUI's log

== onnxruntime (preprocessor GPU check) ==
  [PASS] CUDAExecutionProvider live — preprocessors run on GPU

== comfy-kitchen (NVFP4/FP8 quantization backends) ==
  [PASS] NVFP4 kernels live on the native CUDA backend (forced + numerically verified)

== NVRTC (GPU-FFT custom-node check) ==
  [PASS] no bundled NVRTC — torch uses the system CUDA 13 one (libnvrtc.so.13)

== ptxas (sm_121 capability) ==
  [PASS] ptxas is CUDA >= 13.0 — sm_121-capable (release 13.0)

== Runtime (is the optimization actually active?) ==
  [PASS] running ComfyUI was launched WITH --use-sage-attention

== GPU clocks (stuck-low check) ==
  [PASS] SM clock reached 2418 MHz under load — no stuck-clock state

== Driver / CUDA stack (informational) ==
  [info] driver: 580.95.05   CUDA (driver): 13.0   toolkit (nvcc): 13.0
  [info] no pending NVIDIA/CUDA apt updates (refresh with: sudo apt update)

== Mods (GB10 fixes & config) ==
  [PASS] 05-setuptools-compat active — setuptools pinned within torch's declared constraint
  [PASS] 10-unified-memory-free active — unified-memory-aware get_free_memory() (fixes offload cliff with co-resident CUDA procs)
  [PASS] 20-torch-repair active — torch CUDA 13 build verified/repaired (install-time + pre-launch guard)
  [PASS] 30-manager-config active — ComfyUI-Manager config (personal_cloud, uv, downgrade_blacklist)
  [PASS] 40-sageattention active — SageAttention built natively for sm_121, live-kernel-verified
  [PASS] 50-onnxruntime-gpu active — GPU onnxruntime (sm_121) for DWPose/ControlNet preprocessors

== Unified-memory safety ==
  [PASS] swap disabled (clean OOM instead of silent freeze)

== Summary ==
  20 passed, 0 failed
  No silent-drift issues detected.
```

`status --watch` during a run. Every line is a timeseries, heat-colored by value. Rows for throttle flags, swap and generation telemetry render only when they carry information, and each appears with its window history intact because sampling never stops.

The GENERATION section (its header names the attention backend) comes from ComfyUI's own API: `gen` duration history with the in-flight one ticking in the margin, live sampling speed (`it/s`; a drop mid-window means throttling or background load), `latency` from queue submission to saved output, queue depth, and the node-cache `hit rate` (high means repeat jobs reuse prompt embeds and loaded models). Every sample also lands in `thermal_monitor.log` so the trail survives a hard reboot. When that log passes 50 MB, the next watch start moves it to `thermal_monitor.log.1` and begins fresh:

```console
$ ./spark-comfyui.sh status --watch 3
spark-comfyui v2026.07.15.1 — sparky · driver 580.159.03 — every 3s, window 105s — Ctrl-C stops
log: /home/user/spark-comfyui/thermal_monitor.log

  ─ GPU ──────────────────────────────────────────────────────────────────────
  temp         57°C ↗                   ▁▁▁▃▅▅▅▁▁▁▅▆▆▆▁▁▁▁▅▅▆█▃▁▁  46–61 ~53
  power      77.40W ↗                   ▁▁▁▇███▁▁▁████▁▁▁▁███▅▁▁▁  10.4–77.4 ~32.5
  sm clk    2411MHz                     ▂▂▅█▆▆▆▂▂▂▆▅▅▅▂▂▂▂▅▅▅▅▂▂▂  2398–2463 ~2421 P0
  gpu           96%                     ▁▁▄▇███▁▁▁████▁▁▁▁████▁▁▁  0–96 ~33.6
  ─ SYSTEM ───────────────────────────────────────────────────────────────────
  unified     22.7G                     ▁▁▃██████████████████████  4.1–23.6 ~20.9 of 122G
  cpu            5%                      ▁▃█▅▆▅▃▁▁▄▅▅▅▂▁▁▁▅▅▅▆▁▁▁  1–10 ~3.1
  disk io   0.0MB/s                      ▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁█▁▁▁▁▁▁█▁  0–0.4 ~0.04
  ─ GENERATION · SageAttention ───────────────────────────────────────────────
  gen         12.6s                             ███████▁▁▁▁▁▁▁▁▁▁  12.5–15.6 ~13.4 at 07:52:43
  it/s         0.79                         ▁▃▃▄   █▃▃▄    █▃▃     0.4–1.36 ~0.79
  latency       10s                             ███████▅▅▅▅▅▅▅▁▁▁  10–16 ~12.3
  queue           0                     ▁▁▁█████▁▁▁████▁▁▁▁███▁▁▁  0–1 ~0.34
  hit rate      67%                             ▁▁▁▁▁▁▁██████████  0–67 ~49.6
  session: 3 gens · first 15.6s · steady ~12.6s (12.5–12.6) · ~0.79 it/s

  samples: 35 · elapsed: 1m49s
```

The `session:` line is made for A/B testing. It aggregates every gen that finished under this watch: `first` carries the model-load cost, `steady` (with min and max) excludes it, plus the session-mean sampling rate and an error count if any. Stats reset each launch and gens that predate the watch never count. Run one watch per condition (a flag, a clock cap, a co-resident LLM) and compare the two lines.

Idle and healthy, the dashboard is seven rows. Throttle renders only after a real slowdown flag, swap only if it exists, gen telemetry only with data:

```console
  ─ GPU ──────────────────────────────────────────────────────────────────────
  temp         41°C                     ▅▅▅▅▅▅▅▅▅▅▅▅▅▅▅▅▅▅  41–41 ~41
  power       4.07W ↘                   ▂▂▁▂▁█▇▂▁▁▂▁▂▂▁▂▁▁  4.03–4.32 ~4.09
  sm clk     208MHz                     ▅▅▅▅▅▅▅▅▅▅▅▅▅▅▅▅▅▅  208–208 ~208 P8
  gpu            0%                     ▅▅▅▅▅▅▅▅▅▅▅▅▅▅▅▅▅▅  0–0 ~0
  ─ SYSTEM ───────────────────────────────────────────────────────────────────
  unified      4.1G                     ▅▅▅▅▅▅▅▅▅▅▅▅▅▅▅▅▅▅  4.1–4.1 ~4.1 of 122G
  cpu            1%                     ▁▁▁▁█▁▁▁▁▁▁▁▁▁▁▁▁▁  0–6 ~1.2
  disk io   0.0MB/s                     ▅▅▅▅▅▅▅▅▅▅▅▅▅▅▅▅▅▅  0–0 ~0
  ─ GENERATION · SageAttention ───────────────────────────────────────────────

  samples: 36 · elapsed: 3m00s
```

## Backup, restore, reset

These commands solve two problems: an install rots (a custom node wrecked the venv, doctor's fixes don't stick) and you want to rebuild it without losing your content, or you want to move everything to a fresh Spark. `reset` rebuilds in place. `backup` plus `restore` cover both, including the move.

`backup` archives what took human effort: workflows and settings, inputs, config files, and the custom-node set (git nodes as pinned url+commit manifest entries, plain nodes copied whole). Models are manifested (path and size), never archived, which is why the archive stays small. Safe while ComfyUI is running:

```console
$ ./spark-comfyui.sh backup

==> Staging backup

==> Backup written
  /home/user/spark-comfyui/backups/spark-backup-20260716-074352.tgz (3.7M)
  models manifested, not archived — restore lists what to re-download
```

`restore` rebuilds from the archive: installs first if ComfyUI is missing, merges user state back, re-clones git custom nodes at their exact pinned commits, reruns the mod pass (so torch gets re-verified after node installs), then diffs the models manifest against disk and prints exactly which model files are missing and their sizes. Restoring onto a healthy install reports everything present:

```console
$ ./spark-comfyui.sh restore backups/spark-backup-20260716-074352.tgz

==> Unpacking /home/user/spark-comfyui/backups/spark-backup-20260716-074352.tgz
  format=1
  version=2026.07.16
  date=2026-07-16T07:43:52+02:00
  host=sparky
  comfyui_commit=87d23b81765161624889febfb3b81f19f3c8435b

==> Stopping ComfyUI (restoring over its live user/config files)
ComfyUI is not running

==> Merging user state
  merged user/
  merged input/
  restored comfyui-patches.list

==> Restoring custom nodes
  = comfyui-workflow-models-downloader (present)

==> Applying mods (idempotent, self-healing)
[... all six mods re-verify ...]

==> Models check (against the archive's manifest)
  all manifested models present

==> Restore complete
  Start ComfyUI:  ./spark-comfyui.sh run
```

Moving to a new Spark: clone spark-comfyui on the new machine, copy the backup archive over, rsync the models directory, run `restore`. `doctor` reports the newest backup and its age, so a stale backup shows up in every health check.

`reset` is the in-place rebuild: it deletes ComfyUI, the venv and SageAttention, then reruns install (including the 10-30 min SageAttention build), while models, workflows, settings, inputs, outputs and custom nodes are held aside and moved back afterward (a same-filesystem move, instant even for 74 GB of models). It asks you to type the word `reset` before deleting anything; `--yes` skips the prompt (required when stdin is not a terminal). An interrupted reset resumes safely on rerun.

## Patch list (optional)

Track unmerged upstream PRs or fork branches without dirtying your tree. Create/edit `comfyui-patches.list` next to the script (a commented template is seeded on first install):

```
pr:12345                                             # a ComfyUI pull request
branch:some-origin-branch                            # a branch on origin
remote:https://github.com/user/ComfyUI.git branch    # a fork's branch
```

On every install/update, a `spark-patched` branch is rebuilt from scratch: fresh upstream master, then each entry merged in order. Master stays pristine. Conflicting entries are skipped with a warning; entries already merged upstream are flagged for removal. Empty list (the default) means plain master tracking, which as of mid-2026 is the optimal configuration.

## Troubleshooting

Start with:

```bash
./spark-comfyui.sh doctor
```

Every check prints PASS/FAIL and, on failure, the exact command that fixes it. Common ones:

- **`no kernel image is available for execution on the device`**: stale/shadowed SageAttention build or pre-13.0 toolchain. `doctor` pinpoints which; `update` rebuilds.
- **Sudden slowness after installing a custom node**: usually shadowed torch or onnxruntime. `update` repairs both automatically.
- **Silent hard-reboot during video generation**: run `status --watch`, reproduce, check the last logged lines. A power spike right before death means overcurrent; fix with `tune --clock-cap 2100`.
- **Everything runs but much slower than it should**: `doctor` checks ComfyUI's own log for silent per-call SageAttention fallbacks and for GPU clocks stuck low after a prior OOM/power event (the latter needs a full power cycle, not a reboot).
- **Whole machine freezes near memory limit**: swap thrash on unified memory. Run `tune` (disables swap; you get a clean OOM kill instead).
- **Manager says an action is not allowed**: lower `security_level` in `ComfyUI/user/__manager/config.ini` temporarily (`normal-` or `weak`), restore it afterward.
- **Capability warning at startup** (`sm_121 exceeds torch's supported maximum`): expected on GB10, harmless; PTX JIT covers it.
- **Spark rebooted mid-update?** The DGX Dashboard's system-update flow always ends in an automatic reboot. Don't run `spark-comfyui.sh update` in the same window as a system/firmware update.

## Security notes

- `network_mode = personal_cloud` relaxes Manager's security gating so it works while serving on `0.0.0.0`. Fine on a trusted LAN. Do not expose the port directly to the internet.
- The GPU onnxruntime wheel and any patch-list forks are community builds, not NVIDIA/Comfy-Org releases. Review what you point the script at.

---

MIT, see [LICENSE](LICENSE). The GB10 knowledge encoded here comes from the NVIDIA DGX Spark developer forums, the [dgx-spark-playbooks](https://github.com/NVIDIA/dgx-spark-playbooks), and the community projects that mapped this hardware in public.
