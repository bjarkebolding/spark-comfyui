# spark-comfyui

**One script to install, run, update, and maintain [ComfyUI](https://github.com/Comfy-Org/ComfyUI) on the NVIDIA DGX Spark (GB10 Grace Blackwell).**

## Quick start

Needs: a DGX Spark (GB10) on DGX OS — the NVIDIA driver, CUDA 13.x toolkit, and Python 3.12 all ship with it — plus ~15 GB of disk before models.

```bash
git clone https://github.com/bjarkebolding/spark-comfyui.git
cd spark-comfyui
./spark-comfyui.sh install          # ~30-60 min (SageAttention compiles from source)
./spark-comfyui.sh tune --persist   # recommended: system stability (swap off, persistence mode)
./spark-comfyui.sh run              # UI at http://<spark-ip>:8188
```

Everything installs under the directory the script lives in — fully relocatable, idempotent, safe to re-run. `sudo` is only invoked when a system package is missing or a `tune` setting isn't already applied. Deep GB10 details live in [CLAUDE.md](CLAUDE.md) and [mods/README.md](mods/README.md).

## Commands

| Command | What it does |
|---|---|
| `install [--with-service]` | Full setup: PyTorch cu130 → ComfyUI + Manager deps → the mods pass (setuptools pin, torch verify, Manager config, SageAttention built natively + kernel-verified, GPU onnxruntime). Re-running refreshes rather than breaks. |
| `run [args...]` | Starts ComfyUI with GB10-tuned flags and environment. Extra args pass through to `main.py`. Re-verifies the SageAttention kernel before every launch and auto-rebuilds if something broke it. |
| `stop` | Stops ComfyUI (systemd service or foreground process). |
| `update [--torch] [--rollback]` | Self-updates spark-comfyui itself first (git fast-forward, only when this repo has newer commits), then updates ComfyUI + dependencies; rebuilds SageAttention only when needed; repairs anything shadowed; ends with a clear summary. `--torch` upgrades PyTorch (forces a Sage rebuild). `--rollback` returns to the pre-update revision. |
| `doctor` | Full health check. Verifies every optimization is present **and active**, and diagnoses the GB10 silent-drift traps: shadowed torch/SageAttention/onnxruntime, silent attention fallbacks, dead quantization (NVFP4) backend, stale toolchain, swap, stuck clocks. Every failure names its fix. |
| `status [--watch [SEC]]` | One-page glance: process, GPU temp/power/memory, versions, branch, config. `--watch` opens a live sparkline dashboard (temp, power, SM clock, GPU util, unified RAM, CPU — every 5s or `SEC`) while appending every sample to `thermal_monitor.log` — post-mortem evidence for silent hard-reboots. |
| `tune [--clock-cap MHZ] [--persist]` | System stability: disables swap, sets GPU persistence mode, optional clock cap. `--persist` makes it survive reboots via systemd. |
| `service` | Installs and starts a systemd user service (auto-start, restart-on-failure, survives logout). |

`./spark-comfyui.sh help` prints the full reference; `--version` prints the version.

## What it looks like

A routine `update` — the tool self-updates first, ComfyUI moves forward, every mod re-verifies itself, and the summary says whether a restart is needed:

```console
$ ./spark-comfyui.sh update
                          __                              ____            _
   _________  ____ ______/ /__      _________  ____ ___  / __/_  ____  __(_)
  / ___/ __ \/ __ `/ ___/ //_/_____/ ___/ __ \/ __ `__ \/ /_/ / / / / / / /
 (__  ) /_/ / /_/ / /  / ,< /_____/ /__/ /_/ / / / / / / __/ /_/ / /_/ / /
/____/ .___/\__,_/_/  /_/|_|      \___/\____/_/ /_/ /_/_/  \__, /\__,_/_/
    /_/                                                   /____/
  v1.4.0 — ComfyUI on the NVIDIA DGX Spark (GB10 Grace Blackwell)

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

A healthy `doctor` (banner cropped) — every optimization is checked **live**, not by version strings, and any failure names its exact fix:

```console
$ ./spark-comfyui.sh doctor

== spark-comfyui (self) ==
  [info] git revision d6d7ab4
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

`status --watch` during a generation — the power ramp at denoise start (the overcurrent-reboot trigger on some units) is exactly what the sparklines make visible, and every sample also lands in `thermal_monitor.log` so the trail survives a hard reboot:

```console
$ ./spark-comfyui.sh status --watch 2
spark-comfyui v1.4.0 — live telemetry, every 2s (window 80s) — Ctrl-C stops
log: /home/user/spark-comfyui/thermal_monitor.log

  temp          44°C                      ▁▁▁▁▃▃▅▇▇▆██▃▃▃▃▃▃▃▃  39–59
  power       10.72W                      ▁▁▁▂▃▄▅█████▁▁▁▁▁▁▁▁  10.27–73.88
  sm clk     2411MHz                      ▁▁▁▃▇▇█▇▇▇▆▆▂▂▂▂▂▂▂▂  2398–2463
  gpu             0%                      ▁▁▁▁▆▆██████▁▁▁▁▁▁▁▁  0–96
  unified      23.7G                      ▁▁▂▃▅▇██████████████  4.9–24.3  of 122G
  cpu             0%                       ▂█▆▆██▆▆▆▆▆▃▁▂▁▂▂▂▁  0–9

  swap: disabled (good)
  ComfyUI: RUNNING pid 12345 rss 2.5G [SageAttention]

  a power spike as the LAST sample before a silent reboot = overcurrent
  -> fix: ./spark-comfyui.sh tune --clock-cap 2100
```

## Patch list (optional)

Track unmerged upstream PRs or fork branches without dirtying your tree. Create/edit `comfyui-patches.list` next to the script (a commented template is seeded on first install):

```
pr:12345                                             # a ComfyUI pull request
branch:some-origin-branch                            # a branch on origin
remote:https://github.com/user/ComfyUI.git branch    # a fork's branch
```

On every install/update, a `spark-patched` branch is **rebuilt from scratch**: fresh upstream master, then each entry merged in order. Master stays pristine. Conflicting entries are skipped with a warning; entries already merged upstream are flagged for removal. Empty list (the default) = plain master tracking, which as of mid-2026 is the optimal configuration.

## Troubleshooting

Start with:

```bash
./spark-comfyui.sh doctor
```

Every check prints PASS/FAIL and, on failure, the exact command that fixes it. Common ones:

- **`no kernel image is available for execution on the device`** — stale/shadowed SageAttention build or pre-13.0 toolchain. `doctor` pinpoints which; `update` rebuilds.
- **Sudden slowness after installing a custom node** — usually shadowed torch or onnxruntime. `update` repairs both automatically.
- **Silent hard-reboot during video generation** — run `status --watch`, reproduce, check the last logged lines: a power spike right before death means overcurrent → `tune --clock-cap 2100`.
- **Everything runs but much slower than it should** — `doctor` checks ComfyUI's own log for silent per-call SageAttention fallbacks and for GPU clocks stuck low after a prior OOM/power event (the latter needs a full power cycle, not a reboot).
- **Whole machine freezes near memory limit** — swap thrash on unified memory → `tune` (disables swap; you get a clean OOM kill instead).
- **Manager says an action is not allowed** — lower `security_level` in `ComfyUI/user/__manager/config.ini` temporarily (`normal-` or `weak`), restore it afterward.
- **Capability warning at startup** (`sm_121 exceeds torch's supported maximum`) — expected on GB10, harmless; PTX JIT covers it.
- **Spark rebooted mid-update?** The DGX Dashboard's system-update flow always ends in an automatic reboot. Don't run `spark-comfyui.sh update` in the same window as a system/firmware update.

## Security notes

- `network_mode = personal_cloud` relaxes Manager's security gating so it works while serving on `0.0.0.0`. Fine on a trusted LAN — **do not** expose the port directly to the internet.
- The GPU onnxruntime wheel and any patch-list forks are community builds, not NVIDIA/Comfy-Org releases. Review what you point the script at; that's the trade-off for running hardware this new.

---

MIT — see [LICENSE](LICENSE). The GB10 knowledge encoded here comes from the NVIDIA DGX Spark developer forums, the [dgx-spark-playbooks](https://github.com/NVIDIA/dgx-spark-playbooks), and the community projects that mapped this hardware in public.
