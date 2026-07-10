# spark-comfyui

**One script to install, run, update, and maintain [ComfyUI](https://github.com/Comfy-Org/ComfyUI) on the NVIDIA DGX Spark (GB10 Grace Blackwell).**

The DGX Spark is not an ordinary CUDA box — aarch64 Grace CPU, an sm_121 GPU most toolchains don't target yet, and 128 GB of unified memory. A generic ComfyUI install either fails outright or runs silently degraded: CPU-only torch, attention kernels quietly falling back, preprocessors on CPU. This script makes all of that boring: every optimization is gated on a live test (never a version string), and everything self-heals on every update. Deep GB10 details live in [CLAUDE.md](CLAUDE.md) and [mods/README.md](mods/README.md).

## Quick start

Needs: a DGX Spark (GB10) on DGX OS — the NVIDIA driver, CUDA 13.x toolkit, and Python 3.12 all ship with it — plus ~15 GB of disk before models.

```bash
git clone https://github.com/bjarkebolding/spark-comfyui.git
cd spark-comfyui
./spark-comfyui.sh install          # ~30-60 min (SageAttention compiles from source)
./spark-comfyui.sh tune --persist   # recommended: system stability (swap off, persistence mode)
./spark-comfyui.sh run              # UI at http://<spark-ip>:8188
```

Everything installs under the directory the script lives in — fully relocatable, idempotent, safe to re-run. `sudo` is only invoked when a system package is missing or a `tune` setting isn't already applied.

## Commands

| Command | What it does |
|---|---|
| `install [--with-service]` | Full setup: PyTorch cu130 → ComfyUI + Manager deps → the mods pass (setuptools pin, torch verify, Manager config, SageAttention built natively + kernel-verified, GPU onnxruntime). Re-running refreshes rather than breaks. |
| `run [args...]` | Starts ComfyUI with GB10-tuned flags and environment. Extra args pass through to `main.py`. Re-verifies the SageAttention kernel before every launch and auto-rebuilds if something broke it. |
| `stop` | Stops ComfyUI (systemd service or foreground process). |
| `update [--torch] [--rollback]` | Updates ComfyUI + dependencies; rebuilds SageAttention only when needed; repairs anything shadowed; ends with a clear summary. `--torch` upgrades PyTorch (forces a Sage rebuild). `--rollback` returns to the pre-update revision. |
| `doctor` | Full health check. Verifies every optimization is present **and active**, and diagnoses the GB10 silent-drift traps: shadowed torch/SageAttention/onnxruntime, silent attention fallbacks, stale toolchain, swap, stuck clocks. Every failure names its fix. |
| `status [--watch]` | One-page glance: process, GPU temp/power/memory, versions, branch, config. `--watch` logs a 5-second telemetry trail — post-mortem evidence for silent hard-reboots. |
| `tune [--clock-cap MHZ] [--persist]` | System stability: disables swap, sets GPU persistence mode, optional clock cap. `--persist` makes it survive reboots via systemd. |
| `service` | Installs and starts a systemd user service (auto-start, restart-on-failure, survives logout). |

`./spark-comfyui.sh help` prints the full reference; `--version` prints the version.

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
