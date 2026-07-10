# spark-comfyui

**One script to install, run, update, and maintain [ComfyUI](https://github.com/Comfy-Org/ComfyUI) on the NVIDIA DGX Spark (GB10 Grace Blackwell).**

The DGX Spark is not an ordinary CUDA box: it is aarch64 (Grace CPU), its GPU reports compute capability **sm_121** that most toolchains don't target yet, and its 128 GB of **unified memory** breaks assumptions that ordinary ComfyUI setups make about VRAM. A generic install either fails outright or — worse — runs in silently degraded states: CPU-only torch, attention kernels that quietly fall back, preprocessors on CPU. This script exists to make all of that boring.

```
./spark-comfyui.sh install     # one-shot setup
./spark-comfyui.sh run         # start ComfyUI
./spark-comfyui.sh update      # keep everything current
./spark-comfyui.sh doctor      # something feels wrong? this tells you what & how to fix it
```

Everything installs **under the directory the script lives in** — drop it anywhere, run it, and `ComfyUI/`, `comfyui-env/`, and `SageAttention/` appear next to it. Fully relocatable, idempotent, safe to re-run.

---

## Repository layout

```
spark-comfyui.sh          # the single entry point — vanilla install/run/
                          # update lifecycle + dispatch
mods/                     # everything GB10-specific, applied & self-healed
                          # automatically
  _lib/mod_common.sh      # shared helpers, incl. the venv-package functions
  05-setuptools-compat/   # setuptools pinned within torch's own constraint
  10-unified-memory-free/ # get_free_memory() unified-pool fix
  20-torch-repair/        # torch CUDA verified/repaired (install/update +
                          # before every `run`)
  30-manager-config/      # ComfyUI-Manager config.ini (network_mode, uv,
                          # downgrade_blacklist)
  40-sageattention/       # native sm_121 build + live kernel verification
  50-onnxruntime-gpu/     # community sm_121 GPU wheel for preprocessors
  README.md               # how to write your own mod
README.md · LICENSE · .gitignore
```

The script installs ComfyUI, the venv, and SageAttention *next to itself*, so
keep `spark-comfyui.sh` and `mods/` together — the script locates mods relative
to its own path.

## Requirements

- NVIDIA DGX Spark (GB10) running DGX OS with the NVIDIA driver and **CUDA 13.x toolkit** (`nvcc --version` → release 13.x)
- Python 3.12 (ships with DGX OS)
- `sudo`, only actually invoked when a system package is missing or a `tune` setting isn't already applied
- Disk space: ComfyUI + venv + SageAttention build ≈ 15 GB, before models

## Quick start

```bash
git clone https://github.com/bjarkebolding/spark-comfyui.git
cd spark-comfyui
chmod +x spark-comfyui.sh            # make the script executable (first time only)
./spark-comfyui.sh install          # ~30-60 min (SageAttention compiles from source)
./spark-comfyui.sh tune --persist   # recommended: system stability tuning (see below)
./spark-comfyui.sh run              # UI at http://<spark-ip>:8188
```

## Commands

| Command | What it does |
|---|---|
| `install` | Full setup: PyTorch cu130 → ComfyUI + Manager deps → the mods pass (setuptools pin, torch verify, Manager config, SageAttention built natively + kernel-verified, GPU onnxruntime). Re-running refreshes rather than breaks. |
| `run [args...]` | Starts ComfyUI with GB10-tuned flags and environment. Extra args pass through to `main.py`. Re-verifies the SageAttention kernel before every launch and auto-rebuilds if something broke it. |
| `stop` | Stops ComfyUI (systemd service or foreground process). |
| `update [--torch] [--rollback]` | Updates ComfyUI + dependencies; rebuilds SageAttention only when its source or torch changed; repairs anything shadowed; ends with a clear summary and tells you whether a restart is needed. `--torch` upgrades PyTorch (forces a Sage rebuild). `--rollback` returns to the pre-update revision. |
| `doctor` | Full health check. Verifies every optimization is present **and active**, and diagnoses the GB10 silent-drift traps (see below). Every failure names its fix. |
| `status [--watch]` | One-page glance: process, GPU temp/power/memory, versions, branch, config. `--watch` logs a 5-second telemetry trail — post-mortem evidence for silent hard-reboots. |
| `tune [--clock-cap MHZ] [--persist]` | System stability: disables swap, sets GPU persistence mode, optional clock cap. `--persist` makes it survive reboots via systemd. |
| `service` | Installs and starts a systemd user service (auto-start, restart-on-failure, survives logout). |

`./spark-comfyui.sh help` prints the full reference; `--version` prints the version.

## What makes the GB10 different (and what the script does about it)

| Trap | Symptom | What the script does |
|---|---|---|
| **sm_121 kernel gap** | `no kernel image is available for execution on the device`; ComfyUI silently falls back to slow attention | SageAttention is compiled natively with `TORCH_CUDA_ARCH_LIST="12.1+PTX"` (native cubin **plus** PTX JIT fallback) and must pass a live multi-shape GPU kernel test before it is ever enabled |
| **CPU-shadowed torch** | A custom node's requirements silently replace CUDA torch with a CPU build | Detected and repaired before every launch and every update; additionally **prevented** at the Manager level via `downgrade_blacklist` |
| **pip-shadowed SageAttention** | A later `pip install sageattention` overwrites the local sm_121 build with a kernel-less PyPI wheel — everything still "works", just slower | `run` re-tests the actual kernel before enabling the flag and rebuilds automatically if it fails; `doctor` identifies the shadow |
| **CPU-only onnxruntime** | DWPose / ControlNet preprocessors run on CPU because PyPI ships no aarch64+cu13 GPU wheel | Installs a community sm_121 GPU wheel and verifies `CUDAExecutionProvider` is live; re-asserted on every update |
| **Unified-memory swap freeze** | Heavy video workloads thrash swap and silently freeze the whole box (no logs) | `tune` disables swap → clean OOM kill instead; `run` warns if swap is on |
| **CUDA free-memory under-reporting** | With a second CUDA process resident (e.g. vLLM), ComfyUI sees ~6 GB free instead of 40+, needlessly offloads, and sampling time balloons 5–15× | A source patch makes `get_free_memory()` read the host-available unified pool (`psutil`) instead of the misleading CUDA query; auto-applied and re-healed on every update |
| **Overcurrent hard-reboots** | Instant reboot mid-generation with no logs (GPU power spike trips protection on some units) | `tune --clock-cap 2100` locks clocks to the community-validated stable range; `status --watch` captures the evidence trail |
| **Stale toolchain** | Builds succeed but produce unrunnable kernels | `doctor` checks ptxas is CUDA ≥ 13.0 and NVRTC resolution is sane |
| **Dependency pin drift** | e.g. setuptools upgraded past torch's `<82` pin, breaking later source builds | Pins realigned from torch's own metadata on install/update; `doctor` runs a full `pip check` |

### Runtime tuning applied by `run`

- `CUDA_CACHE_MAXSIZE=4GB` — kernel cache; ~3× faster denoise steps on reruns (first run JITs PTX→SASS for sm_121, then reuses from disk)
- `NCCL_P2P_DISABLE=1` — single GPU, so P2P negotiation is pure overhead; skip it
- `--disable-pinned-memory` — pinned memory is overhead, not a win, on the unified fabric
- `--bf16-unet --bf16-vae --bf16-text-enc` — native fast path on GB10 (opt out: `SPARK_BF16=0`)
- `--use-sage-attention` — only when the build passed live kernel verification
- `--disable-dynamic-vram` — opt-in (`SPARK_STATIC_VRAM=1`): keep models resident when they fit

**Everything GB10-specific beyond a vanilla install** (auto-applied, re-healed on every update; opt out with `SPARK_SOURCE_PATCHES=0`) lives as self-contained **mods** under [`mods/`](mods/) — each in its own directory with a `run.sh` implementing an `apply`/`verify`/`describe` contract, discovered and run in filename order. That covers three kinds of things: a source patch to ComfyUI's own Python (`10-unified-memory-free` makes `get_free_memory()` read the host-available unified pool instead of the misleading CUDA query — the single most impactful fix when another CUDA process shares the GPU), a config-tree mod (`30-manager-config`, ComfyUI-Manager's `config.ini`), and venv-package steps described in the table above (`05-setuptools-compat`, `20-torch-repair`, `40-sageattention`, `50-onnxruntime-gpu`). Source patches are idempotent, back up the original once (`*.spark-orig`), revert themselves if they'd produce invalid Python, and no-op if upstream changes the code they target. The venv-package mods that can take real time or whose failure genuinely breaks the install (torch repair, the SageAttention build) stream their output live and abort the script loudly on failure rather than degrading silently. Adding your own is a matter of dropping a new directory in `mods/` — see [`mods/README.md`](mods/README.md).
- Deliberately **not** used: `--gpu-only` / `--highvram` (fights unified memory), `torch.compile` (≈0 % gain on GB10), inductor FX graph cache (served stale graphs), `PYTORCH_NO_CUDA_MEMORY_CACHING` (causes fragmentation → OOM), Flash Attention (FA3 can't target sm_121; FA2 compiles from source in ~2 h but loses to SDPA on Blackwell — only worth it if a custom node hard-imports `flash_attn`)

## Patch list (optional)

Track unmerged upstream PRs or fork branches without dirtying your tree. Create/edit `comfyui-patches.list` next to the script (a commented template is seeded on first install):

```
pr:12345                                             # a ComfyUI pull request
branch:some-origin-branch                            # a branch on origin
remote:https://github.com/user/ComfyUI.git branch    # a fork's branch
```

On every install/update, a `spark-patched` branch is **rebuilt from scratch**: fresh upstream master, then each entry merged in order. Master stays pristine. Conflicting entries are skipped with a warning; entries already merged upstream are flagged for removal. Empty list (the default) = plain master tracking, which as of mid-2026 is the optimal configuration — the major Spark PRs (async offload, audio VAE) are already merged upstream.

## Configuration

All paths and knobs are environment-overridable:

| Variable | Default | Purpose |
|---|---|---|
| `INSTALL_DIR` | `<script dir>/ComfyUI` | ComfyUI checkout |
| `VENV_DIR` | `<script dir>/comfyui-env` | Python virtualenv |
| `SAGE_SRC` | `<script dir>/SageAttention` | SageAttention source |
| `SAGE_REF` | a commit on the 2.2.x line, field-verified on GB10 | Pinned SageAttention build ref — bump deliberately (3.x had mosaic artifacts on Spark) |
| `REPO_URL` | `github.com/Comfy-Org/ComfyUI.git` | ComfyUI clone source (point at a fork/mirror) |
| `PORT` | `8188` | Web UI port |
| `TORCH_INDEX` | PyTorch cu130 index | Wheel source |
| `ORT_WHEEL_URL` | community sm_121 wheel | GPU onnxruntime |
| `PATCH_LIST` | `<script dir>/comfyui-patches.list` | Patch list location |
| `SPARK_BF16` | `1` | Set `0` to disable forced bf16 flags |
| `SPARK_STATIC_VRAM` | `0` | Set `1` to keep models resident when they fit (`--disable-dynamic-vram`; faster prompt→image iteration) |
| `SPARK_SOURCE_PATCHES` | `1` | Set `0` to skip **all six GB10 mods** — not just the memory-reporting patch, but also torch repair, SageAttention, onnxruntime, and Manager config |

ComfyUI-Manager is configured automatically (`user/__manager/config.ini`): `network_mode = personal_cloud` (required for full Manager function on a `0.0.0.0` listener), `security_level = normal`, `use_uv = True` (fast node installs), `file_logging = True`, and `downgrade_blacklist` protecting the torch trio. Keys you edit by hand are respected on re-runs (only `network_mode` is re-asserted).

## Troubleshooting

Start with:

```bash
./spark-comfyui.sh doctor
```

Every check prints PASS/FAIL and, on failure, the exact command that fixes it. Common ones:

- **`no kernel image is available for execution on the device`** — stale/shadowed SageAttention build or pre-13.0 toolchain. `doctor` pinpoints which; `update` rebuilds.
- **Sudden slowness after installing a custom node** — usually shadowed torch or onnxruntime. `update` repairs both automatically.
- **Silent hard-reboot during video generation** — run `status --watch`, reproduce, check the last logged lines: a power spike right before death means overcurrent → `tune --clock-cap 2100`.
- **Whole machine freezes near memory limit** — swap thrash on unified memory → `tune` (disables swap; you get a clean OOM kill instead).
- **Manager says an action is not allowed** — lower `security_level` in `ComfyUI/user/__manager/config.ini` temporarily (`normal-` or `weak`), restore it afterward.
- **Capability warning at startup** (`sm_121 exceeds torch's supported maximum`) — expected on GB10, harmless; PTX JIT covers it.

## Security notes

- `network_mode = personal_cloud` relaxes Manager's security gating so it works while serving on `0.0.0.0`. Fine on a trusted LAN — **do not** expose the port directly to the internet.
- The GPU onnxruntime wheel and any patch-list forks are community builds, not NVIDIA/Comfy-Org releases. Review what you point the script at; that's the trade-off for running hardware this new.

## Acknowledgements

The GB10 knowledge encoded here comes from the community that mapped this hardware in public: the NVIDIA DGX Spark developer forums, the [nvidia-dgx-spark-playbooks](https://github.com/NVIDIA/dgx-spark-playbooks), community setup guides and kits, and the PyTorch team's aarch64+cu130 wheels. Errors in synthesis are this project's, not theirs.

## License

MIT — see [LICENSE](LICENSE).
