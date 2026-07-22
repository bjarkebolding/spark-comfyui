# spark-comfyui

**One script to install, run, update, and maintain [ComfyUI](https://github.com/Comfy-Org/ComfyUI) on the NVIDIA DGX Spark (GB10 Grace Blackwell) — fully containerized.**

The whole GB10-tuned stack (cu130 PyTorch, native sm_121 SageAttention, GPU onnxruntime, the Spark mods) lives in a docker image. Your content (models, workflows, custom nodes, outputs) lives in a plain `data/` directory next to the script and is bind-mounted in. Custom-node code, which is arbitrary third-party Python, runs confined: non-root, no capabilities, nothing visible but your content and the GPU.

## Quick start

Needs a DGX Spark (GB10) on DGX OS (docker, the NVIDIA Container Toolkit and the r580 driver ship with it), plus ~25 GB of disk before models.

```bash
git clone https://github.com/bjarkebolding/spark-comfyui.git
cd spark-comfyui
./spark-comfyui.sh install          # one-time image build, 10-30 min
./spark-comfyui.sh tune --persist   # recommended: swap off, persistence mode
./spark-comfyui.sh run              # UI at http://<spark-ip>:8188
```

Models go in `data/models/checkpoints` (etc.). No venv, no system Python changes; `sudo` is only used by `tune`.

## Commands

| Command | What it does |
|---|---|
| `install` | Builds the image: ComfyUI at a pinned commit, cu130 PyTorch, SageAttention compiled for sm_121, GPU onnxruntime (sha256-pinned), GB10 mods. Idempotent. |
| `run [args...]` | Starts ComfyUI in the container, foreground. Every start installs custom-node requirements and live-verifies the GPU stack before serving. Extra args pass to `main.py`. |
| `service [--disable]` | Same, detached with a docker restart policy: survives crashes and reboots. |
| `stop` | Stops ComfyUI. |
| `update [--torch] [--rollback]` | Self-updates the tool, then rebuilds the image on current ComfyUI master (cached layers: minutes). The old image stays as `:previous`; `--rollback` swaps back instantly. `--torch` forces fresh PyTorch wheels. |
| `doctor` | Health check: tool and host (driver, docker, image, swap, backups), then the live GPU gates (torch CUDA, sm_121 SageAttention kernel, onnxruntime provider, NVFP4) inside a throwaway container. Every failure names its fix. |
| `status [--watch [SEC]]` | One-page glance, or a live sparkline dashboard with generation telemetry and a `session:` A/B summary. Every sample lands in `thermal_monitor.log`, the post-mortem trail for silent hard-reboots. |
| `tune [--clock-cap MHZ] [--persist]` | Host stability: swap off, persistence mode, optional clock cap (~2100 fixes overcurrent hard-reboots). |
| `backup [--with-output] [FILE]` | Small tgz of workflows, settings, inputs, configs and the custom-node set. Models are manifested, never archived. Safe while running. |
| `restore FILE` | Rebuilds from a backup: image if missing, content into `data/`, custom nodes re-cloned at pinned commits, missing models listed with sizes. |
| `reset [--yes]` | Removes the container, all image tags and the cache volume; rebuilds from scratch. `data/` is never touched. |

## What it looks like

`status --watch` during a run. Heat-colored timeseries; rows appear only when they carry information:

```console
$ ./spark-comfyui.sh status --watch 3

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
  gen         13.1s                             ███████▁▁▁▁▁▁▁▁▁▁  13.1–29.8 ~17.3
  it/s         0.63                         ▁▃▃▄   █▃▃▄    █▃▃     0.4–1.36 ~0.72
  latency       13s                             ███████▅▅▅▅▅▅▅▁▁▁  13–30 ~17.4
  queue           0                     ▁▁▁█████▁▁▁████▁▁▁▁███▁▁▁  0–1 ~0.34
  hit rate      77%                             ▁▁▁▁▁▁▁██████████  0–77 ~57.8
  session: 4 gens · first 29.8s · steady ~13.6s (13.6–13.6) · ~0.63 it/s
```

The `session:` line is made for A/B tests: run one watch per condition and compare. That is how the container itself was validated: native steady 13.59s vs container 13.61s on the same workflow and seeds, every seed-matched output pair bit-identical.

A healthy `doctor` ends like this — the gates run inside a throwaway container from the exact image `run` uses:

```console
== SageAttention (live kernel) ==
  [PASS] sm_121 kernel runs

== onnxruntime GPU ==
  [PASS] CUDAExecutionProvider available

== NVFP4 (comfy-kitchen forced cuda backend) ==
  [PASS] forced NVFP4 quantize+matmul passed

All container gates passed.
Host checks: 4 passed. Everything healthy.
```

## Mounts

Everything lives under `data/` by default. To relocate entries or add extra mounts, edit `spark-mounts.conf` next to the script (a commented template is seeded on install):

```
models = /mnt/fast-ssd/models
output = /mnt/nas/comfyui-output
mount = /mnt/nas/sdxl-models:/opt/ComfyUI/models/nas:ro
```

`status` always prints the resolved table. A typo in a `mount =` line dies loudly instead of silently shadowing your data with an empty directory. Point `extra_model_paths.yaml` entries at the container side of `mount =` lines.

Relative paths resolve against the config file's own directory. For several setups, keep one file each and pick one per invocation with `--mounts`:

```bash
./spark-comfyui.sh --mounts ./nas-profile.conf run
./spark-comfyui.sh --mounts ./nas-profile.conf status   # check what it resolves to
```

The flag works with every command and must name a file that exists, so a typo fails loudly instead of quietly mounting the defaults. `MOUNTS_CONF=PATH` does the same thing as an environment variable. Use it on `backup` and `restore` too so they act on that setup. To run two setups at once, add `CONTAINER_NAME` and `PORT` overrides.

## Patch list (optional)

`comfyui-patches.list` next to the script merges upstream PRs or fork branches (`pr:12345`, `branch:name`, `remote:<url> <branch>`) on top of ComfyUI inside the image build. A conflict fails the build loudly. Empty list means plain master tracking.

## Troubleshooting

Start with `./spark-comfyui.sh doctor` — every failure names its fix. Common ones:

- **An update broke generation**: `update --rollback`, restart.
- **A custom node will not load**: check the start log; the entrypoint installs each node's requirements and warns per node. A node needing a system library the image lacks is worth an issue.
- **Silent hard-reboot during video generation**: `status --watch`, reproduce, read the last logged lines. A power spike right before death is overcurrent; fix with `tune --clock-cap 2100`.
- **Machine freezes near memory limit**: swap thrash on unified memory; run `tune`.
- **`sm_121 exceeds torch's supported maximum` at startup**: expected on GB10, harmless.

## Security notes

- Custom nodes run as a non-root user with all capabilities dropped and only your content directories and the GPU visible. A malicious node cannot read your SSH keys or anything else on the host.
- The image is reproducible from this repo: pinned ComfyUI commit, sha256-pinned onnxruntime wheel, pinned SageAttention. `update --rollback` returns to the previous image atomically.
- Manager's `personal_cloud` mode is fine on a trusted LAN; do not expose the port to the internet.

---

MIT, see [LICENSE](LICENSE). The GB10 knowledge here comes from the NVIDIA DGX Spark developer forums, the [dgx-spark-playbooks](https://github.com/NVIDIA/dgx-spark-playbooks), and the community projects that mapped this hardware in public.
