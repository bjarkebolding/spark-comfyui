# spark-comfyui

**One script to install, run, update, and maintain [ComfyUI](https://github.com/Comfy-Org/ComfyUI) on the NVIDIA DGX Spark (GB10 Grace Blackwell) — fully containerized.**

ComfyUI and its whole GB10-tuned stack (cu130 PyTorch, native sm_121 SageAttention, GPU onnxruntime, the Spark mods) live in a docker image. Your content (models, workflows, custom nodes, outputs) lives in a plain `data/` directory next to the script and is bind-mounted in. Custom-node code, which is arbitrary third-party Python, runs confined: non-root, no capabilities, no privilege escalation, nothing visible but your content and the GPU.

## Quick start

Needs: a DGX Spark (GB10) on DGX OS (docker, the NVIDIA Container Toolkit, and the r580 driver all ship with it), plus ~25 GB of disk for the image before models.

```bash
git clone https://github.com/bjarkebolding/spark-comfyui.git
cd spark-comfyui
./spark-comfyui.sh install          # one-time image build, 10-30 min (SageAttention compiles)
./spark-comfyui.sh tune --persist   # recommended: host stability (swap off, persistence mode)
./spark-comfyui.sh run              # UI at http://<spark-ip>:8188
./spark-comfyui.sh update           # now and then: tool + image rebuild, cached layers, minutes
```

Models go in `data/models/checkpoints` (etc.). No venv, no system Python changes; `sudo` is only invoked by `tune` for host settings. Deep GB10 details live in [CLAUDE.md](CLAUDE.md) and [mods/README.md](mods/README.md).

Coming from the native (pre-container) version? See [Migrating](#migrating-from-the-native-version) — the tooling lives in the `v2026.07.20` tag.

## Commands

| Command | What it does |
|---|---|
| `install` | Checks docker + the NVIDIA runtime, seeds the config templates, builds the image: ComfyUI at a pinned commit, cu130 PyTorch, SageAttention compiled natively for sm_121, the community GPU onnxruntime wheel (sha256-pinned), and the GB10 mods baked in. Re-running refreshes rather than breaks. |
| `run [args...]` | Starts ComfyUI in the container, foreground (Ctrl-C stops). Every start re-installs custom-node requirements, verifies torch sees the GPU, and runs a live SageAttention kernel test before serving — a broken stack refuses to launch instead of degrading silently. Extra args pass to `main.py`. |
| `service [--disable]` | Same, detached with a docker restart policy: survives crashes and reboots. `--disable` removes it. |
| `stop` | Stops ComfyUI (container, service, or a stray process). |
| `update [--torch] [--rollback]` | Self-updates spark-comfyui itself first, then rebuilds the image on current ComfyUI master. Docker layer caching makes a routine update minutes, not a full build. The replaced image stays tagged `:previous`; `--rollback` swaps back to it instantly. `--torch` forces fresh cu130 wheels (SageAttention rebuilds on top). |
| `doctor` | Health check in three layers: the tool itself (version, pending update, backup age), the host (driver, docker runtime, image age, a running container on an outdated image, swap), and the live GPU gates — torch CUDA, the sm_121 SageAttention kernel, the onnxruntime GPU provider, and a forced NVFP4 quantize+matmul — executed inside a throwaway container from the exact image `run` uses. Every failure names its fix. |
| `status [--watch [SEC]]` | One-page glance: process (tagged when containerized), GPU, memory, image commit, resolved mounts. `--watch` opens the live dashboard: heat-colored sparkline timeseries, rows that appear only when they carry information, generation telemetry from ComfyUI's own API, and a `session:` summary line made for A/B testing. Every sample lands in `thermal_monitor.log` — the post-mortem evidence for silent hard-reboots. |
| `tune [--clock-cap MHZ] [--persist]` | Host-side stability: disables swap (prevents unified-memory freezes), sets GPU persistence mode, optional clock cap (~2100 fixes overcurrent hard-reboots). `--persist` survives reboots. |
| `backup [--with-output] [FILE]` | Small tgz of the hand-made state: workflows, settings, inputs, config files, the custom-node set (git nodes as pinned manifest entries, plain nodes copied whole), and a manifest of every model file. Models are never archived. Safe while ComfyUI runs. |
| `restore FILE` | Rebuilds from a backup: builds the image if missing, merges content into `data/`, re-clones custom nodes at their pinned commits (their requirements install at next start), then lists exactly which model files are missing and their sizes. |
| `reset [--yes]` | The nuclear option, now content-safe by construction: removes the container, every image tag and the cache volume, then rebuilds the image from scratch. `data/` is never touched. |

`./spark-comfyui.sh help` prints the full reference; `--version` prints the version.

## What it looks like

A healthy `doctor`. The GPU gates run inside a throwaway container from the same image `run` uses, so what passes here is exactly what launches:

```console
$ ./spark-comfyui.sh doctor

== spark-comfyui (self) ==
  [info] git revision f9c8533
  [info] Backup: spark-backup-20260720-095245.tgz (today)
  [info] up to date with the published repo

== container host ==
  [PASS] NVIDIA driver 580.159.03
  [PASS] docker daemon up, nvidia runtime registered
  [PASS] image spark-comfyui:latest (built 2026-07-20, ComfyUI 66655153499f)
  [info] rollback point present (spark-comfyui:previous)
  [info] container not running
  [PASS] swap disabled on the host

==> Running the GPU gates inside a throwaway container

== torch / CUDA ==
  torch 2.13.0+cu130 | compiled CUDA 13.0
  device: NVIDIA GB10 | sm_121
  [PASS] torch is the cu130 CUDA build and sees the GPU

== SageAttention (live kernel) ==
  [PASS] sm_121 kernel runs

== onnxruntime GPU ==
  [PASS] CUDAExecutionProvider available

== NVFP4 (comfy-kitchen forced cuda backend) ==
  [PASS] forced NVFP4 quantize+matmul passed

All container gates passed.

Host checks: 4 passed. Everything healthy.
```

A routine `update`. The upstream commit is the docker cache key, so only the layers from the ComfyUI clone down rebuild; torch and the SageAttention compile stay cached:

```console
$ ./spark-comfyui.sh update

==> Resolving upstream ComfyUI master
==> Building spark-comfyui:2026.07.20 (ComfyUI 66655153499f, SageAttention d1a57a546c3d)
[... cached layers replay, new ComfyUI layers build ...]
==> Image ready: spark-comfyui:latest (also tagged :2026.07.20)
==> Updated. The old image stays as spark-comfyui:previous
  Roll back with: ./spark-comfyui.sh update --rollback
```

`status --watch` during a run. Every line is a timeseries, heat-colored by value; rows for throttle flags, swap and generation telemetry render only when they carry information. Works identically for containerized and native servers — the container's process is visible to the host:

```console
$ ./spark-comfyui.sh status --watch 3
spark-comfyui v2026.07.19 — sparky · driver 580.159.03 — every 3s, window 105s — Ctrl-C stops
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
  gen         13.1s                             ███████▁▁▁▁▁▁▁▁▁▁  13.1–29.8 ~17.3 at 10:36:54
  it/s         0.63                         ▁▃▃▄   █▃▃▄    █▃▃     0.4–1.36 ~0.72
  latency       13s                             ███████▅▅▅▅▅▅▅▁▁▁  13–30 ~17.4
  queue           0                     ▁▁▁█████▁▁▁████▁▁▁▁███▁▁▁  0–1 ~0.34
  hit rate      77%                             ▁▁▁▁▁▁▁██████████  0–77 ~57.8
  session: 4 gens · first 29.8s · steady ~13.6s (13.6–13.6) · ~0.63 it/s

  samples: 35 · elapsed: 1m49s
```

The `session:` line aggregates every gen that finished under this watch: `first` carries the model-load cost, `steady` excludes it. Run one watch per condition and compare the two lines. That is exactly how the containerized runtime was validated before it became the only runtime: native steady 13.59s vs container 13.61s on the same Krea-2 workflow and seeds, and every seed-matched output pair bit-identical.

## Content and mounts

All user content lives under `data/` next to the script: `models`, `user` (workflows, settings), `input`, `output`, `custom_nodes`, and optionally `extra_model_paths.yaml`. The container sees it at the standard ComfyUI paths.

To relocate entries or add extra mounts (a NAS share, a second model drive), edit `spark-mounts.conf` next to the script — a commented template is seeded on install:

```
# Per-entry overrides. Relative paths resolve against this file.
models = /mnt/fast-ssd/models
output = /mnt/nas/comfyui-output

# Additional bind mounts, repeatable: HOST:CONTAINER[:ro]. The container
# path must be under /opt/ComfyUI and the host path must already exist.
mount = /mnt/nas/sdxl-models:/opt/ComfyUI/models/nas:ro
```

Point `extra_model_paths.yaml` entries at the CONTAINER side of `mount =` lines. `status` always prints the resolved table, so there is never a question of what is mounted where:

```console
$ ./spark-comfyui.sh status
[...]
== configured mounts (resolved) ==
  models                    /home/user/spark-comfyui/data/models
  user                      /home/user/spark-comfyui/data/user
  input                     /home/user/spark-comfyui/data/input
  output                    /home/user/spark-comfyui/data/output
  custom_nodes              /home/user/spark-comfyui/data/custom_nodes
  extra_model_paths.yaml    /home/user/spark-comfyui/data/extra_model_paths.yaml
  (overrides: /home/user/spark-comfyui/spark-mounts.conf)
```

The whole `data/` directory can be relocated with the `DATA_DIR` env var. A typo in a `mount =` line dies loudly instead of silently shadowing your data with an empty directory.

## Backup, restore, reset

`backup` archives what took human effort: workflows and settings, inputs, config files, and the custom-node set (git nodes as pinned url+commit manifest entries, plain nodes copied whole). Models are manifested (path and size), never archived; a real backup was 3.7M against 74 GB of models. Safe while ComfyUI runs:

```console
$ ./spark-comfyui.sh backup

==> Staging backup

==> Backup written
  /home/user/spark-comfyui/backups/spark-backup-20260720-095245.tgz (1.2M)
  models manifested, not archived — restore lists what to re-download
```

`restore` rebuilds from the archive: builds the image if it is missing, merges content into `data/`, re-clones git custom nodes at their exact pinned commits, then diffs the models manifest against disk and prints exactly which model files are missing and their sizes. Node requirements and the torch guard run in the container entrypoint at next start, so a restore is content-only and cannot break the runtime:

```console
$ ./spark-comfyui.sh restore backups/spark-backup-20260720-095245.tgz

==> Unpacking /home/user/spark-comfyui/backups/spark-backup-20260720-095245.tgz
  format=1
  version=2026.07.19
  date=2026-07-20T09:52:45+02:00
  host=sparky
  comfyui_commit=66655153499f89052aa72d5a869f556b25f0e9c6
  [info] container spark-comfyui is not running

==> Merging user state
  merged user/
  merged input/
  restored comfyui-patches.list

==> Restoring custom nodes
  + testnode (plain copy)
  [info] node requirements and the torch guard run in the container entrypoint at next start

==> Models check (against the archive's manifest)
  all manifested models present

==> Restore complete
  Start ComfyUI:  ./spark-comfyui.sh run
```

Moving to a new Spark: clone spark-comfyui there, copy the backup archive, rsync the models directory, run `restore`. `doctor` reports the newest backup and its age.

`reset` removes the container, every image tag and the cache volume, then rebuilds the image from scratch (including the 10-30 min SageAttention compile). Because content lives outside the image by design, `data/` is never touched — there is nothing to hold aside and nothing that can be lost.

## Migrating from the native version

If you installed spark-comfyui before the container era (a `ComfyUI/` checkout and a `comfyui-env/` venv next to the script), the migration tooling lives in the `v2026.07.20` tag. `install` and `update` detect the old layout and stop with these exact instructions:

```bash
git checkout v2026.07.20
./spark-comfyui.sh migrate --keep-legacy   # content -> data/, instant renames
git checkout main
./spark-comfyui.sh install                 # one-time image build
./spark-comfyui.sh run
```

The `migrate` in that tag can also delete the old checkout, venv and SageAttention tree afterward (run it again without `--keep-legacy`, reclaims ~15-20 GB).

To stay on the last native version instead: `git checkout v2026.07.19`, the last native release.

## Patch list (optional)

Track unmerged upstream PRs or fork branches. Create/edit `comfyui-patches.list` next to the script (a commented template is seeded on install):

```
pr:12345                                             # a ComfyUI pull request
branch:some-origin-branch                            # a branch on origin
remote:https://github.com/user/ComfyUI.git branch    # a fork's branch
```

The list is merged on top of upstream inside the image build, on a `spark-patched` branch. A changed list rebuilds exactly the right layers; a merge conflict fails the build loudly instead of shipping silently unpatched. Empty list (the default) means plain master tracking.

## Troubleshooting

Start with:

```bash
./spark-comfyui.sh doctor
```

Every check prints PASS/FAIL and, on failure, the exact command that fixes it. Common ones:

- **`docker: command not found` or no nvidia runtime**: DGX OS ships both; on other setups install docker plus the NVIDIA Container Toolkit. `doctor` checks both explicitly.
- **An update broke generation**: `./spark-comfyui.sh update --rollback` swaps back to the previous image instantly, then restart.
- **A custom node will not load**: check the start log — the entrypoint installs every node's `requirements.txt` and warns per node instead of dying. A node needing a system library the image lacks is an issue worth filing; the image is the dependency contract.
- **Silent hard-reboot during video generation**: run `status --watch`, reproduce, check the last logged lines. A power spike right before death means overcurrent; fix with `tune --clock-cap 2100`.
- **Whole machine freezes near memory limit**: swap thrash on unified memory. Run `tune` (disables swap; you get a clean OOM kill instead).
- **Manager says an action is not allowed**: lower `security_level` in `data/user/__manager/config.ini` temporarily (`normal-` or `weak`), restore it afterward.
- **Capability warning at startup** (`sm_121 exceeds torch's supported maximum`): expected on GB10, harmless; PTX JIT covers it.
- **Spark rebooted mid-update?** The DGX Dashboard's system-update flow always ends in an automatic reboot. Don't run `spark-comfyui.sh update` in the same window as a system/firmware update.

## Security notes

Containerization is the security model, not a convenience:

- Custom nodes are arbitrary third-party code. In the container they run as a non-root user with all capabilities dropped, `no-new-privileges`, and only your content directories and the GPU visible. A malicious node cannot read your SSH keys, your shell history, or anything else on the host.
- The image is the full dependency set, reproducible from this repo: ComfyUI at a pinned commit, the onnxruntime wheel pinned by sha256, SageAttention pinned to a field-verified commit. `update --rollback` returns to the previous known-good image atomically.
- `network_mode = personal_cloud` relaxes Manager's security gating so it works while serving on `0.0.0.0`. Fine on a trusted LAN. Do not expose the port directly to the internet.
- The GPU onnxruntime wheel and any patch-list forks are community builds, not NVIDIA/Comfy-Org releases. Review what you point the script at.

---

MIT, see [LICENSE](LICENSE). The GB10 knowledge encoded here comes from the NVIDIA DGX Spark developer forums, the [dgx-spark-playbooks](https://github.com/NVIDIA/dgx-spark-playbooks), and the community projects that mapped this hardware in public.
