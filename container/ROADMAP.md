# Container-only spark-comfyui: architecture and roadmap

Status: accepted plan on `container-dev`, 2026-07-20. The next major
version is container-only. The native install path is removed, not
deprecated in place. This document is the reference for that cutover:
what the final shape is, what each native feature maps to, how mounts are
configured, and in what order the work lands.

## Final shape

One script, one image, one data directory.

```
spark-comfyui/
  spark-comfyui.sh          # the only entry point, as today
  mods/                     # unchanged contract; build-time + entrypoint halves
  container/                # Dockerfile, entrypoint.sh, build-mods.sh
  spark-mounts.conf         # mount configuration, seeded template
  comfyui-patches.list      # unchanged, applied at image build
  data/                     # ALL user content, bind-mounted into the container
    models/  user/  input/  output/  custom_nodes/
    extra_model_paths.yaml  # optional, mounted read-only if present
  backups/
```

There is no ComfyUI checkout, no venv, and no SageAttention tree on the
host. Everything reproducible lives in the image. Everything precious
lives in `data/`. The two never mix. `USER_CONTENT` stays the single
definition of precious content; it now names entries under `data/`.

## Mount configuration (the contract)

Resolution order, first match wins, `container status` always shows the
resolved table:

1. `spark-mounts.conf` next to the script. Seeded as a commented template
   on first install, same pattern as `comfyui-patches.list`. Format:

   ```
   # Per-entry overrides. Relative paths resolve against this file.
   # models = /mnt/fast-ssd/models
   # output = /mnt/nas/comfyui-output
   #
   # Additional bind mounts, repeatable. HOST:CONTAINER[:ro]
   # The container side must be under /opt/ComfyUI. Pair extra model
   # locations with entries in data/extra_model_paths.yaml, which sees
   # the CONTAINER paths.
   # mount = /mnt/nas/sdxl-models:/opt/ComfyUI/models/nas:ro
   ```

2. `DATA_DIR` env var: relocates the whole `data/` directory.
3. Default: `data/` next to the script.

Rules: per-entry keys are exactly the `USER_CONTENT` names. Unknown keys
die loudly. Host paths are created if missing (except `mount =` lines,
which must exist; a typo must not silently create an empty dir and hide a
NAS). `extra_model_paths.yaml` is mounted read-only. The models story for
network storage is ComfyUI's own `extra_model_paths.yaml` plus a `mount =`
line; the conf template documents the pairing explicitly.

## Native feature map (parity table)

Every native command and what happens to it:

- `install`: docker + NVIDIA toolkit preflight, image build, seed
  `spark-mounts.conf` and `comfyui-patches.list`, create `data/`. No sudo,
  no apt, no venv. On a legacy tree it stops and points at `migrate`.
- `run`: today's `container run`. Foreground, hardened flags, mounts from
  the resolved table.
- `service`: replaced by docker restart policy. `service` starts the
  container detached with `--restart unless-stopped` (no `--rm`), which
  survives reboots via the docker daemon. `service --disable` removes it.
  The systemd user unit and linger dance are deleted.
- `stop`: docker stop, covers both foreground and service modes.
- `update`: today's `container update` (self-update with resume hook,
  rebuild, `:previous`, restart notice). `--torch` maps to
  `--no-cache-filter` on the named torch stage, forcing fresh cu130
  wheels; Sage rebuilds automatically because its layer depends on the
  torch layer. `--rollback` is the tag swap.
- `doctor`: one merged command. Host sections (self version, update
  probe, backup age, docker daemon, nvidia runtime, driver, image age,
  drift, tune state, swap) plus the four live gates executed in a
  throwaway `--gpus` container (torch with diag, Sage sm_121 kernel, onnx
  provider, kitchen NVFP4), plus `pip check` inside the image.
- `status` and `status --watch`: survives nearly intact. All sampling is
  host-side and container processes are visible in the host process
  table, so pgrep, RSS, and the per-process GPU split keep working; the
  generation telemetry is HTTP against the published port. Adaptations:
  the pid pattern must also match the container venv path
  (`/opt/comfyui-env/bin/python`), and the sage-flag detection reads the
  container's command line. Field-verify every row.
- `tune`: unchanged, host-side by nature (clock caps, swap, persistence).
- `backup`: already venv-free. One change: the ComfyUI commit for `meta`
  comes from an image label (`org.spark-comfyui.comfy-sha`, baked at
  build) instead of a host checkout.
- `restore`: reworked. Ensure image exists (build if missing), restore
  content into the resolved mounts, clone manifest nodes into
  `data/custom_nodes` (git on host, prompt-proofed as today). The
  per-node pip step is deleted; the entrypoint installs requirements on
  every start, which self-heals restores by construction.
- `reset`: massively simplified. Content is outside by design, so reset
  is: remove containers, remove images, remove the cache volume, rebuild
  with `--no-cache`. The hold-dir protocol, the wipe guard, and the
  interrupted-reset resume logic are deleted. `data/` is never touched.
- `rollback` alias: maps to `update --rollback`.
- Mods: contract unchanged. 05 and 10 apply at image build, 20/30/40/50
  semantics live in the entrypoint and the build. `doctor` still reports
  per-mod verify status.

## New command

- `migrate [--keep-legacy]`: one-time, for existing native installs.
  Moves the `USER_CONTENT` entries from the old ComfyUI tree into `data/`
  (same-filesystem rename, instant), then with `--keep-legacy` leaves the
  old checkout, venv and SageAttention tree in place, otherwise deletes
  them (reclaims roughly 15 to 20 GB). Idempotent, resumable, and it
  reuses the no-overwrite rule from the old reset.

## The upgrade cliff (do not skip this)

`update` self-pulls main HEAD and re-execs. The first container-only push
to main therefore lands on every existing native install at its next
`update`. Required behavior: the new script detects a legacy layout
(ComfyUI checkout plus venv, no `data/`) and dies loudly with the
migration instructions instead of proceeding. A `legacy` branch is cut
from the last native release for users who want to stay; the README names
it. This gate ships in the same commit that removes the native path.

## Cutover gate (golden rule 3 applied to the whole initiative)

Before the native path is deleted, one full A/B on the production Spark:
the same workflow, native vs container, fresh launch each, comparing
first-gen, steady-state, and it/s via the `status --watch` session line.
Acceptance: within noise (the 2026-07-13 bench methodology, threshold 3
percent). If the container loses, find out why before cutting over. The
unified-memory mod, pinned-memory flags, and bf16 flags are identical by
construction, so a regression would point at docker overhead or shm, both
diagnosable.

## Phases

- **3a, layout and config**: `data/` layout, `spark-mounts.conf` parsing
  and template, `DATA_DIR`, resolved-table display in `status`, `migrate`.
  The container keeps working from a native-layout checkout during this
  phase (mount resolution falls back to the old paths when `data/` is
  absent).
- **3b, lifecycle parity**: `install` rework, `service` via restart
  policy, `update --torch` (named stages, `--no-cache-filter`),
  patch-list application at image build (COPY into build so a changed
  list busts exactly the right layer), merged `doctor`, reworked
  `restore`, simplified `reset`, image label for backup meta.
- **3c, observability and proof**: `status`/`--watch` adaptations
  field-verified row by row, then the cutover A/B bench on the production
  Spark. Nothing gets deleted before this passes.
- **3d, the cut**: remove the native path, add the legacy-layout gate to
  `update`, cut the `legacy` branch, rewrite README (install is: install
  docker toolkit if absent, clone, `install`, `run`), migration guide,
  major CalVer release with loud notes, forum post.
- **4, later, optional**: multi-stage slimming (measure first; the
  entrypoint needs build tools for source-built node deps, so the win may
  be smaller than it looks), prebuilt GHCR images (supply-chain decision,
  needs arm64 CI and image signing), rootless docker or podman
  evaluation, read-only rootfs (currently rejected: the entrypoint
  installs node requirements into the container filesystem by design).

## Risks, named

1. Container performance regression under unified memory. Mitigated by
   the cutover gate; nothing is deleted until the A/B passes.
2. Existing users hitting the upgrade cliff. Mitigated by the legacy
   gate, the `legacy` branch, and the migration guide.
3. Custom nodes needing system libraries the image lacks (DGX OS has more
   than the image). Policy: add to the image on field report, as with the
   GLSL X libs. The image is the dependency contract now.
4. Entrypoint requirements-reinstall latency with many nodes. Measured
   per start; if it grows painful, the deliberate fix is a venv overlay
   volume, a decision for phase 4, not an accident.
5. Docker daemon becomes a hard dependency. Accepted: DGX OS ships it,
   and the security win is the point of the initiative.
