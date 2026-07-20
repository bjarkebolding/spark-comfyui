#!/usr/bin/env bash
# =============================================================================
#  spark-comfyui.sh — ComfyUI on NVIDIA DGX Spark (GB10 Grace Blackwell)
#  Version 2026.07.20.3 | License: MIT
# =============================================================================
#  Runs ComfyUI in a hardened container tuned for the Spark's aarch64 CPU,
#  sm_121 GPU and 128 GB unified memory. One script for the whole lifecycle;
#  the image holds everything reproducible, your content lives in data/
#  next to this script and is bind-mounted in.
#
#  Commands:
#    install                   One-shot setup: checks docker + the NVIDIA
#                              Container Toolkit, seeds the config templates,
#                              builds the image (ComfyUI, cu130 torch, native
#                              sm_121 SageAttention, GPU onnxruntime, GB10
#                              mods baked in). No venv, no sudo. 10-30 min
#                              the first time, minutes on rebuilds.
#    run [args...]             Start ComfyUI in the container, foreground
#                              (Ctrl-C stops). Extra args pass to main.py.
#                              Every start re-verifies the live GPU kernel
#                              gates and installs custom-node requirements.
#    service [--disable]       Same, detached with a docker restart policy:
#                              survives crashes and reboots. --disable
#                              removes it.
#    stop                      Stop ComfyUI (container, service or a stray
#                              process).
#    update [--torch|--rollback]
#                              Self-update this tool, then rebuild the image
#                              on current ComfyUI master (cached layers make
#                              that minutes, not a full build). The replaced
#                              image stays tagged :previous; --rollback
#                              swaps back to it instantly. --torch forces
#                              fresh cu130 torch wheels (SageAttention
#                              rebuilds on top). Optional: PRs/branches in
#                              comfyui-patches.list (pr:<N> | branch:<name> |
#                              remote:<url> <branch>) are merged on top of
#                              upstream inside the image build.
#    doctor                    Health check: self/update probe, host (driver,
#                              docker runtime, image age, drift, swap,
#                              backups), then the live GPU gates (torch CUDA,
#                              SageAttention sm_121 kernel, GPU onnxruntime,
#                              NVFP4) run inside a throwaway container.
#    status [--watch [SEC]]    One-page glance: process, GPU, memory, image.
#                              --watch shows a live dashboard (sparkline
#                              timeseries: temp/power/clock/util/RAM/CPU,
#                              every 5s or SEC) and appends every sample to
#                              thermal_monitor.log — the evidence trail for
#                              diagnosing silent hard-reboots survives them.
#    tune [--clock-cap MHZ] [--persist]
#                              Host-side stability: disable swap (prevents
#                              unified-memory freezes), persistence mode,
#                              optional clock cap (~2100 fixes overcurrent
#                              hard-reboots). --persist survives reboots.
#    backup [--with-output] [FILE]
#                              Archive the small precious state: workflows,
#                              settings, inputs, patch list, custom-node list,
#                              and a manifest of your models (listed, never
#                              copied — they are huge). --with-output also
#                              archives generated images. Safe while running.
#    restore FILE              Rebuild from a backup archive: builds the
#                              image if needed, merges content into data/,
#                              re-clones custom nodes (their requirements
#                              install at next start), and lists which
#                              models you still need to fetch separately.
#    reset [--yes]             Remove the container, every image tag and the
#                              cache volume, then rebuild from scratch. The
#                              nuclear option; your content (data/) is
#                              never touched.
#
#  Upgrading from a pre-container (native) install: 'install' detects the
#  old layout and prints the move commands (five renames into data/).
#
#  Mounts: data/ holds models, user, input, output, custom_nodes. Override
#  per-entry paths or add extra mounts (e.g. a NAS share) in
#  spark-mounts.conf — a commented template is seeded on install, and
#  'status' always shows the resolved table. Custom-node code runs
#  confined: non-root, no capabilities, nothing mounted but your content.
#
#  Typical day: install once -> run (or service) -> update now and then.
#  Something feels wrong? -> doctor tells you what and how to fix it.
#  Re-running install is safe: completed steps are skipped or refreshed.
# =============================================================================
set -euo pipefail

# Date versioning (CalVer): YYYY.MM.DD, with .N appended for a second
# behavior-changing release on the same day. Bumped in the same push as any
# behavior change (pushing to main IS releasing); docs-only pushes don't bump.
VERSION="2026.07.20.3"

# ----------------------------- Configuration --------------------------------
# Everything is self-contained under the directory this script lives in, so
# you can drop spark-comfyui.sh into any folder and it installs/runs there.
# Resolve the real location even if invoked via a symlink or a relative path.
SELF="$(readlink -f "${BASH_SOURCE[0]}")"
BASE_DIR="${BASE_DIR:-$(dirname "$SELF")}"

# All overridable via environment if you want them elsewhere.
# Pinned, not tracking thu-ml/SageAttention's default branch: 3.x showed
# mosaic artifacts on GB10 (a visual regression the live kernel test can't
# catch — it checks shape/finiteness, not output correctness). This is the
# exact commit field-verified on GB10 sm_121 (38 commits past the v2.2.0
# tag, still pre-3.0). Bump deliberately, not automatically.
SAGE_REF="${SAGE_REF:-d1a57a546c3d395b1ffcbeecc66d81db76f3b4b5}"
REPO_URL="${REPO_URL:-https://github.com/Comfy-Org/ComfyUI.git}"
TORCH_INDEX="${TORCH_INDEX:-https://download.pytorch.org/whl/cu130}"
# Community sm_121/aarch64/cu13 GPU onnxruntime (no official PyPI wheel
# exists). The #sha256= fragment pins the exact bytes: pip verifies it before
# installing, so a compromised or force-pushed hosting repo fails loudly
# instead of installing silently. Overriding ORT_WHEEL_URL replaces the pin
# too — re-add a fragment for your own wheel if you want the same guarantee.
ORT_WHEEL_URL="${ORT_WHEEL_URL:-https://huggingface.co/Jay0515/onnxruntime-gpu-aarch64-cuda13-sm121/resolve/main/onnxruntime_gpu-1.25.0-cp312-cp312-linux_aarch64.whl#sha256=da487cc1ccd3aa11389efec14c6f0f8b6bd7ca6734423de3b528e578023cb200}"
PORT="${PORT:-8188}"

# Network resilience: the wheels involved are huge (torch cu130 is >1 GB,
# the GPU onnxruntime ~220 MB) and a single dropped connection would abort
# a long install. Make pip retry transient failures instead of dying.
export PIP_RETRIES="${PIP_RETRIES:-5}"
export PIP_DEFAULT_TIMEOUT="${PIP_DEFAULT_TIMEOUT:-120}"

# Optional patch list: one entry per line, applied on top of fresh upstream
# master onto a local 'spark-patched' branch on every install/update.
#   pr:9876                          # a ComfyUI GitHub pull request
#   branch:some-upstream-branch      # a branch on origin
#   remote:https://github.com/u/ComfyUI.git their-branch   # a fork's branch
# Lines starting with # and blank lines are ignored.
PATCH_LIST="${PATCH_LIST:-$BASE_DIR/comfyui-patches.list}"

# GB10 mods live in mods/<name>/run.sh and are discovered, applied, and
# verified through a small contract (see mods/README.md). Toggle all mods
# off with SPARK_SOURCE_PATCHES=0.
MODS_DIR="${MODS_DIR:-$BASE_DIR/mods}"

# Containerized runtime (EXPERIMENTAL, `container` subcommands): image and
# container name, both overridable.
CONTAINER_IMAGE="${CONTAINER_IMAGE:-spark-comfyui}"
CONTAINER_NAME="${CONTAINER_NAME:-spark-comfyui}"
# Container-only layout: all user content under one
# data/ directory, mount overrides in spark-mounts.conf next to the script.
DATA_DIR="${DATA_DIR:-$BASE_DIR/data}"
MOUNTS_CONF="${MOUNTS_CONF:-$BASE_DIR/spark-mounts.conf}"

# User content inside the ComfyUI tree: what reset holds aside while it
# wipes and reinstalls everything else. backup/restore cover the same set
# (each entry with its own rules, so they don't loop over this list).
readonly USER_CONTENT=(models user input output custom_nodes extra_model_paths.yaml)

log()  { printf '\n\033[1;32m==> %s\033[0m\n' "$*"; }
warn() { printf '\033[1;33m[warn] %s\033[0m\n' "$*"; }
die()  { printf '\033[1;31m[error] %s\033[0m\n' "$*" >&2; exit 1; }
ok()   { printf '  \033[1;32m[PASS]\033[0m %s\n' "$*"; PASS=$((PASS+1)); }
bad()  { printf '  \033[1;31m[FAIL]\033[0m %s\n' "$*"; FAIL=$((FAIL+1)); }
info() { printf '  \033[1;36m[info]\033[0m %s\n' "$*"; }
hdr()  { printf '\n\033[1m== %s ==\033[0m\n' "$*"; }

# ASCII banner (figlet 'slant', 76 cols) — printed once by the dispatch on
# every invocation except --version, which stays a bare parseable line.
banner() {
  printf '\033[1;32m'
  cat <<'ART'
                          __                              ____            _
   _________  ____ ______/ /__      _________  ____ ___  / __/_  ____  __(_)
  / ___/ __ \/ __ `/ ___/ //_/_____/ ___/ __ \/ __ `__ \/ /_/ / / / / / / /
 (__  ) /_/ / /_/ / /  / ,< /_____/ /__/ /_/ / / / / / / __/ /_/ / /_/ / /
/____/ .___/\__,_/_/  /_/|_|      \___/\____/_/ /_/ /_/_/  \__, /\__,_/_/
    /_/                                                   /____/
ART
  printf '\033[0m  \033[1mv%s\033[0m — ComfyUI on the NVIDIA DGX Spark (GB10 Grace Blackwell)\n' "$VERSION"
}

usage() { awk 'NR>1 && /^#/ {sub(/^# ?/,""); print; next} NR>1 {exit}' "$0"; }

# The GB10 helper library (torch_cuda_diag, sage_kernel_ok, onnx_gpu_ok,
# kitchen_nvfp4_ok, repair_torch, the mod contract helpers). Since the cut
# it runs mostly INSIDE the image (entrypoint, build-mods, the doctor
# gates); the host-side source stays so test harnesses and future host
# checks share one copy.
# shellcheck disable=SC1091
source "$MODS_DIR/_lib/mod_common.sh"

# Seed a self-documenting patch-list template (all comments = empty list).
# The container build COPYs this file into the image and merges its entries
# there (container/build-patches.sh).
seed_patch_list() {
  [[ -f "$PATCH_LIST" ]] && return 0
  cat > "$PATCH_LIST" <<'TPL'
# comfyui-patches.list — PRs/branches merged on top of upstream master on
# every install/update (rebuilt fresh each time on the 'spark-patched'
# branch; master stays pristine). Formats:
#   pr:12345
#   branch:some-origin-branch
#   remote:https://github.com/user/ComfyUI.git their-branch
#
# As of 2026-07: the big Spark PRs (async offload #10953/#11069/#11171/#13221,
# audio VAE #13486) are MERGED upstream — tracking master needs no patches.
#
# Optional experiment — UMA mmap loading + LoRA memory-doubling fix (fork,
# unreviewed; features are opt-in via: run --fast-mmap-load --cuda-uma):
# remote:https://github.com/stardust7700/ComfyUI.git master
TPL
  echo "Seeded patch-list template: $PATCH_LIST (empty — all comments)"
}


install_self() {
  # The script anchors all paths to its own location (BASE_DIR), so wherever
  # it lives IS the install root — no need to copy it elsewhere. Just make
  # sure it's executable so the systemd service and cron can invoke it.
  chmod +x "$SELF" 2>/dev/null || true
}

# Self-update: pull the spark-comfyui repo itself (script + mods) before
# updating anything else, so updater fixes and new mods take effect in this
# very run. Acts ONLY when all of these hold: BASE_DIR is a git clone with
# an upstream branch, the remote is strictly ahead (fast-forwardable), and
# the user hasn't diverged (local commits) — otherwise it warns or silently
# skips and the update continues on the current version. After a successful
# pull it re-execs the fresh script exactly once (SPARK_SELF_UPDATED guard):
# bash reads scripts incrementally, so the running copy must never continue
# executing from a file that just changed underneath it.
# Opt out entirely with SPARK_SELF_UPDATE=0.
self_update() {
  [[ "${SPARK_SELF_UPDATE:-1}" == "1" ]] || return 0
  [[ -z "${SPARK_SELF_UPDATED:-}" ]] || return 0   # already updated this run
  [[ -d "$BASE_DIR/.git" ]] || return 0            # not installed via git clone
  local local_rev upstream_rev
  if ! timeout 30 git -C "$BASE_DIR" fetch -q origin 2>/dev/null; then
    warn "self-update: could not reach the spark-comfyui repo (offline?) —
continuing with the current version"
    return 0
  fi
  local_rev="$(git -C "$BASE_DIR" rev-parse HEAD 2>/dev/null)" || return 0
  upstream_rev="$(git -C "$BASE_DIR" rev-parse '@{u}' 2>/dev/null)" || return 0
  [[ "$local_rev" == "$upstream_rev" ]] && return 0  # already current
  if ! git -C "$BASE_DIR" merge-base --is-ancestor "$local_rev" "$upstream_rev"; then
    warn "spark-comfyui's own repo has local commits that diverge from upstream —
not self-updating. Reconcile manually: git -C $BASE_DIR pull"
    return 0
  fi
  log "Updating spark-comfyui itself: ${local_rev:0:8} -> ${upstream_rev:0:8}"
  if git -C "$BASE_DIR" merge -q --ff-only "$upstream_rev" 2>/dev/null; then
    echo "Re-running update with the new version..."
    # SELF_UPDATE_RESUME lets a caller name the command to re-exec into
    # (cmd_container_update sets "container update"); default is the native
    # update. Deliberately unquoted: the value may be multiple words.
    # shellcheck disable=SC2086
    SPARK_SELF_UPDATED=1 exec "$SELF" ${SELF_UPDATE_RESUME:-update} "$@"
  else
    warn "self-update could not fast-forward (uncommitted local edits in
$BASE_DIR?) — continuing with the current version. To update manually:
  git -C $BASE_DIR pull"
  fi
  return 0
}

# =============================================================================
#  tune — system-level stability & performance (field-validated on GB10)
# =============================================================================
cmd_tune() {
  local clock_cap="" persist=0
  while [[ $# -gt 0 ]]; do
    case "$1" in
      --clock-cap) clock_cap="${2:?--clock-cap needs a MHz value, e.g. 2100}"; shift 2 ;;
      --persist)   persist=1; shift ;;
      *) die "Unknown tune option: $1 (use --clock-cap MHZ and/or --persist)" ;;
    esac
  done
  # Validate up front: a bad value would otherwise die mid-run on nvidia-smi's
  # error, skipping the remaining tune steps. 300 is the -lgc floor used below.
  if [[ -n "$clock_cap" ]]; then
    [[ "$clock_cap" =~ ^[0-9]+$ ]] && (( clock_cap >= 300 && clock_cap <= 4000 )) \
      || die "--clock-cap needs a MHz value between 300 and 4000, e.g. 2100 (got: $clock_cap)"
  fi

  log "Applying DGX Spark system tuning"

  # 1) Swap off. On unified memory, approaching the limit doesn't OOM-kill
  #    cleanly — it thrashes swap, saturates the bus and silently freezes
  #    the whole box. No swap = clean OOM kill instead. Each sudo step below
  #    is skipped if the system's already in the target state, so re-running
  #    tune (e.g. after a reboot without --persist) doesn't ask for a
  #    password it doesn't need.
  if [[ -n "$(swapon --noheadings 2>/dev/null)" ]]; then
    sudo swapoff -a
    echo "  swap disabled"
  else
    echo "  swap already disabled"
  fi
  if [[ "$(sysctl -n vm.swappiness 2>/dev/null)" != "10" ]]; then
    sudo sysctl -w vm.swappiness=10 >/dev/null
  fi
  if grep -q '^vm.swappiness=10' /etc/sysctl.conf 2>/dev/null; then
    echo "  vm.swappiness=10 (already set in /etc/sysctl.conf)"
  else
    echo 'vm.swappiness=10' | sudo tee -a /etc/sysctl.conf >/dev/null
    echo "  vm.swappiness=10 (written to /etc/sysctl.conf — survives reboot on its own)"
  fi

  # 2) Driver persistence mode: avoids re-init latency between runs.
  if [[ "$(nvidia-smi --query-gpu=persistence_mode --format=csv,noheader 2>/dev/null)" == "Enabled" ]]; then
    echo "  persistence mode: already on"
  else
    sudo nvidia-smi -pm 1 >/dev/null && echo "  persistence mode: on"
  fi

  # 3) Optional clock cap. GB10 idles ~14W and spikes to ~85W instantly when
  #    denoising starts; on some units that trips overcurrent protection ->
  #    instant hard reboot with NO logs. Capping to ~2100 MHz holds ~50W and
  #    eliminated the crashes in community testing (power limit via
  #    'nvidia-smi -pl' is N/A on GB10, so clocks are the only lever).
  if [[ -n "$clock_cap" ]]; then
    sudo nvidia-smi -lgc 300,"$clock_cap" \
      && echo "  GPU clocks locked to 300-${clock_cap} MHz"
  else
    echo "  (no clock cap requested — if you get silent hard reboots under"
    echo "   heavy video generation, re-run: $0 tune --clock-cap 2100)"
  fi

  # 4) Optionally persist across reboots (swapoff and -lgc do not survive).
  if [[ "$persist" -eq 1 ]]; then
    local unit=/etc/systemd/system/comfyui-tune.service lgc_line=""
    if [[ -n "$clock_cap" ]]; then
      lgc_line="ExecStart=/usr/bin/nvidia-smi -lgc 300,$clock_cap"
    elif [[ -f "$unit" ]]; then
      # Rewriting the unit without --clock-cap must not silently drop a
      # previously persisted cap: losing it re-exposes the overcurrent
      # hard-reboot the cap exists to prevent. Carry the old line over.
      lgc_line="$(grep -m1 '^ExecStart=.*nvidia-smi -lgc' "$unit" 2>/dev/null || true)"
      [[ -n "$lgc_line" ]] \
        && echo "  keeping previously persisted clock cap (${lgc_line##*-lgc })"
    fi
    sudo tee "$unit" >/dev/null <<UNIT
[Unit]
Description=DGX Spark tuning for ComfyUI (swap off, persistence mode, clocks)
After=multi-user.target

[Service]
Type=oneshot
ExecStart=/usr/sbin/swapoff -a
ExecStart=/usr/bin/nvidia-smi -pm 1
$lgc_line

[Install]
WantedBy=multi-user.target
UNIT
    sudo systemctl daemon-reload
    sudo systemctl enable comfyui-tune.service >/dev/null
    echo "  tuning persisted across reboots (comfyui-tune.service)"
  else
    warn "the swap-off and clock-cap settings do NOT survive a reboot on their
own (vm.swappiness above already does) — add --persist to make all of it permanent"
  fi
}

# =============================================================================
#  status — one-page health overview
# =============================================================================
# One row of the --watch dashboard: label, current value+unit with a trend
# arrow, a sparkline of the sample window, and the window's min–max and mean.
# The series is a space-joined list of numbers in which "-" marks a missed
# sample. When warn/crit thresholds are given (absolute values, "" = off),
# every glyph — and the current reading — is heat-colored green/yellow/red by
# its own value; unthresholded rows render in a single accent color. Colors
# wrap the *padded* value text so escape bytes never skew the column widths.
# The bar glyphs are split into an array (not indexed with substr) so this
# renders identically under gawk (char-based substr) and mawk (byte-based).
_watch_row() {
  awk -v label="$1" -v cur="$2" -v unit="$3" -v s="$4" \
      -v warn="${5:-}" -v crit="${6:-}" -v extra="${7:-}" '
  function heat(x) {
    if (warn == "") return "\033[36m"
    if (crit != "" && x + 0 >= crit + 0) return "\033[31m"
    if (x + 0 >= warn + 0) return "\033[33m"
    return "\033[32m"
  }
  BEGIN {
    split("▁ ▂ ▃ ▄ ▅ ▆ ▇ █", bar, " ")
    d = "\033[2m"; b = "\033[1m"; r = "\033[0m"
    n = split(s, v, " "); lo = 1e30; hi = -1e30; sum = 0; cnt = 0
    prev = ""; last = ""
    for (i = 1; i <= n; i++) if (v[i] != "-") {
      x = v[i] + 0
      if (x < lo) lo = x
      if (x > hi) hi = x
      sum += x; cnt++
      prev = last; last = x
    }
    out = ""; pc = ""
    for (i = 1; i <= n; i++) {
      if (v[i] == "-") { out = out " "; continue }
      x = v[i] + 0
      lvl = (hi > lo) ? int((x - lo) / (hi - lo) * 7 + 0.5) : 4
      c = heat(x)
      if (c != pc) { out = out c; pc = c }
      out = out bar[lvl + 1]
    }
    if (pc != "") out = out r
    # Trend arrow: last sample vs the one before, dead-banded to 5% of the
    # window span so steady-state noise does not flicker.
    arrow = " "
    if (prev != "" && hi > lo) {
      if (last - prev > (hi - lo) * 0.05) arrow = "↗"
      else if (prev - last > (hi - lo) * 0.05) arrow = "↘"
    }
    if (cur == "-") { curtxt = sprintf("%9s", "n/a"); curcol = d }
    else { curtxt = sprintf("%9s", cur unit); curcol = b heat(cur) }
    stats = ""
    if (cnt > 0) stats = sprintf("%.4g–%.4g ~%.4g", lo, hi, sum / cnt)
    if (extra != "") stats = stats " " extra
    printf "  %-8s%s%s%s %s%s%s %s  %s%s%s", \
      label, curcol, curtxt, r, d, arrow, r, out, d, stats, r
  }'
}

# Row-visibility tests for the sample series (passed as "${h_x[@]}"; "$*"
# joins to a scalar first — ${arr[*]//pat/} substitutes per-element and
# re-adds the joining spaces, so it can never yield an empty string).
# _series_nonzero: any real nonzero sample. _series_any: any sample at all.
_series_nonzero() { local j="$*"; [[ -n "${j//[ .0-]/}" ]]; }
_series_any()     { local j="$*"; [[ -n "${j//[ -]/}" ]]; }

# A dim "─ NAME ────…" section rule, sized to the dashboard width.
# (sed, not tr: tr is byte-oriented and would shred the multibyte ─ glyph.)
_watch_hdr() {
  local name="$1" width="$2" fill n
  n=$(( width - ${#name} - 5 )); (( n < 1 )) && n=1
  fill="$(printf '%*s' "$n" '' | sed 's/ /─/g')"
  printf '  \033[2m─ %s %s\033[0m' "$name" "$fill"
}

# Generation telemetry, straight from a running ComfyUI's own HTTP API
# (all stock endpoints — /history, /queue, /internal/logs/raw):
#   - duration of the newest *finished* prompt (an errored one says so),
#     elapsed time of any in-flight one (in-flight = execution_start with no
#     terminal message yet, or a non-empty running queue)
#   - that prompt's node-cache hit rate: nodes listed in its
#     execution_cached message vs the prompt's total node count — high means
#     repeat submissions properly reuse static inputs (prompt embeds, VAE…)
#   - live sampling speed: the newest tqdm "N.NNit/s" (or s/it, inverted) in
#     the server's terminal ring buffer, only while a prompt is in flight
#   - queue depth + the running/pending prompt ids (the watch loop timestamps
#     ids on first sight to measure true submission→saved latency)
# Prints nine tab-separated fields:
#   <last gen s|-> <in-flight s|-> <ok|err|-> <finished HH:MM:SS|->
#   <finished id|-> <cache hit %|-> <queue depth|-> <it/s|-> <qids csv|(empty)>
_watch_comfy() {
  python3 -c '
import json, re, sys, time, urllib.request
port = sys.argv[1]
def get(path):
    try:
        with urllib.request.urlopen("http://127.0.0.1:%s%s" % (port, path), timeout=1) as r:
            return json.loads(r.read())
    except Exception:
        return None
TERMINAL = ("execution_success", "execution_error", "execution_interrupted")
hist = get("/history?max_items=8") or {}
last = None
run_start = None
for hid, item in hist.items():
    ts = {}
    cached = None
    for name, payload in item.get("status", {}).get("messages", []):
        if isinstance(payload, dict):
            if "timestamp" in payload:
                ts[name] = payload["timestamp"]
            if name == "execution_cached":
                cached = payload.get("nodes")
    start = ts.get("execution_start")
    if start is None:
        continue
    ends = [ts[k] for k in TERMINAL if k in ts]
    if ends:
        if last is None or max(ends) > last[1]:
            try:
                total = len(item["prompt"][2])
            except Exception:
                total = 0
            pct = 100.0 * len(cached) / total if cached is not None and total else None
            last = (start, max(ends), "execution_success" in ts, hid, pct)
    elif run_start is None or start > run_start:
        run_start = start
q = get("/queue") or {}
run = [x[1] for x in q.get("queue_running", []) if isinstance(x, list) and len(x) > 1]
pend = [x[1] for x in q.get("queue_pending", []) if isinstance(x, list) and len(x) > 1]
depth = len(run) + len(pend) if q else "-"
gen = act = flag = fin = fid = cache = its = "-"
if last:
    gen = "%.1f" % ((last[1] - last[0]) / 1000.0)
    flag = "ok" if last[2] else "err"
    fin = time.strftime("%H:%M:%S", time.localtime(last[1] / 1000.0))
    fid = last[3]
    if last[4] is not None:
        cache = "%.0f" % last[4]
if run_start is not None:
    act = "%.0f" % (time.time() - run_start / 1000.0)
elif run:
    act = "0"
if act != "-":
    logs = get("/internal/logs/raw") or {}
    txt = "".join(e.get("m", "") for e in logs.get("entries", [])[-60:])
    hits = re.findall(r"([0-9.]+)\s*(it/s|s/it)", txt)
    if hits:
        v, u = hits[-1]
        try:
            its = "%.2f" % (float(v) if u == "it/s" else (1.0 / float(v)))
        except (ValueError, ZeroDivisionError):
            pass
print("\t".join(str(x) for x in (gen, act, flag, fin, fid, cache, depth, its, ",".join(run + pend))))
' "$1" 2>/dev/null
}

cmd_status() {
  # --watch: live dashboard (sparkline timeseries per metric) + an append-only
  # log line per sample. The LOG is the actual point: it survives the silent
  # hard-reboots this exists to diagnose (a power spike as the final sample
  # = overcurrent -> fix with: tune --clock-cap 2100). The dashboard is just
  # the live view of the same samples; when stdout isn't a terminal (redirect,
  # journal), it drops away and plain log lines are emitted instead.
  if [[ "${1:-}" == "--watch" || "${1:-}" == "-w" ]]; then
    local interval=5 logfile="$BASE_DIR/thermal_monitor.log"
    [[ "${2:-}" =~ ^[1-9][0-9]*$ ]] && interval="$2"
    # Rotate once per watch start when the log outgrows 50 MB (an always-on
    # watch appends ~4 MB/day at the 5s default, more at 1s): the previous
    # evidence survives in .1, and the box never fills up from telemetry.
    if [[ -f "$logfile" ]] \
       && (( "$(stat -c %s "$logfile" 2>/dev/null || echo 0)" > 52428800 )); then
      mv -f "$logfile" "$logfile.1"
      echo "rotated: previous $(basename "$logfile") -> $(basename "$logfile").1"
    fi
    local i tick=0 cols win width
    # Sparkline window adapts to the terminal (label+value+stats ≈ 44 cols).
    cols="$(tput cols 2>/dev/null || echo "${COLUMNS:-100}")"
    win=$(( cols - 44 )); (( win < 24 )) && win=24; (( win > 64 )) && win=64
    width=$(( win + 42 ))
    local -a h_temp h_pwr h_clk h_util h_thr h_ram h_swap
    local -a h_cpu h_io h_gen h_its h_lat h_q h_hit
    # Pre-fill the ring buffers so the sparkline width is constant from tick 1.
    for ((i = 0; i < win; i++)); do
      h_temp[i]='-' h_pwr[i]='-' h_clk[i]='-' h_util[i]='-'
      h_thr[i]='-' h_ram[i]='-' h_swap[i]='-' h_cpu[i]='-'
      h_io[i]='-' h_gen[i]='-' h_its[i]='-' h_lat[i]='-' h_q[i]='-' h_hit[i]='-'
    done
    local gpu_csv t p c u pst thr evt evt_val mem_used mem_tot mem_cache
    local swap_used swap_tot cpu_pct load1 io_now io_rate now_ts prev_io='' prev_ts=''
    local prev_idle=0 prev_tot=0 pid cand rss gen_hdr
    local gself gother state_bad
    local -a lines
    # Session A/B aggregates (see the fin_id accounting in the loop): g_durs
    # holds the duration of every successful gen finished under this watch,
    # in order; the render distills them into the "session:" summary line.
    local g_base='' g_init=0 g_err=0 gsum
    local -a g_durs=() its_hist=()
    local gen_s act_s gen_flag gen_fin gen_extra fin_id cache_hit qdepth its qids qid
    # Submission→saved latency: prompt ids are timestamped when first seen in
    # the queue; when one finishes, latency = now - first-seen. Only covers
    # jobs submitted while the watch is running (resolution = one interval).
    local -A seen=()
    local -a qarr
    local last_lat='-' lat_done=''
    local BLD=$'\033[1m' DIM=$'\033[2m' RED=$'\033[31m' RST=$'\033[0m'
    # Probe once for the extended GPU fields (pstate, clock-event flags).
    # Older drivers that reject them fall back to the minimal set — the
    # P-state tag on the sm-clk row and the throttle row then never render.
    # (utilization.memory was evaluated and dropped: the counter reads a
    # constant 0 on GB10 even at 90W full load.)
    local qgpu="temperature.gpu,power.draw,clocks.sm,utilization.gpu"
    if nvidia-smi --query-gpu="$qgpu,pstate,clocks_event_reasons.active" \
         --format=csv,noheader,nounits >/dev/null 2>&1; then
      qgpu="$qgpu,pstate,clocks_event_reasons.active"
    fi
    local drv; drv="$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -1)"
    # Unified-memory heat thresholds: yellow at 85% of the pool, red at 95%.
    local mem_warn mem_crit
    read -r mem_warn mem_crit <<<"$(awk '/^MemTotal:/ {printf "%.0f %.0f", $2*0.85/1048576, $2*0.95/1048576; exit}' /proc/meminfo)"
    local tty=0
    if [[ -t 1 ]]; then tty=1; printf '\033[2J\033[?25l'; trap 'printf "\033[?25h\n"' EXIT
    else log "Logging every ${interval}s to $logfile (Ctrl-C to stop)"; fi
    while true; do
      # ---- sample everything once per tick --------------------------------
      gpu_csv="$(nvidia-smi --query-gpu="$qgpu" \
                   --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -d ' ')"
      IFS=',' read -r t p c u pst evt <<<"$gpu_csv"
      [[ "${t:-}"   =~ ^[0-9.]+$ ]] || t='-'
      [[ "${p:-}"   =~ ^[0-9.]+$ ]] || p='-'
      [[ "${c:-}"   =~ ^[0-9.]+$ ]] || c='-'
      [[ "${u:-}"   =~ ^[0-9.]+$ ]] || u='-'
      [[ "${pst:-}" =~ ^P[0-9]+$ ]] || pst='-'
      evt_val=0
      [[ "${evt:-}" =~ ^0x[0-9a-fA-F]+$ ]] && evt_val=$(( 16#${evt#0x} )) || evt='-'
      # Serious clock-event flags only. sw-power-cap (0x4) is benign/constant
      # on GB10 even at idle, so it stays in the log's raw EVT hex but never
      # in the dashboard; the slowdown bits are the overcurrent/thermal story.
      state_bad=''
      (( evt_val & 0x08 ))  && state_bad+=' HW-SLOWDOWN'
      (( evt_val & 0x20 ))  && state_bad+=' SW-THERMAL'
      (( evt_val & 0x40 ))  && state_bad+=' HW-THERMAL'
      (( evt_val & 0x80 ))  && state_bad+=' HW-POWER-BRAKE'
      # Count of serious slowdown bits (0 = healthy; any spike is forensic gold)
      thr='-'; [[ "$evt" != '-' ]] && thr="$(wc -w <<<"$state_bad")"
      read -r mem_used mem_tot mem_cache swap_used swap_tot <<<"$(awk '
        /^MemTotal:/ {mt=$2} /^MemAvailable:/ {ma=$2}
        /^Buffers:/ {bu=$2} /^Cached:/ {ca=$2}
        /^SwapTotal:/ {st=$2} /^SwapFree:/ {sf=$2}
        END {printf "%.1f %.0f %.1f %.1f %.0f", (mt-ma)/1048576, mt/1048576, (bu+ca)/1048576, (st-sf)/1048576, st/1048576}
        ' /proc/meminfo)"
      # Whole-box CPU% from the /proc/stat delta (first tick has no delta yet)
      read -r cpu_pct prev_idle prev_tot <<<"$(awk -v pi="$prev_idle" -v pt="$prev_tot" '
        /^cpu / {idle = $5 + $6; tot = 0
                 for (f = 2; f <= NF; f++) tot += $f
                 pct = (pt > 0 && tot > pt) ? 100 * (1 - (idle - pi) / (tot - pt)) : -1
                 printf "%.0f %d %d", pct, idle, tot; exit}' /proc/stat)"
      [[ "$cpu_pct" == "-1" ]] && cpu_pct='-'
      read -r load1 _ </proc/loadavg
      # Disk throughput (MB/s, read+write) over real block devices, from the
      # /proc/diskstats sector-count delta — model loads show up here.
      io_now="$(awk '$3 ~ /^(nvme[0-9]+n[0-9]+|sd[a-z]+)$/ {s += ($6 + $10) * 512}
                     END {printf "%.0f", s}' /proc/diskstats)"
      now_ts="${EPOCHREALTIME/,/.}"   # decimal separator is locale-dependent
      io_rate="$(awk -v a="$prev_io" -v b="$io_now" -v t0="$prev_ts" -v t1="$now_ts" \
        'BEGIN { if (a == "" || t1 - t0 <= 0) { print "-"; exit }
                 printf "%.1f", (b - a) / (t1 - t0) / 1048576 }')"
      prev_io="$io_now"; prev_ts="$now_ts"
      # Prefer the actual python server over wrapper shells whose command
      # line merely contains the launch string (bash -c, script(1), sudo…).
      pid=""
      for cand in $(pgrep -f 'main.py --listen' 2>/dev/null); do
        pid="${pid:-$cand}"   # fallback: first match
        if [[ "$(cat "/proc/$cand/comm" 2>/dev/null)" == python* ]]; then
          pid="$cand"; break
        fi
      done
      rss='-'
      if [[ -n "$pid" ]]; then
        rss="$(awk '/^VmRSS:/ {printf "%.1f", $2/1048576}' "/proc/$pid/status" 2>/dev/null || true)"
        [[ -n "$rss" ]] || rss='-'
        # The attention backend is the classic A/B dimension, so it lives in
        # the section header as run context.
        if pgrep -af 'main.py --listen' 2>/dev/null | grep -q 'use-sage-attention'
        then gen_hdr="GENERATION · SageAttention"
        else gen_hdr="GENERATION · SDPA"; fi
      else
        gen_hdr="GENERATION · ComfyUI not running"
      fi
      # Who holds the unified pool: ComfyUI's own CUDA allocation vs the sum
      # of every co-resident process (vLLM…) — the latter explains "why is
      # generation suddenly offloading". Log-only since 2026.07.15.1 (CGPU/
      # OGPU fields); no dashboard row.
      gself='-' gother='-'
      IFS=$'\t' read -r gself gother <<<"$(nvidia-smi \
        --query-compute-apps=pid,used_memory \
        --format=csv,noheader,nounits 2>/dev/null | awk -F', *' -v comfy="${pid:-x}" '
          { gb = $2 / 1024
            if ($1 == comfy) self += gb; else other += gb }
          END { printf "%.1f\t%.1f", self, other }')"
      [[ -n "$gself" ]]  || gself='-'
      [[ -n "$gother" ]] || gother='-'
      # No pid -> "ComfyUI's own allocation" is not a number, it's a non-fact
      # (the awk above still prints 0.0 with zero compute apps resident).
      [[ -n "$pid" ]] || gself='-'
      # Generation telemetry from ComfyUI's own API (see _watch_comfy). A gen
      # in flight at the moment of a silent reboot is the smoking gun the log
      # exists for, so the numbers go into the evidence line too.
      gen_s='-' act_s='-' gen_flag='-' gen_fin='-' fin_id='-' cache_hit='-' qdepth='-' its='-' qids=''
      [[ -n "$pid" ]] && IFS=$'\t' read -r gen_s act_s gen_flag gen_fin fin_id cache_hit qdepth its qids \
        <<<"$(_watch_comfy "$PORT")"
      [[ -n "$gen_s" ]]     || gen_s='-'
      [[ -n "$act_s" ]]     || act_s='-'
      [[ -n "$gen_flag" ]]  || gen_flag='-'
      [[ -n "$gen_fin" ]]   || gen_fin='-'
      [[ -n "$fin_id" ]]    || fin_id='-'
      [[ -n "$cache_hit" ]] || cache_hit='-'
      [[ -n "$qdepth" ]]    || qdepth='-'
      [[ -n "$its" ]]       || its='-'
      gen_extra=''
      [[ "$gen_fin" != '-' ]]    && gen_extra="at $gen_fin"
      [[ "$gen_flag" == 'err' ]] && gen_extra+=" ${RED}ERROR${RST}"
      [[ "$act_s" != '-' ]]      && gen_extra+="${gen_extra:+ · }now ${act_s}s…"
      # Session A/B aggregates: every gen that *finishes while the watch is
      # running* is recorded exactly once — whatever fin_id says on the first
      # tick is only a baseline, so history predating this watch never skews
      # a bench run. One watch session per A/B condition = one summary line.
      if (( ! g_init )); then
        g_base="$fin_id"; g_init=1
      elif [[ "$fin_id" != '-' && "$fin_id" != "$g_base" ]]; then
        g_base="$fin_id"
        if [[ "$gen_flag" == 'err' ]]; then (( g_err++ )) || true
        elif [[ "$gen_s" != '-' ]]; then g_durs+=("$gen_s"); fi
      fi
      [[ "$its" != '-' ]] && its_hist+=("$its")
      # Latency check MUST run before the populate/prune below: on the very
      # tick a gen finishes the queue is already empty and idle, and pruning
      # first would wipe the finished id's first-seen timestamp.
      if [[ "$fin_id" != '-' && "$fin_id" != "$lat_done" && -n "${seen[$fin_id]:-}" ]]; then
        last_lat=$(( EPOCHSECONDS - seen[$fin_id] ))
        lat_done="$fin_id"
        unset "seen[$fin_id]"
      fi
      if [[ -n "$qids" ]]; then
        IFS=',' read -ra qarr <<<"$qids"
        for qid in "${qarr[@]}"; do
          [[ -n "${seen[$qid]:-}" ]] || seen[$qid]="$EPOCHSECONDS"
        done
      elif [[ "$act_s" == '-' ]]; then
        seen=()   # queue drained and idle: forget cancelled/stale ids
      fi
      h_temp=("${h_temp[@]:1}" "$t");        h_pwr=("${h_pwr[@]:1}" "$p")
      h_clk=("${h_clk[@]:1}" "$c");          h_util=("${h_util[@]:1}" "$u")
      h_thr=("${h_thr[@]:1}" "$thr");        h_ram=("${h_ram[@]:1}" "$mem_used")
      h_swap=("${h_swap[@]:1}" "$swap_used"); h_cpu=("${h_cpu[@]:1}" "$cpu_pct")
      h_io=("${h_io[@]:1}" "$io_rate");      h_gen=("${h_gen[@]:1}" "$gen_s")
      h_its=("${h_its[@]:1}" "$its");        h_lat=("${h_lat[@]:1}" "$last_lat")
      h_q=("${h_q[@]:1}" "$qdepth");         h_hit=("${h_hit[@]:1}" "$cache_hit")

      # ---- durable evidence line (always) ----------------------------------
      printf '%(%F %T)T GPU=%s°C PWR=%sW SM=%sMHz UTIL=%s%% RAM=%s/%sG SWAP=%sG CPU=%s%% CACHE=%sG LOAD=%s IO=%sMB/s RSS=%sG CGPU=%sG OGPU=%sG PST=%s EVT=%s GEN=%ss ACT=%ss ITS=%s LAT=%ss Q=%s HIT=%s%%\n' \
        -1 "$t" "$p" "$c" "$u" "$mem_used" "$mem_tot" "$swap_used" "$cpu_pct" \
        "$mem_cache" "$load1" "$io_rate" "$rss" "$gself" "$gother" "$pst" "$evt" \
        "$gen_s" "$act_s" "$its" "$last_lat" "$qdepth" "$cache_hit" >>"$logfile"

      # ---- live view --------------------------------------------------------
      # Quiet-when-healthy: rows whose healthy state is a flat line of zeros
      # or n/a earn their spot only when they have a story — throttle after
      # any slowdown bit in the window, swap only when it exists at all,
      # gen telemetry only with data. Ring buffers always advance, so a row
      # appears with its window history intact. The log line is NOT
      # conditional — it always carries every field (incl. RSS and the
      # per-process GPU split, which have no dashboard row anymore).
      if (( tty )); then
        (( tick++ )) || true
        # The A/B line: first gen carries the model load, steady excludes it
        # — compare "steady" between two watch sessions run under different
        # conditions (flags, clock caps, co-resident load…).
        gsum=''
        if (( ${#g_durs[@]} + g_err )); then
          gsum="$(awk -v d="${g_durs[*]}" -v e="$g_err" -v its="${its_hist[*]}" 'BEGIN {
            n = split(d, v, " ")
            out = sprintf("session: %d gen%s", n + e, (n + e == 1) ? "" : "s")
            if (n >= 1) out = out sprintf(" · first %.1fs", v[1])
            if (n >= 2) {
              sum = 0; lo = 1e30; hi = -1e30
              for (i = 2; i <= n; i++) {
                sum += v[i]
                if (v[i] < lo) lo = v[i]
                if (v[i] > hi) hi = v[i]
              }
              out = out sprintf(" · steady ~%.1fs (%.4g–%.4g)", sum / (n - 1), lo, hi)
            }
            m = split(its, w, " ")
            if (m) { s = 0; for (i = 1; i <= m; i++) s += w[i]
                     out = out sprintf(" · ~%.2f it/s", s / m) }
            if (e) out = out sprintf(" · %d errored", e)
            print out }')"
        fi
        lines=(
          "${BLD}spark-comfyui v$VERSION${RST} — $(hostname)${drv:+ · driver $drv} — every ${interval}s, window $((win * interval))s — Ctrl-C stops"
          "${DIM}log: $logfile${RST}"
          ""
          "$(_watch_hdr "GPU" "$width")"
          "$(_watch_row "temp"   "$t" "°C"  "${h_temp[*]}" 70 80)"
          "$(_watch_row "power"  "$p" "W"   "${h_pwr[*]}"  60 80)"
          "$(_watch_row "sm clk" "$c" "MHz" "${h_clk[*]}"  "" "" "${pst/#-/}")"
          "$(_watch_row "gpu"    "$u" "%"   "${h_util[*]}")"
        )
        _series_nonzero "${h_thr[@]}" && lines+=(
          "$(_watch_row "throttle" "$thr" "" "${h_thr[*]}" 1 1 "${state_bad# }")")
        lines+=(
          "$(_watch_hdr "SYSTEM" "$width")"
          "$(_watch_row "unified" "$mem_used" "G" "${h_ram[*]}" "$mem_warn" "$mem_crit" "of ${mem_tot}G")"
        )
        [[ "$swap_tot" != "0" ]] && lines+=(
          "$(_watch_row "swap" "$swap_used" "G" "${h_swap[*]}" 0.1 1 "of ${swap_tot}G ${RED}ENABLED — run tune!${RST}")")
        lines+=(
          "$(_watch_row "cpu"     "$cpu_pct" "%"    "${h_cpu[*]}" 85 95)"
          "$(_watch_row "disk io" "$io_rate" "MB/s" "${h_io[*]}")"
          "$(_watch_hdr "$gen_hdr" "$width")"
        )
        { _series_any "${h_gen[@]}" || [[ "$act_s" != '-' ]]; } && lines+=(
          "$(_watch_row "gen" "$gen_s" "s" "${h_gen[*]}" "" "" "$gen_extra")")
        _series_any "${h_its[@]}" && lines+=(
          "$(_watch_row "it/s" "$its" "" "${h_its[*]}")")
        _series_any "${h_lat[@]}" && lines+=(
          "$(_watch_row "latency" "$last_lat" "s" "${h_lat[*]}")")
        _series_nonzero "${h_q[@]}" && lines+=(
          "$(_watch_row "queue" "$qdepth" "" "${h_q[*]}")")
        _series_any "${h_hit[@]}" && lines+=(
          "$(_watch_row "hit rate" "$cache_hit" "%" "${h_hit[*]}")")
        [[ -n "$gsum" ]] && lines+=("  ${DIM}${gsum}${RST}")
        lines+=(
          ""
          "  ${DIM}samples: $tick · elapsed: $((SECONDS / 60))m$(printf '%02d' $((SECONDS % 60)))s${RST}"
        )
        printf '\033[H'
        printf '%s\033[K\n' "${lines[@]}"
        printf '\033[J'
      else
        tail -1 "$logfile"
      fi
      sleep "$interval"
    done
  fi

  hdr "Process"
  if pgrep -f "main.py --listen" >/dev/null 2>&1; then
    local pid where=""
    pid="$(pgrep -f 'main.py --listen' | head -1)"
    # Container processes are visible in the host process table, so the
    # same pgrep finds both worlds; the tag says which one this is.
    command -v docker >/dev/null 2>&1 \
      && [[ -n "$(docker ps -q -f "name=^${CONTAINER_NAME}$" 2>/dev/null)" ]] \
      && where=", containerized"
    echo "  ComfyUI RUNNING (pid $pid$where) -> http://$(hostname -I 2>/dev/null | awk '{print $1}'):$PORT"
    pgrep -af "main.py --listen" | grep -q "use-sage-attention" \
      && echo "  attention: SageAttention" || echo "  attention: PyTorch SDPA"
  else
    echo "  ComfyUI not running (start: $0 run)"
  fi
  systemctl --user is-active comfyui.service >/dev/null 2>&1 \
    && echo "  systemd service: active" || true

  hdr "GPU"
  nvidia-smi --query-gpu=name,temperature.gpu,power.draw,clocks.sm,memory.used,memory.total \
    --format=csv,noheader 2>/dev/null | sed 's/^/  /' || echo "  nvidia-smi unavailable"

  hdr "System"
  free -g | awk '/Mem:/  {printf "  unified memory: %s/%s GB used\n", $3, $2}
                 /Swap:/ {printf "  swap: %s GB total %s\n", $2, ($2==0 ? "(disabled — good)" : "(ENABLED — run tune!)")}'

  hdr "Versions"
  echo "  spark-comfyui: $VERSION"
  if command -v docker >/dev/null 2>&1 \
       && docker image inspect "$CONTAINER_IMAGE:latest" >/dev/null 2>&1; then
    # The image label carries the commit (no host checkout exists).
    local csha built
    csha="$(docker image inspect -f '{{index .Config.Labels "org.spark-comfyui.comfy-sha"}}' "$CONTAINER_IMAGE:latest" 2>/dev/null || true)"
    built="$(docker image inspect -f '{{.Created}}' "$CONTAINER_IMAGE:latest" | cut -dT -f1)"
    echo "  ComfyUI: ${csha:0:12} (image $CONTAINER_IMAGE:latest, built $built)"
  else
    echo "  ComfyUI: no image built yet (run: $0 install)"
  fi
  if [[ -f "$PATCH_LIST" ]] && grep -qE '^[^#[:space:]]' "$PATCH_LIST"; then
    echo "  patch list: $(grep -cE '^[^#[:space:]]' "$PATCH_LIST") entries in $PATCH_LIST"
  fi
  # The Manager config lives under the resolved user mount.
  resolve_mounts
  local cfg="" ci
  for ci in "${!RESOLVED_ENTRIES[@]}"; do
    [[ "${RESOLVED_ENTRIES[$ci]}" == user ]] && cfg="${RESOLVED_PATHS[$ci]}/__manager/config.ini"
  done
  [[ -f "$cfg" ]] && grep -q 'network_mode *= *personal_cloud' "$cfg" \
    && echo "  Manager: network_mode = personal_cloud" \
    || echo "  Manager: personal_cloud NOT set"
}

# =============================================================================
#  stop
# =============================================================================
cmd_stop() {
  if command -v docker >/dev/null 2>&1 \
       && [[ -n "$(docker ps -q -f "name=^${CONTAINER_NAME}$" 2>/dev/null)" ]]; then
    # docker stop, never pkill — the container python matches the pkill
    # pattern below (host-visible process), and killing pid 1 of a
    # service-mode container would only make the restart policy relaunch
    # it.
    docker stop "$CONTAINER_NAME" >/dev/null
    echo "container stopped"
  elif pgrep -f "main.py --listen" >/dev/null 2>&1; then
    pkill -f "main.py --listen"
    echo "ComfyUI process stopped"
  else
    echo "ComfyUI is not running"
  fi
}

# =============================================================================
#  backup / restore — the small precious state, without the models
# =============================================================================
# The archive holds what took human effort (workflows, settings, inputs, the
# custom-node set, config files) plus a manifest of the models — never the
# model files themselves (74+ GB; restore prints what to re-download instead).
# Safe to run while ComfyUI is serving.
cmd_backup() {
  local with_output=0 out=""
  while [[ $# -gt 0 ]]; do
    case "$1" in
      --with-output) with_output=1; shift ;;
      -*) die "Unknown backup option: $1 (use --with-output and/or FILE)" ;;
      *)  [[ -z "$out" ]] || die "backup takes at most one FILE argument"
          out="$1"; shift ;;
    esac
  done
  resolve_mounts
  local mp_models mp_user mp_input mp_output mp_nodes mp_yaml
  mp_models="$(_mount_path models)"       mp_user="$(_mount_path user)"
  mp_input="$(_mount_path input)"         mp_output="$(_mount_path output)"
  mp_nodes="$(_mount_path custom_nodes)"  mp_yaml="$(_mount_path extra_model_paths.yaml)"
  if [[ -z "$out" ]]; then
    mkdir -p "$BASE_DIR/backups"
    out="$BASE_DIR/backups/spark-backup-$(date +%Y%m%d-%H%M%S).tgz"
  fi
  case "$out" in /*) ;; *) out="$PWD/$out" ;; esac

  # The ComfyUI commit comes from the image label.
  local ccommit
  ccommit="$(docker image inspect \
    -f '{{index .Config.Labels "org.spark-comfyui.comfy-sha"}}' \
    "$CONTAINER_IMAGE:latest" 2>/dev/null || true)"
  [[ -n "$ccommit" ]] || ccommit=unknown

  log "Staging backup"
  local stage; stage="$(mktemp -d)"
  trap 'rm -rf "$stage"' EXIT
  {
    echo "format=1"
    echo "version=$VERSION"
    echo "date=$(date -Is)"
    echo "host=$(hostname)"
    echo "comfyui_commit=$ccommit"
  } > "$stage/meta"

  [[ -f "$PATCH_LIST" ]] && cp -a "$PATCH_LIST" "$stage/comfyui-patches.list"
  [[ -f "$mp_yaml" ]] && cp -a "$mp_yaml" "$stage/extra_model_paths.yaml"

  # Custom nodes: git clones become manifest lines (url + commit, re-cloned on
  # restore); non-git entries are copied whole. "User-installed" = not tracked
  # by ComfyUI's own git (stock files like websocket_image_save.py are).
  : > "$stage/custom-nodes.manifest"
  local entry name url sha
  if [[ -d "$mp_nodes" ]]; then
    while IFS= read -r entry; do
      name="$(basename "$entry")"
      [[ "$name" == "__pycache__" ]] && continue
      if [[ -d "$entry/.git" ]]; then
        url="$(git -C "$entry" remote get-url origin 2>/dev/null || echo unknown)"
        sha="$(git -C "$entry" rev-parse HEAD 2>/dev/null || echo unknown)"
        printf '%s\t%s\t%s\n' "$name" "$url" "$sha" >> "$stage/custom-nodes.manifest"
      else
        mkdir -p "$stage/custom_nodes_plain"
        cp -a "$entry" "$stage/custom_nodes_plain/$name"
      fi
    done < <(find "$mp_nodes" -mindepth 1 -maxdepth 1 2>/dev/null)
  fi

  # Models are manifested (size + relative path), never copied.
  if [[ -d "$mp_models" ]]; then
    find "$mp_models" -type f -printf '%s\t%P\n' | sort -k2 > "$stage/models.manifest"
  else
    : > "$stage/models.manifest"
  fi

  # user/, input/ and output/ are tarred straight from the live tree (no
  # staging copy of possibly-large dirs): tar excludes the logs and caches,
  # --ignore-failed-read plus tolerating exit 1 (a file changed or vanished
  # mid-read) is what makes "safe while ComfyUI is serving" true. They
  # enter the archive through stage-dir symlinks with -h (dereference), so
  # a per-entry mount override still lands under its entry name. Symlinks
  # INSIDE the dirs are dereferenced too as a side effect; for
  # settings/workflow trees that is acceptable.
  local members=(-C "$stage" meta models.manifest custom-nodes.manifest)
  [[ -f "$stage/comfyui-patches.list" ]]   && members+=(comfyui-patches.list)
  [[ -f "$stage/extra_model_paths.yaml" ]] && members+=(extra_model_paths.yaml)
  [[ -d "$stage/custom_nodes_plain" ]]     && members+=(custom_nodes_plain)
  if [[ -d "$mp_user" ]]; then
    ln -s "$mp_user" "$stage/user"; members+=(user)
  fi
  if [[ -d "$mp_input" && -n "$(find "$mp_input" -mindepth 1 -print -quit)" ]]; then
    ln -s "$mp_input" "$stage/input"; members+=(input)
  fi
  if [[ "$with_output" -eq 1 && -d "$mp_output" ]]; then
    ln -s "$mp_output" "$stage/output"; members+=(output)
  fi
  local rc=0
  tar -czhf "$out" --exclude='__pycache__' --exclude='*.log' \
    --ignore-failed-read "${members[@]}" || rc=$?
  [[ "$rc" -le 1 ]] || die "tar failed (exit $rc) writing $out"
  [[ "$rc" -eq 1 ]] && warn "some files changed while being archived (ComfyUI serving?) — they may be stale in this backup"
  rm -rf "$stage"
  trap - EXIT

  log "Backup written"
  echo "  $out ($(du -h "$out" | cut -f1))"
  echo "  models manifested, not archived — restore lists what to re-download"
}

cmd_restore() {
  local archive="${1:-}"
  [[ -n "$archive" ]] || die "usage: $0 restore FILE"
  [[ -f "$archive" && -r "$archive" ]] || die "cannot read backup archive: $archive"
  case "$archive" in /*) ;; *) archive="$PWD/$archive" ;; esac

  log "Unpacking $archive"
  local stage; stage="$(mktemp -d)"
  trap 'rm -rf "$stage"' EXIT
  tar -xzf "$archive" -C "$stage"
  grep -qx 'format=1' "$stage/meta" 2>/dev/null \
    || die "not a spark-comfyui backup (meta lacks format=1): $archive"
  sed 's/^/  /' "$stage/meta"

  resolve_mounts
  need_docker
  check_legacy_layout
  if ! docker image inspect "$CONTAINER_IMAGE:latest" >/dev/null 2>&1; then
    log "No container image — building first"
    cmd_container_build
  fi
  cmd_container_stop

  # Archive members carry entry names; each restores into its RESOLVED
  # host path, so per-entry mount overrides are honored.
  log "Merging user state"
  local d mp
  for d in user input output; do
    [[ -d "$stage/$d" ]] || continue
    mp="$(_mount_path "$d")"
    mkdir -p "$mp"
    cp -a "$stage/$d/." "$mp/"
    echo "  merged $d/"
  done
  local src dst
  for d in extra_model_paths.yaml comfyui-patches.list; do
    src="$stage/$d"
    [[ -f "$src" ]] || continue
    case "$d" in
      extra_model_paths.yaml) dst="$(_mount_path extra_model_paths.yaml)" ;;
      comfyui-patches.list)   dst="$PATCH_LIST" ;;
      *) continue ;;   # a list entry without a case arm must not reuse $dst
    esac
    if [[ -f "$dst" ]] && ! cmp -s "$src" "$dst"; then
      cp -a "$dst" "$dst.bak"
      warn "live $(basename "$dst") differed from the archive — saved aside as $(basename "$dst").bak"
    fi
    cp -a "$src" "$dst"
    echo "  restored $(basename "$dst")"
  done

  log "Restoring custom nodes"
  local nodes_dir
  nodes_dir="$(_mount_path custom_nodes)"
  mkdir -p "$nodes_dir"
  local entry name url sha ndir
  if [[ -d "$stage/custom_nodes_plain" ]]; then
    while IFS= read -r entry; do
      name="$(basename "$entry")"
      if [[ -e "$nodes_dir/$name" ]]; then
        echo "  = $name (present)"
      else
        cp -a "$entry" "$nodes_dir/$name"
        echo "  + $name (plain copy)"
      fi
    done < <(find "$stage/custom_nodes_plain" -mindepth 1 -maxdepth 1)
  fi
  if [[ -f "$stage/custom-nodes.manifest" ]]; then
    while IFS=$'\t' read -r name url sha; do
      [[ -n "$name" ]] || continue
      # The name lands in a path below custom_nodes/; a tampered archive
      # must not be able to point it elsewhere. (Plain-copy names above come
      # from basename and can't carry a path.)
      case "$name" in
        .|..|*/*)
          warn "manifest names invalid node '$name' — skipped"; continue ;;
      esac
      ndir="$nodes_dir/$name"
      if [[ -e "$ndir" ]]; then
        echo "  = $name (present)"
        continue
      fi
      echo "  + $name (cloning $url)"
      # </dev/null: a prompting clone (ssh host key, credentials) must not
      # eat the manifest lines this loop is reading from stdin.
      if ! GIT_TERMINAL_PROMPT=0 git clone "$url" "$ndir" </dev/null; then
        warn "could not clone $name from $url — skipped"
        continue
      fi
      # Detached checkout of the archived commit; a miss (force-pushed
      # upstream, shallow mirror) is a warning, not a failed restore.
      if [[ -n "$sha" && "$sha" != "unknown" ]] \
         && ! git -C "$ndir" checkout -q "$sha" 2>/dev/null; then
        warn "$name: could not check out $sha — staying on clone HEAD"
      fi
    done < "$stage/custom-nodes.manifest"
  fi

  # The entrypoint installs every node's requirements and verifies torch on
  # each start, so a restore is content-only by design.
  info "node requirements and the torch guard run in the container entrypoint at next start"

  log "Models check (against the archive's manifest)"
  local models_dir
  models_dir="$(_mount_path models)"
  local missing_count=0 missing_bytes=0 size relpath
  if [[ -f "$stage/models.manifest" ]]; then
    while IFS=$'\t' read -r size relpath; do
      [[ -n "$relpath" ]] || continue
      # A corrupt manifest line must not kill the restore via the size
      # arithmetic below (set -e); treat the size as unknown instead.
      [[ "$size" =~ ^[0-9]+$ ]] || size=0
      [[ -f "$models_dir/$relpath" ]] && continue
      printf '  missing: %s (%s)\n' "$relpath" "$(numfmt --to=iec "$size" 2>/dev/null || echo "${size}B")"
      missing_count=$((missing_count + 1))
      missing_bytes=$((missing_bytes + size))
    done < "$stage/models.manifest"
    if [[ "$missing_count" -gt 0 ]]; then
      warn "$missing_count models missing, $(numfmt --to=iec "$missing_bytes" 2>/dev/null || echo "${missing_bytes}B"): download or rsync them separately"
    else
      echo "  all manifested models present"
    fi
  fi
  rm -rf "$stage"
  trap - EXIT

  log "Restore complete"
  echo "  Start ComfyUI:  $0 run"
}

# =============================================================================
#  the containerized runtime
# =============================================================================
# The image holds everything reproducible (ComfyUI at a pinned commit, venv
# with cu130 torch, native sm_121 SageAttention, GPU onnxruntime, build-time
# mods); the USER_CONTENT set is bind-mounted from data/. The build has no
# GPU, so the live kernel gates run in the container entrypoint on every
# start (container/entrypoint.sh), not at build time.

# Mount resolution. Per entry: a
# spark-mounts.conf key wins, else DATA_DIR/<entry>. Fills
# RESOLVED_ENTRIES/RESOLVED_PATHS (parallel arrays) and EXTRA_MOUNTS
# (HOST:CONTAINER[:ro] strings, validated).
resolve_mounts() {
  RESOLVED_ENTRIES=() RESOLVED_PATHS=() EXTRA_MOUNTS=()
  local -A conf=()
  local line key val known entry
  if [[ -f "$MOUNTS_CONF" ]]; then
    local conf_dir
    conf_dir="$(dirname "$(readlink -f "$MOUNTS_CONF")")"
    while IFS= read -r line; do
      line="${line%%#*}"
      line="${line#"${line%%[![:space:]]*}"}"
      line="${line%"${line##*[![:space:]]}"}"
      [[ -z "$line" ]] && continue
      [[ "$line" == *=* ]] \
        || die "spark-mounts.conf: not a KEY = PATH line: '$line'"
      key="${line%%=*}"; key="${key%"${key##*[![:space:]]}"}"
      val="${line#*=}";  val="${val#"${val%%[![:space:]]*}"}"
      [[ -n "$val" ]] || die "spark-mounts.conf: empty path for '$key'"
      if [[ "$key" == "mount" ]]; then
        [[ "$val" == *:/opt/ComfyUI/* ]] \
          || die "spark-mounts.conf: mount must be HOST:CONTAINER[:ro] with the
container path under /opt/ComfyUI — got: '$val'"
        local mhost="${val%%:*}"
        [[ -e "$mhost" ]] \
          || die "spark-mounts.conf: mount host path does not exist: $mhost
(refusing to invent an empty dir — a typo must not silently shadow your data)"
        EXTRA_MOUNTS+=("$val")
        continue
      fi
      known=0
      for entry in "${USER_CONTENT[@]}"; do
        [[ "$key" == "$entry" ]] && known=1
      done
      [[ "$known" == 1 ]] \
        || die "spark-mounts.conf: unknown key '$key' (valid: ${USER_CONTENT[*]}, mount)"
      [[ "$val" == /* ]] || val="$conf_dir/$val"
      conf["$key"]="$val"
    done < "$MOUNTS_CONF"
  fi
  for entry in "${USER_CONTENT[@]}"; do
    RESOLVED_ENTRIES+=("$entry")
    if [[ -n "${conf[$entry]:-}" ]]; then
      RESOLVED_PATHS+=("${conf[$entry]}")
    else
      RESOLVED_PATHS+=("$DATA_DIR/$entry")
    fi
  done
}

# A native-era layout (pre-container) has a ComfyUI checkout where data/
# should be. The move is five renames, so the gate carries the
# instructions itself. Without it, install would create an empty data/
# and silently shadow the user's content.
check_legacy_layout() {
  [[ -d "$BASE_DIR/ComfyUI/.git" && ! -d "$DATA_DIR" ]] || return 0
  die "a native-era install layout was detected (ComfyUI checkout, no data/).
This version keeps all content in data/. Move yours first (instant
renames; run from $BASE_DIR):
  mkdir -p data
  mv ComfyUI/models ComfyUI/user ComfyUI/input ComfyUI/output \\
     ComfyUI/custom_nodes data/ 2>/dev/null
  mv ComfyUI/extra_model_paths.yaml data/ 2>/dev/null
then re-run: $0 install
The old ComfyUI/, comfyui-env/ and SageAttention/ trees are reproducible;
delete them once you are satisfied."
}

# The single directory holding the whole USER_CONTENT set: the legacy
# checkout (native layout) or DATA_DIR (container-only layout).
# backup/restore operate through this root. Per-entry spark-mounts.conf
# overrides scatter the set across parents, which backup/restore do not
# support yet; that dies loudly instead of silently archiving half the
# content.
# Resolved host path for one USER_CONTENT entry (resolve_mounts must have
# run in this shell). backup/restore go through this, so per-entry
# spark-mounts.conf overrides are honored: the archive always uses entry
# names, wherever the entries live on the host.
_mount_path() {
  local i
  for i in "${!RESOLVED_ENTRIES[@]}"; do
    if [[ "${RESOLVED_ENTRIES[$i]}" == "$1" ]]; then
      echo "${RESOLVED_PATHS[$i]}"
      return 0
    fi
  done
  return 1
}

seed_mounts_conf() {
  [[ -f "$MOUNTS_CONF" ]] && return 0
  cat > "$MOUNTS_CONF" <<'EOF'
# spark-mounts.conf — where the container finds your content.
# Uncommented lines are KEY = PATH. Relative paths resolve against this
# file's directory. Without overrides everything lives under data/ next to
# the script. 'container status' always shows the resolved table.
#
# Per-entry overrides (keys match the content set exactly):
# models = /mnt/fast-ssd/models
# output = /mnt/nas/comfyui-output
# user = data/user
# input = data/input
# custom_nodes = data/custom_nodes
# extra_model_paths.yaml = data/extra_model_paths.yaml
#
# Additional bind mounts, repeatable: HOST:CONTAINER[:ro]. The container
# path must be under /opt/ComfyUI and the host path must already exist.
# Pair extra model locations with entries in extra_model_paths.yaml, which
# sees the CONTAINER paths:
# mount = /mnt/nas/sdxl-models:/opt/ComfyUI/models/nas:ro
EOF
  info "seeded mount config template: $MOUNTS_CONF"
}

need_docker() {
  command -v docker >/dev/null 2>&1 \
    || die "docker not found — DGX OS ships it; otherwise install Docker and
the NVIDIA Container Toolkit, then re-run."
  docker info 2>/dev/null | grep -q "nvidia" \
    || warn "the nvidia runtime is not visible in 'docker info' — GPU
passthrough may fail (install/configure the NVIDIA Container Toolkit)"
}

cmd_container_build() {
  need_docker
  seed_mounts_conf
  log "Resolving upstream ComfyUI master"
  local comfy_sha
  comfy_sha="$(timeout 30 git ls-remote "$REPO_URL" refs/heads/master 2>/dev/null \
    | awk 'NR==1{print $1}')"
  [[ -n "$comfy_sha" ]] \
    || die "could not resolve ComfyUI master from $REPO_URL (offline or
unreachable) — check the network and re-run: $0 container build"
  local date_tag; date_tag="$(date +%Y.%m.%d)"
  log "Building $CONTAINER_IMAGE:$date_tag (ComfyUI ${comfy_sha:0:12}, SageAttention ${SAGE_REF:0:12})"
  echo "  First build downloads torch (>1 GB) and compiles SageAttention"
  echo "  (10-30 min). Rebuilds reuse every layer that didn't change."
  # --provenance=false: buildx otherwise attaches a provenance attestation
  # stamped with the build time, giving every build a fresh manifest digest
  # even when all layers are cached and the content is identical — which
  # breaks cmd_container_update's changed-vs-current comparison
  # (field-diagnosed 2026-07-20: two builds, same config timestamp,
  # different "image IDs").
  docker build \
    --provenance=false \
    -f "$BASE_DIR/container/Dockerfile" \
    --build-arg TORCH_INDEX="$TORCH_INDEX" \
    --build-arg REPO_URL="$REPO_URL" \
    --build-arg COMFY_SHA="$comfy_sha" \
    --build-arg SAGE_REF="$SAGE_REF" \
    --build-arg ORT_WHEEL_URL="$ORT_WHEEL_URL" \
    -t "$CONTAINER_IMAGE:$date_tag" \
    -t "$CONTAINER_IMAGE:latest" \
    "$@" \
    "$BASE_DIR"
  log "Image ready: $CONTAINER_IMAGE:latest (also tagged :$date_tag)"
  echo "  Launch it: $0 container run"
}

# Shared docker-run argument assembly for cmd_container_run (foreground,
# --rm) and cmd_container_service (detached, restart policy). Fills the
# CRUN_ARGS array: hardening flags (no capabilities, no privilege
# escalation, only the GPU), the cache volume, and the resolved content
# mounts. Dirs are created if missing; the yaml entry is a file and only
# mounted when it exists (docker would create a dir).
_container_run_args() {
  docker image inspect "$CONTAINER_IMAGE:latest" >/dev/null 2>&1 \
    || die "image $CONTAINER_IMAGE:latest not found — run: $0 container build"
  resolve_mounts
  CRUN_ARGS=(
    --name "$CONTAINER_NAME"
    --gpus all
    --shm-size 1g
    --cap-drop ALL
    --security-opt no-new-privileges
    -p "$PORT:8188"
    -v "$CONTAINER_IMAGE-cache:/home/comfy/.cache"
    -e SPARK_BF16
    -e SPARK_STATIC_VRAM
  )
  local entry host i
  for i in "${!RESOLVED_ENTRIES[@]}"; do
    entry="${RESOLVED_ENTRIES[$i]}" host="${RESOLVED_PATHS[$i]}"
    if [[ "$entry" == *.yaml ]]; then
      [[ -f "$host" ]] && CRUN_ARGS+=(-v "$host:/opt/ComfyUI/$entry:ro")
    else
      mkdir -p "$host"
      CRUN_ARGS+=(-v "$host:/opt/ComfyUI/$entry")
    fi
  done
  local m
  for m in "${EXTRA_MOUNTS[@]}"; do
    CRUN_ARGS+=(-v "$m")
  done
}

cmd_container_run() {
  need_docker
  _container_run_args
  log "Launching containerized ComfyUI on port $PORT (Ctrl-C stops it)"
  # --rm: every launch starts from the immutable image; runtime pip state
  # lives at most until the container exits, and the cache volume keeps
  # downloads and compiled sm_121 kernels fast across recreation.
  exec docker run --rm "${CRUN_ARGS[@]}" "$CONTAINER_IMAGE:latest" "$@"
}

# The container-world 'service': a detached container with a docker restart
# policy instead of a systemd unit. The docker daemon restarts it after
# crashes and reboots; no user lingering, no unit files.
cmd_container_service() {
  need_docker
  if [[ "${1:-}" == "--disable" ]]; then
    if [[ -n "$(docker ps -aq -f "name=^${CONTAINER_NAME}$")" ]]; then
      docker rm -f "$CONTAINER_NAME" >/dev/null
      log "Service disabled and container removed"
    else
      info "no service container to disable"
    fi
    return 0
  fi
  [[ -z "${1:-}" ]] || die "Unknown container service option: $1 (valid: --disable)"
  if [[ -n "$(docker ps -q -f "name=^${CONTAINER_NAME}$")" ]]; then
    info "container $CONTAINER_NAME is already running ($0 container stop to stop it)"
    return 0
  fi
  # A stopped leftover with the same name (e.g. a previously disabled
  # service after a reboot) blocks the new run; a RUNNING one was handled
  # above, so removing here is safe.
  docker rm -f "$CONTAINER_NAME" >/dev/null 2>&1 || true
  _container_run_args
  log "Starting containerized ComfyUI as a service (docker restart policy)"
  docker run -d --restart unless-stopped "${CRUN_ARGS[@]}" \
    "$CONTAINER_IMAGE:latest" >/dev/null
  echo "  Running detached on port $PORT; survives crashes and reboots."
  echo "  Logs:    docker logs -f $CONTAINER_NAME"
  echo "  Stop:    $0 container stop   (docker restarts it on next boot)"
  echo "  Disable: $0 container service --disable"
}

# Container-world install: no venv, no apt, no sudo. Preflight, seed the
# config templates, create data/, build the image. Idempotent.
cmd_container_install() {
  need_docker
  check_legacy_layout
  install_self
  seed_mounts_conf
  seed_patch_list
  mkdir -p "$DATA_DIR"
  cmd_container_build
  local ip_hint
  ip_hint="$(hostname -I 2>/dev/null | awk '{print $1}')"
  log "Done!"
  cat <<EOF

  Content root:     $DATA_DIR   (mounts: $MOUNTS_CONF)
  Start ComfyUI:    $0 container run     (foreground)
        or:         $0 container service (background, survives reboots)
  Health check:     $0 container doctor
  Update later:     $0 container update
  Web UI:           http://${ip_hint:-<spark-ip>}:$PORT
  Models go in:     $DATA_DIR/models/checkpoints (etc.)
EOF
}

# Container-world reset: content is outside by design, so reset only
# removes what is reproducible (containers, every image tag, the cache
# volume) and rebuilds from scratch. data/ is never touched.
cmd_container_reset() {
  need_docker
  local yes=0
  [[ "${1:-}" == "--yes" ]] && yes=1
  if [[ "$yes" != 1 ]]; then
    [[ -t 0 ]] || die "stdin is not a terminal — re-run with: $0 container reset --yes"
    echo "  This removes the container, ALL $CONTAINER_IMAGE image tags and the"
    echo "  cache volume, then rebuilds the image from scratch (no cache,"
    echo "  including the 10-30 min SageAttention compile)."
    echo "  Your content is not touched."
    local answer
    read -r -p "  Proceed? [y/N] " answer
    [[ "$answer" == y || "$answer" == Y ]] || die "aborted — nothing removed"
  fi
  cmd_container_stop
  docker rm -f "$CONTAINER_NAME" >/dev/null 2>&1 || true
  local ids
  ids="$(docker images "$CONTAINER_IMAGE" -q | sort -u)"
  if [[ -n "$ids" ]]; then
    # shellcheck disable=SC2086
    docker rmi -f $ids >/dev/null
    log "Removed all $CONTAINER_IMAGE images"
  fi
  docker volume rm "$CONTAINER_IMAGE-cache" >/dev/null 2>&1 \
    && log "Removed cache volume" || true
  cmd_container_build --no-cache
}

cmd_container_stop() {
  need_docker
  if [[ -n "$(docker ps -q -f "name=^${CONTAINER_NAME}$")" ]]; then
    log "Stopping container $CONTAINER_NAME"
    docker stop "$CONTAINER_NAME" >/dev/null
    echo "Stopped."
  else
    info "container $CONTAINER_NAME is not running"
  fi
}

cmd_container_shell() {
  need_docker
  docker exec -it "$CONTAINER_NAME" bash \
    || die "could not exec into $CONTAINER_NAME — is it running? ($0 container run)"
}

cmd_container_update() {
  local rollback=0 torch=0 arg
  for arg in "$@"; do
    case "$arg" in
      --rollback) rollback=1 ;;
      --torch)    torch=1 ;;
      *) die "Unknown container update option: $arg" ;;
    esac
  done
  need_docker
  check_legacy_layout

  if (( rollback )); then
    docker image inspect "$CONTAINER_IMAGE:previous" >/dev/null 2>&1 \
      || die "no $CONTAINER_IMAGE:previous image — nothing to roll back to
(the tag appears after the first 'container update' that changes the image)"
    local cur prev
    cur="$(docker image inspect -f '{{.Id}}' "$CONTAINER_IMAGE:latest" 2>/dev/null || true)"
    prev="$(docker image inspect -f '{{.Id}}' "$CONTAINER_IMAGE:previous")"
    if [[ "$cur" == "$prev" ]]; then
      info "latest and previous are the same image — nothing to roll back"
      return 0
    fi
    docker tag "$CONTAINER_IMAGE:previous" "$CONTAINER_IMAGE:latest"
    [[ -n "$cur" ]] && docker tag "$cur" "$CONTAINER_IMAGE:previous"
    log "Rolled back: $CONTAINER_IMAGE:latest is now ${prev:7:12}"
    echo "  (:previous now holds the image you rolled back FROM, so running"
    echo "  'container update --rollback' again toggles forward.)"
    echo "  Restart to pick it up: $0 container stop && $0 container run"
    return 0
  fi

  # Self-update the tool first, same as the native update; the resume hook
  # makes the post-update re-exec land back here instead of in cmd_update.
  local SELF_UPDATE_RESUME="container update"
  self_update "$@"

  # Hold a temp tag on the current image through the build: the rebuild
  # moves :latest and the date tag to the new image, and the containerd
  # image store garbage-collects a tagless image INSTANTLY (field-hit
  # 2026-07-20: tagging :previous after the build found the old image
  # already gone). Promote the temp tag to :previous only on a real change,
  # so an unchanged rebuild never clobbers an older rollback point.
  # --torch: bust exactly the torch stage (fresh cu130 wheels); the
  # SageAttention stage sits on top of it and rebuilds automatically, same
  # semantic as the native update --torch.
  local build_args=()
  (( torch )) && build_args+=(--no-cache-filter=torch)
  local before after
  before="$(docker image inspect -f '{{.Id}}' "$CONTAINER_IMAGE:latest" 2>/dev/null || true)"
  [[ -n "$before" ]] && docker tag "$before" "$CONTAINER_IMAGE:pre-update"
  cmd_container_build "${build_args[@]}"
  after="$(docker image inspect -f '{{.Id}}' "$CONTAINER_IMAGE:latest")"
  if [[ "$before" == "$after" ]]; then
    [[ -n "$before" ]] && docker rmi "$CONTAINER_IMAGE:pre-update" >/dev/null
    log "Already current — the rebuild produced the same image"
  else
    if [[ -n "$before" ]]; then
      docker tag "$CONTAINER_IMAGE:pre-update" "$CONTAINER_IMAGE:previous"
      docker rmi "$CONTAINER_IMAGE:pre-update" >/dev/null
      log "Updated. The old image stays as $CONTAINER_IMAGE:previous"
      echo "  Roll back with: $0 container update --rollback"
    else
      log "Updated (first image — nothing previous to keep)"
    fi
  fi
  if [[ -n "$(docker ps -q -f "name=^${CONTAINER_NAME}$")" ]]; then
    warn "the running container still uses the old image — restart to pick up
the update: $0 container stop && $0 container run"
  fi
}

cmd_container_status() {
  need_docker
  echo
  echo "== image =="
  if docker image inspect "$CONTAINER_IMAGE:latest" >/dev/null 2>&1; then
    docker images "$CONTAINER_IMAGE" \
      --format '{{.Tag}}\t{{.ID}}\t{{.Size}}\t(created {{.CreatedSince}})' \
      | expand -t 14,28,38 | sed 's/^/  /'
  else
    echo "  none — run: $0 container build"
  fi
  echo
  echo "== container =="
  local cid
  cid="$(docker ps -q -f "name=^${CONTAINER_NAME}$")"
  if [[ -n "$cid" ]]; then
    docker ps -f "id=$cid" --format '{{.Names}}: {{.Status}}, port {{.Ports}}' \
      | sed 's/^/  /'
    # Quiet when healthy: only flag when the running image is not :latest
    # (i.e. an update happened under a running server).
    local running_img latest_img
    running_img="$(docker inspect -f '{{.Image}}' "$cid")"
    latest_img="$(docker image inspect -f '{{.Id}}' "$CONTAINER_IMAGE:latest" 2>/dev/null || true)"
    if [[ -n "$latest_img" && "$running_img" != "$latest_img" ]]; then
      warn "running an image that is no longer :latest — restart to pick up
the update: $0 container stop && $0 container run"
    fi
    echo
    echo "== mounts =="
    docker inspect "$cid" \
      --format '{{range .Mounts}}{{.Type}}{{"\t"}}{{.Source}}{{"\t"}}-> {{.Destination}}{{"\n"}}{{end}}' \
      | sed '/^$/d' | expand -t 8,64 | sed 's/^/  /'
  else
    echo "  not running — start: $0 container run"
  fi
  echo
  echo "== configured mounts (resolved) =="
  resolve_mounts
  local ci
  for ci in "${!RESOLVED_ENTRIES[@]}"; do
    printf '%s\t%s\n' "${RESOLVED_ENTRIES[$ci]}" "${RESOLVED_PATHS[$ci]}"
  done | expand -t 26 | sed 's/^/  /'
  local cm
  for cm in "${EXTRA_MOUNTS[@]}"; do
    echo "  extra: $cm"
  done
  [[ -f "$MOUNTS_CONF" ]] \
    && echo "  (overrides: $MOUNTS_CONF)" \
    || echo "  (no spark-mounts.conf — defaults; seeded on next build)"
  echo
  echo "== cache volume =="
  if docker volume inspect "$CONTAINER_IMAGE-cache" >/dev/null 2>&1; then
    echo "  $CONTAINER_IMAGE-cache (pip + compiled CUDA kernels; safe to"
    echo "  delete at the cost of a slower next start)"
  else
    echo "  none yet (created on first 'container run')"
  fi
}

cmd_container_doctor() {
  need_docker
  # ok/bad increment these shared counters.
  PASS=0; FAIL=0

  # Version first: doctor output doubles as a bug report. The pending-update
  # probe lives here (and only here) on purpose — it needs a network fetch,
  # which has no business on the run/stop/status hot paths. warn, not bad: a
  # pending release is not a health failure.
  hdr "spark-comfyui (self)"
  if [[ -d "$BASE_DIR/.git" ]]; then
    info "git revision $(git -C "$BASE_DIR" rev-parse --short HEAD 2>/dev/null)"
  else
    info "not a git clone — self-update unavailable"
  fi
  if [[ -d "$BASE_DIR/.git" ]] && timeout 5 git -C "$BASE_DIR" fetch -q origin 2>/dev/null; then
    local self_local self_up remote_ver
    self_local="$(git -C "$BASE_DIR" rev-parse HEAD 2>/dev/null)"
    self_up="$(git -C "$BASE_DIR" rev-parse '@{u}' 2>/dev/null || echo "$self_local")"
    if [[ "$self_local" == "$self_up" ]]; then
      info "up to date with the published repo"
    elif git -C "$BASE_DIR" merge-base --is-ancestor "$self_local" "$self_up" 2>/dev/null; then
      remote_ver="$(git -C "$BASE_DIR" show '@{u}:spark-comfyui.sh' 2>/dev/null \
        | sed -n 's/^VERSION="\([^"]*\)".*/\1/p' | head -1)"
      warn "a newer spark-comfyui is published (v${remote_ver:-?}) — get it: $0 update"
    else
      info "local clone has diverged from the published repo (local commits)"
    fi
  fi

  hdr "container host"
  local drv
  drv="$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -1 || true)"
  if [[ -n "$drv" ]]; then
    ok "NVIDIA driver $drv"
  else
    bad "nvidia-smi not working on the host — the container cannot get a GPU"
  fi
  if docker info 2>/dev/null | grep -q nvidia; then
    ok "docker daemon up, nvidia runtime registered"
  else
    bad "nvidia runtime missing from docker — install the NVIDIA Container Toolkit"
  fi
  if docker image inspect "$CONTAINER_IMAGE:latest" >/dev/null 2>&1; then
    local csha created
    csha="$(docker image inspect -f '{{index .Config.Labels "org.spark-comfyui.comfy-sha"}}' "$CONTAINER_IMAGE:latest" 2>/dev/null || true)"
    created="$(docker image inspect -f '{{.Created}}' "$CONTAINER_IMAGE:latest" | cut -dT -f1)"
    ok "image $CONTAINER_IMAGE:latest (built $created, ComfyUI ${csha:0:12})"
    docker image inspect "$CONTAINER_IMAGE:previous" >/dev/null 2>&1 \
      && info "rollback point present ($CONTAINER_IMAGE:previous)" \
      || info "no rollback point yet (:previous appears after the first changing update)"
  else
    bad "image $CONTAINER_IMAGE:latest missing — run: $0 container build"
  fi
  local cid
  cid="$(docker ps -q -f "name=^${CONTAINER_NAME}$")"
  if [[ -n "$cid" ]]; then
    local running_img latest_img
    running_img="$(docker inspect -f '{{.Image}}' "$cid")"
    latest_img="$(docker image inspect -f '{{.Id}}' "$CONTAINER_IMAGE:latest" 2>/dev/null || true)"
    if [[ -n "$latest_img" && "$running_img" != "$latest_img" ]]; then
      bad "running container uses an outdated image — restart: $0 container stop && $0 container run"
    else
      ok "container running ($(docker ps -f "id=$cid" --format '{{.Status}}'))"
    fi
  else
    info "container not running"
  fi
  if [[ -n "$(swapon --noheadings 2>/dev/null)" ]]; then
    warn "swap is ENABLED on the host — heavy workloads can freeze the box ($0 tune)"
  else
    ok "swap disabled on the host"
  fi
  # Informational, not pass/fail: when the newest backup was taken. The -d
  # gate covers the never-backed-up install; the || true covers a nonzero
  # find/head pipeline status under pipefail. Both are load-bearing.
  local newest_line="" newest_backup backup_age age_txt
  [[ -d "$BASE_DIR/backups" ]] && newest_line="$(find "$BASE_DIR/backups" \
    -maxdepth 1 -name 'spark-backup-*.tgz' -printf '%T@ %p\n' 2>/dev/null \
    | sort -rn | head -1 || true)"
  if [[ -n "$newest_line" ]]; then
    newest_backup="${newest_line#* }"
    backup_age=$(( ($(date +%s) - ${newest_line%%.*}) / 86400 ))
    age_txt="$backup_age days old"
    [[ "$backup_age" -eq 0 ]] && age_txt="today"
    [[ "$backup_age" -eq 1 ]] && age_txt="1 day old"
    info "Backup: $(basename "$newest_backup") ($age_txt)"
  else
    info "Backup: none in backups/ (run: $0 backup)"
  fi

  docker image inspect "$CONTAINER_IMAGE:latest" >/dev/null 2>&1 \
    || die "image missing — the GPU gates need it; run: $0 container build"
  log "Running the GPU gates inside a throwaway container"
  # Same live gates the native doctor runs, executed in the image itself:
  # what passes here is exactly what the entrypoint will see at launch.
  local gate_rc=0
  docker run --rm -i --gpus all --entrypoint bash "$CONTAINER_IMAGE:latest" -s <<'EOS' || gate_rc=$?
set -uo pipefail
ok()   { printf '  \033[1;32m[PASS]\033[0m %s\n' "$*"; }
bad()  { printf '  \033[1;31m[FAIL]\033[0m %s\n' "$*"; fails=$((fails+1)); }
log()  { printf '\n\033[1m== %s ==\033[0m\n' "$*"; }
warn() { printf '\033[1;33m[warn] %s\033[0m\n' "$*"; }
die()  { printf '\033[1;31m[error] %s\033[0m\n' "$*" >&2; exit 1; }
info() { printf '  \033[1;36m[info]\033[0m %s\n' "$*"; }
fails=0
source /opt/spark/mods/_lib/mod_common.sh

log "torch / CUDA"
if python - <<'PY'
import torch
print(f"  torch {torch.__version__} | compiled CUDA {torch.version.cuda}")
assert (torch.version.cuda or "").startswith("13")
assert torch.cuda.is_available()
cap = torch.cuda.get_device_capability(0)
print(f"  device: {torch.cuda.get_device_name(0)} | sm_{cap[0]}{cap[1]}")
PY
then ok "torch is the cu130 CUDA build and sees the GPU"
else torch_cuda_diag; bad "torch cannot use the GPU — the diag lines above name the cause"
fi

log "SageAttention (live kernel)"
if sage_kernel_ok; then ok "sm_121 kernel runs"
else bad "kernel failed — rebuild the image (container build)"; fi

log "onnxruntime GPU"
if onnx_gpu_ok; then ok "CUDAExecutionProvider available"
else bad "GPU provider missing — rebuild the image (container build)"; fi

log "NVFP4 (comfy-kitchen forced cuda backend)"
if kitchen_nvfp4_ok; then ok "forced NVFP4 quantize+matmul passed"
else bad "NVFP4 gate failed on the cuda backend"; fi

echo
if [[ "$fails" -eq 0 ]]; then echo "All container gates passed."
else echo "$fails gate(s) FAILED."; exit 1; fi
EOS
  echo
  if [[ "$FAIL" -eq 0 && "$gate_rc" -eq 0 ]]; then
    echo "Host checks: $PASS passed. Everything healthy."
  else
    echo "Host checks: $PASS passed, $FAIL failed; GPU gates $( [[ $gate_rc -eq 0 ]] && echo passed || echo FAILED )."
    return 1
  fi
}

cmd_container() {
  local sub="${1:-}"
  shift || true
  case "$sub" in
    install) cmd_container_install ;;
    build)   cmd_container_build "$@" ;;
    run)     cmd_container_run "$@" ;;
    service) cmd_container_service "$@" ;;
    stop)    cmd_container_stop ;;
    shell)   cmd_container_shell ;;
    update)  cmd_container_update "$@" ;;
    status)  cmd_container_status ;;
    doctor)  cmd_container_doctor ;;
    reset)   cmd_container_reset "$@" ;;
    *) die "Unknown container subcommand: ${sub:-<none>} (try: container install | build | run | service | stop | update | status | doctor | reset | shell)" ;;
  esac
}

# ------------------------------- Dispatch -----------------------------------
# Sourced rather than executed (test harnesses source this file for its
# function definitions): stop here, never dispatch.
if [[ "${BASH_SOURCE[0]}" != "$0" ]]; then return 0; fi
CMD="${1:-}"
shift || true
# Banner on every invocation; --version excepted (kept one-line parseable).
case "$CMD" in
  -v|--version|version) ;;
  *) banner ;;
esac
case "$CMD" in
  install)  cmd_container_install ;;
  run)      cmd_container_run "$@" ;;
  service)  cmd_container_service "$@" ;;
  stop)     cmd_stop ;;
  update)   cmd_container_update "$@" ;;
  doctor)   cmd_container_doctor ;;
  status)   cmd_status "$@" ;;
  tune)     cmd_tune "$@" ;;
  backup)   cmd_backup "$@" ;;
  restore)  cmd_restore "$@" ;;
  reset)    cmd_container_reset "$@" ;;
  shell)    cmd_container_shell ;;
  # --- hidden backward-compat aliases (old command spellings still work) ---
  container) cmd_container "$@" ;;
  verify)   cmd_container_doctor ;;
  monitor)  cmd_status --watch ;;
  rollback) cmd_container_update --rollback ;;
  ""|-h|--help|help) usage ;;
  -v|--version|version) echo "spark-comfyui $VERSION" ;;
  *) die "Unknown command: $CMD (try: install | run | service | stop | update | doctor | status | tune | backup | restore | reset)" ;;
esac
