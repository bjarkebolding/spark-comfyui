#!/usr/bin/env bash
# =============================================================================
#  spark-comfyui.sh — ComfyUI on NVIDIA DGX Spark (GB10 Grace Blackwell)
#  Version 1.1.0 | License: MIT
# =============================================================================
#  One script for the whole lifecycle, tuned for the Spark's aarch64 CPU,
#  sm_121 GPU and 128 GB unified memory.
#
#  Commands:
#    install [--with-service]  One-shot setup: torch cu130, ComfyUI + Manager
#                              deps, SageAttention (built + kernel-verified),
#                              GPU onnxruntime (sm_121), Manager config.
#                              --with-service also installs the systemd
#                              user service (see 'service' below).
#    run [args...]             Start ComfyUI with GB10-optimized flags.
#                              Extra args pass through to main.py.
#    stop                      Stop ComfyUI (service or foreground process).
#    update [--torch|--rollback]
#                              Update ComfyUI + deps; rebuild SageAttention
#                              only if needed; repair anything shadowed.
#                              --torch upgrades PyTorch (forces Sage rebuild).
#                              --rollback returns to the pre-update revision.
#                              Optional: list PRs/branches to merge on top of
#                              upstream in comfyui-patches.list next to this
#                              script (pr:<N> | branch:<name> | remote:<url>
#                              <branch>); re-applied fresh on every update.
#    doctor                    Full health check: verifies every optimization
#                              is present AND active, and diagnoses the GB10
#                              silent-drift traps (shadowed torch/Sage/ONNX,
#                              stale ptxas/NVRTC, swap). Names each fix.
#    status [--watch]          One-page glance: process, GPU, memory, versions.
#                              --watch logs temp/power/RAM every 5s (evidence
#                              trail for diagnosing silent hard-reboots).
#    tune [--clock-cap MHZ] [--persist]
#                              System stability: disable swap (prevents
#                              unified-memory freezes), persistence mode,
#                              optional clock cap (~2100 fixes overcurrent
#                              hard-reboots). --persist survives reboots.
#    service                   Install + start a systemd user service.
#
#  Typical day: install once -> run -> update now and then.
#  Something feels wrong? -> doctor tells you what and how to fix it.
#  Re-running install is safe: completed steps are skipped or refreshed.
# =============================================================================
set -euo pipefail

VERSION="1.1.0"

# ----------------------------- Configuration --------------------------------
# Everything is self-contained under the directory this script lives in, so
# you can drop spark-comfyui.sh into any folder and it installs/runs there.
# Resolve the real location even if invoked via a symlink or a relative path.
SELF="$(readlink -f "${BASH_SOURCE[0]}")"
BASE_DIR="${BASE_DIR:-$(dirname "$SELF")}"

# All overridable via environment if you want them elsewhere, e.g.:
#   INSTALL_DIR=/data/ComfyUI VENV_DIR=/data/venv ./spark-comfyui.sh install
INSTALL_DIR="${INSTALL_DIR:-$BASE_DIR/ComfyUI}"
VENV_DIR="${VENV_DIR:-$BASE_DIR/comfyui-env}"
SAGE_SRC="${SAGE_SRC:-$BASE_DIR/SageAttention}"
# Pinned, not tracking thu-ml/SageAttention's default branch: 3.x showed
# mosaic artifacts on GB10 (a visual regression the live kernel test can't
# catch — it checks shape/finiteness, not output correctness). This is the
# exact commit field-verified on GB10 sm_121 (38 commits past the v2.2.0
# tag, still pre-3.0). Bump deliberately, not automatically.
SAGE_REF="${SAGE_REF:-d1a57a546c3d395b1ffcbeecc66d81db76f3b4b5}"
REPO_URL="${REPO_URL:-https://github.com/Comfy-Org/ComfyUI.git}"
TORCH_INDEX="${TORCH_INDEX:-https://download.pytorch.org/whl/cu130}"
# Community sm_121/aarch64/cu13 GPU onnxruntime (no official PyPI wheel exists)
ORT_WHEEL_URL="${ORT_WHEEL_URL:-https://huggingface.co/Jay0515/onnxruntime-gpu-aarch64-cuda13-sm121/resolve/main/onnxruntime_gpu-1.25.0-cp312-cp312-linux_aarch64.whl}"
PORT="${PORT:-8188}"
SAGE_MARKER="$VENV_DIR/.sage_ok"

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
PATCH_BRANCH="spark-patched"

# GB10 mods live in mods/<name>/run.sh and are discovered, applied, and
# verified through a small contract (see mods/README.md). Toggle all mods
# off with SPARK_SOURCE_PATCHES=0.
MODS_DIR="${MODS_DIR:-$BASE_DIR/mods}"

log()  { printf '\n\033[1;32m==> %s\033[0m\n' "$*"; }
warn() { printf '\033[1;33m[warn] %s\033[0m\n' "$*"; }
die()  { printf '\033[1;31m[error] %s\033[0m\n' "$*" >&2; exit 1; }
ok()   { printf '  \033[1;32m[PASS]\033[0m %s\n' "$*"; PASS=$((PASS+1)); }
bad()  { printf '  \033[1;31m[FAIL]\033[0m %s\n' "$*"; FAIL=$((FAIL+1)); }
info() { printf '  \033[1;36m[info]\033[0m %s\n' "$*"; }
hdr()  { printf '\n\033[1m== %s ==\033[0m\n' "$*"; }

usage() { awk 'NR>1 && /^#/ {sub(/^# ?/,""); print; next} NR>1 {exit}' "$0"; }

activate_venv() {
  [[ -f "$VENV_DIR/bin/activate" ]] \
    || die "venv not found at $VENV_DIR — run: $0 install"
  # shellcheck disable=SC1091
  source "$VENV_DIR/bin/activate"
}

# GB10 venv-package helpers (need_nvcc, sage_kernel_ok, onnx_gpu_ok,
# ensure_onnx_gpu, ensure_setuptools_compat, repair_torch,
# build_and_verify_sage) live in mods/_lib/mod_common.sh now — shared by the
# mods that wrap them (05/20/40/50) and by the direct call sites below
# (cmd_run's shadow check, cmd_rollback, cmd_doctor's diagnostics).
# shellcheck disable=SC1091
source "$MODS_DIR/_lib/mod_common.sh"

# Bring the local checkout's base branch up to date with upstream, safely,
# even when a previous run left us on the spark-patched branch.
# Sets: COMFY_BASE (branch name), COMFY_MOVED (0/1 upstream changed).
sync_comfyui() {
  cd "$INSTALL_DIR"
  COMFY_BASE="$(git symbolic-ref --short refs/remotes/origin/HEAD 2>/dev/null | sed 's|^origin/||')"
  COMFY_BASE="${COMFY_BASE:-master}"
  local old_head new_base
  old_head="$(git rev-parse HEAD)"
  # Rollback point = wherever we were before this sync (base or patched).
  mkdir -p "$VENV_DIR" 2>/dev/null || true
  echo "$old_head" > "$VENV_DIR/.last_comfyui_rev" 2>/dev/null || true
  git fetch -q origin
  git checkout -q "$COMFY_BASE" 2>/dev/null \
    || git checkout -qb "$COMFY_BASE" "origin/$COMFY_BASE"
  local base_before
  base_before="$(git rev-parse HEAD)"
  git merge -q --ff-only "origin/$COMFY_BASE"
  new_base="$(git rev-parse HEAD)"
  COMFY_MOVED=0
  if [[ "$base_before" != "$new_base" ]]; then
    COMFY_MOVED=1
    echo "ComfyUI $COMFY_BASE updated: ${base_before:0:8} -> ${new_base:0:8}"
    git log --oneline "${base_before}..${new_base}" | head -15
  else
    echo "ComfyUI $COMFY_BASE already up to date (${new_base:0:8})"
  fi
}

# Apply the patch list (PRs/branches) on top of fresh base, on the
# spark-patched branch. Master stays pristine; the branch is rebuilt from
# scratch every time so the result is reproducible. Conflicting entries are
# skipped with a warning, already-merged entries are flagged for removal.
# Sets: PATCHES_ACTIVE (0/1 any patch merged).
apply_patches() {
  PATCHES_ACTIVE=0
  cd "$INSTALL_DIR"
  # Seed a self-documenting template on first run (all comments = empty list).
  if [[ ! -f "$PATCH_LIST" ]]; then
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
  fi
  if ! grep -qE '^[^#[:space:]]' "$PATCH_LIST"; then
    return 0   # no patches requested -> stay on the base branch
  fi
  log "Applying patch list ($PATCH_LIST) onto fresh $COMFY_BASE -> $PATCH_BRANCH"
  # Merge commits need a committer identity; don't depend on global config.
  local GITC=(git -c user.name="spark-comfyui" -c user.email="spark-comfyui@local")
  git branch -f "$PATCH_BRANCH" "$COMFY_BASE"
  git checkout -q "$PATCH_BRANCH"
  local line num br url desc failed=0
  while IFS= read -r line; do
    line="${line%%#*}"
    line="$(echo "$line" | xargs 2>/dev/null || true)"
    [[ -z "$line" ]] && continue
    case "$line" in
      pr:*|PR:*)
        num="${line#*:}"; desc="PR #$num"
        git fetch -q origin "+pull/${num}/head:__patch_tmp" \
          || { warn "  ! cannot fetch $desc (does it exist?)"; failed=1; continue; } ;;
      branch:*)
        br="${line#branch:}"; desc="origin branch '$br'"
        git fetch -q origin "+${br}:__patch_tmp" \
          || { warn "  ! cannot fetch $desc"; failed=1; continue; } ;;
      remote:*)
        url="${line#remote:}"; br="${url##* }"; url="${url%% *}"
        desc="'$br' from $url"
        git fetch -q "$url" "+${br}:__patch_tmp" \
          || { warn "  ! cannot fetch $desc"; failed=1; continue; } ;;
      *)
        warn "  ! unrecognized patch line: '$line' (use pr:<N>, branch:<name>, or remote:<url> <branch>)"
        continue ;;
    esac
    if git merge-base --is-ancestor __patch_tmp HEAD; then
      echo "  = $desc already contained in $COMFY_BASE — consider removing it from the list"
    elif "${GITC[@]}" merge -q --no-edit __patch_tmp >/dev/null 2>&1; then
      echo "  + merged $desc"
      PATCHES_ACTIVE=1
    else
      git merge --abort 2>/dev/null || true
      warn "  ! CONFLICT merging $desc — skipped. It may need manual rebasing
    against current $COMFY_BASE, or it conflicts with another listed patch."
      failed=1
    fi
    git branch -D __patch_tmp >/dev/null 2>&1 || true
  done < "$PATCH_LIST"
  if [[ "$PATCHES_ACTIVE" -eq 1 ]]; then
    echo "Running on branch '$PATCH_BRANCH' ($(git rev-parse --short HEAD))"
  else
    git checkout -q "$COMFY_BASE"   # nothing merged -> no point staying on it
  fi
  [[ "$failed" -eq 1 ]] && warn "some patches were skipped — review the messages above"
  return 0
}

# GB10 mods live in mods/<name>/run.sh and are discovered, applied, and
# verified through a small contract (see mods/README.md). Some edit
# ComfyUI's own source and self-heal after every git pull (idempotent);
# others manage venv packages (torch, SageAttention, onnxruntime) with the
# same idempotent-apply/verify contract, plus the MOD_CRITICAL/MOD_STREAM/
# mod_prerun extensions documented in mods/_lib/mod_common.sh.

# Run a verb (flags|apply|prerun|verify|describe) against one mod's run.sh in
# a subshell so its variables never leak between mods. Echoes whatever the
# verb echoes (buffered verbs only); returns the verb's exit status.
_run_mod() {
  local mod_dir="$1" verb="$2"
  ( set -euo pipefail
    # shellcheck disable=SC2034  # consumed by the sourced mod run.sh
    MOD_DIR="$mod_dir"  # used by sourced run.sh
    # shellcheck disable=SC1091
    source "$MODS_DIR/_lib/mod_common.sh"
    # shellcheck disable=SC1090
    source "$mod_dir/run.sh"
    case "$verb" in
      flags)    echo "MOD_CRITICAL=${MOD_CRITICAL:-0} MOD_STREAM=${MOD_STREAM:-0}" ;;
      apply)    mod_apply ;;
      prerun)   if declare -F mod_prerun >/dev/null; then mod_prerun; fi ;;
      verify)   mod_verify ;;
      describe) mod_describe ;;
    esac )
}

# Ordered list of mod directories (numeric prefixes control order; _lib skipped).
_list_mods() {
  [[ -d "$MODS_DIR" ]] || return 0
  find "$MODS_DIR" -mindepth 2 -maxdepth 2 -name run.sh -printf '%h\n' 2>/dev/null \
    | grep -v '/_' | sort
}

# Reads KEY=value lines a mod wrote via mod_export into our own scope.
# STATUS= is a reserved key handled separately by _invoke_mod and skipped
# here. Only ALLOWLISTED keys are imported: mods are sourced shell, and an
# open declare -g channel would let any mod typo (or a third-party mod)
# silently overwrite main-script globals like INSTALL_DIR or PATH mid-run.
# Adding a new exported key = extend this pattern (deliberate, one line).
_export_mod_state() {
  local line
  while IFS= read -r line; do
    [[ "$line" =~ ^(SAGE_ACTION|ORT_STATE)= ]] || continue
    declare -g "${line%%=*}=${line#*=}"
  done < "$1"
}

# Invoke <verb> ('apply' or 'prerun') on one mod, honoring its declared
# MOD_CRITICAL/MOD_STREAM flags. Non-streamed mods behave exactly as before:
# buffered output, failures become "skipped:error", status is the first
# token of the mod's echoed output. Streamed mods inherit stdout/stderr live
# (needed for a 10-30 min build) and report their status via a state-file
# STATUS= line instead of an echoed one, since stdout is no longer captured.
# Either way, a MOD_CRITICAL mod's failure aborts the whole script instead
# of being swallowed — for steps whose failure means the install/launch is
# genuinely broken (torch has no CUDA, the GPU kernel doesn't work), not for
# optional source patches.
# Sets: MOD_LAST_NAME, MOD_LAST_STATUS.
_invoke_mod() {
  local mod_dir="$1" verb="$2" flags critical stream state_file crit_kv stream_kv
  MOD_LAST_NAME="$(basename "$mod_dir")"
  flags="$(_run_mod "$mod_dir" flags 2>/dev/null || echo "MOD_CRITICAL=0 MOD_STREAM=0")"
  read -r crit_kv stream_kv <<< "$flags"
  critical="${crit_kv#MOD_CRITICAL=}"; stream="${stream_kv#MOD_STREAM=}"
  # Every mod gets a state file to optionally mod_export extra KEY=value
  # pairs into our scope (e.g. 50-onnxruntime-gpu's ORT_STATE), regardless of
  # whether it streams — only the STATUS-reporting convention differs below.
  state_file="$(mktemp)"

  if [[ "$stream" == "1" ]]; then
    if MOD_STATE_FILE="$state_file" _run_mod "$mod_dir" "$verb"; then
      MOD_LAST_STATUS="$(grep '^STATUS=' "$state_file" | tail -1 | cut -d= -f2- || true)"
      MOD_LAST_STATUS="${MOD_LAST_STATUS:-applied}"
      _export_mod_state "$state_file"
    else
      rm -f "$state_file"
      if [[ "$critical" == "1" ]]; then
        die "mod '$MOD_LAST_NAME' failed (critical) — see the output above for
details. This step is required; fix the underlying issue and re-run: $0 $CMD"
      fi
      MOD_LAST_STATUS="skipped:error"
      return 0
    fi
  else
    if MOD_LAST_STATUS="$(MOD_STATE_FILE="$state_file" _run_mod "$mod_dir" "$verb" 2>/dev/null)"; then
      _export_mod_state "$state_file"
    else
      rm -f "$state_file"
      if [[ "$critical" == "1" ]]; then
        die "mod '$MOD_LAST_NAME' failed (critical) — its output was suppressed
(buffered mode). Fix the underlying issue, then re-run: $0 $CMD"
      fi
      MOD_LAST_STATUS="skipped:error"
      return 0
    fi
  fi
  rm -f "$state_file"
}

apply_source_patches() {
  SOURCE_PATCH_STATE=""
  if [[ "${SPARK_SOURCE_PATCHES:-1}" != "1" ]]; then
    SOURCE_PATCH_STATE="disabled (SPARK_SOURCE_PATCHES=0)"
    echo "Source mods: $SOURCE_PATCH_STATE"; return 0
  fi
  local mods; mods="$(_list_mods)"
  if [[ -z "$mods" ]]; then
    SOURCE_PATCH_STATE="no mods found"; echo "Source mods: none"; return 0
  fi
  log "Applying mods (idempotent, self-healing)"
  local applied=() present=() skipped=() name status verb detail
  while IFS= read -r mod_dir; do
    [[ -n "$mod_dir" ]] || continue
    _invoke_mod "$mod_dir" apply
    name="$MOD_LAST_NAME"; status="$MOD_LAST_STATUS"
    # Status protocol: first token is the class (applied|present|skipped),
    # optional remainder is human detail. Lets config mods report what changed.
    verb="${status%%[: ]*}"; detail="${status#"$verb"}"; detail="${detail#[: ]}"
    case "$verb" in
      applied)  applied+=("$name");  echo "  + $name${detail:+ — $detail}" ;;
      present)  present+=("$name");  echo "  = $name (already active)${detail:+ — $detail}" ;;
      *)        skipped+=("$name:${detail:-$status}"); warn "  ! $name — ${detail:-$status}" ;;
    esac
  done <<< "$mods"
  local parts=()
  [[ ${#applied[@]} -gt 0 ]] && parts+=("applied: ${applied[*]}")
  [[ ${#present[@]} -gt 0 ]] && parts+=("active: ${present[*]}")
  [[ ${#skipped[@]} -gt 0 ]] && parts+=("SKIPPED: ${skipped[*]}")
  local IFS=' | '; SOURCE_PATCH_STATE="${parts[*]}"
  [[ -z "$SOURCE_PATCH_STATE" ]] && SOURCE_PATCH_STATE="nothing to do"
  if [[ ${#skipped[@]} -gt 0 ]]; then
    warn "a mod could not be applied — upstream may have changed the code it
targets, or a step failed. Generation still works; that mod's optimization is
simply inactive until resolved. Details: $0 doctor"
  fi
}

# Runs before every 'run' (not just install/update): invokes mod_prerun on
# mods that define it (currently only 20-torch-repair, ComfyUI's own
# pre-launch guard against a custom node silently swapping in CPU torch).
# Mods without mod_prerun are no-ops here. Silent when nothing needs fixing.
apply_prerun_mods() {
  [[ "${SPARK_SOURCE_PATCHES:-1}" == "1" ]] || return 0
  local mods; mods="$(_list_mods)"
  [[ -z "$mods" ]] && return 0
  while IFS= read -r mod_dir; do
    [[ -n "$mod_dir" ]] || continue
    _invoke_mod "$mod_dir" prerun
  done <<< "$mods"
}


install_self() {
  # The script anchors all paths to its own location (BASE_DIR), so wherever
  # it lives IS the install root — no need to copy it elsewhere. Just make
  # sure it's executable so the systemd service and cron can invoke it.
  chmod +x "$SELF" 2>/dev/null || true
}

# =============================================================================
#  install
# =============================================================================
cmd_install() {
  local with_service=0
  for arg in "$@"; do
    case "$arg" in
      --with-service) with_service=1 ;;
      *) die "Unknown install option: $arg" ;;
    esac
  done

  log "Preflight checks"
  [[ "$(uname -m)" == "aarch64" ]] \
    || die "This script targets DGX Spark (aarch64). Detected: $(uname -m)"
  command -v nvidia-smi >/dev/null 2>&1 \
    || die "nvidia-smi not found. Install/repair the NVIDIA driver first."
  local gpu_name
  gpu_name="$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1 || true)"
  echo "GPU detected: ${gpu_name:-unknown}"
  [[ "$gpu_name" == *GB10* ]] \
    || warn "GPU is not reporting as GB10 — continuing, but tuning targets DGX Spark."

  # python3-dev matters: without matching dev headers, SageAttention cannot
  # compile and would silently fall back to slower attention. Only prompt
  # for sudo if something's actually missing — DGX OS ships much of this
  # already, and a re-run on an already-provisioned box shouldn't ask for
  # a password it doesn't need.
  local sys_pkgs=(git python3-venv python3-dev python3-pip) need_pkgs=() pkg
  for pkg in "${sys_pkgs[@]}"; do
    dpkg -s "$pkg" >/dev/null 2>&1 || need_pkgs+=("$pkg")
  done
  if [[ ${#need_pkgs[@]} -gt 0 ]]; then
    log "Missing: ${need_pkgs[*]} — refreshing apt's package index (sudo apt-get
update), then installing just these and their own dependencies (sudo apt-get
install). No dist-upgrade, no upgrade of anything already installed."
    sudo apt-get update -qq
    sudo apt-get install -y "${need_pkgs[@]}"
  else
    log "System packages already present (${sys_pkgs[*]}) — no sudo needed"
  fi

  log "Creating virtual environment at $VENV_DIR"
  python3 -m venv "$VENV_DIR"
  activate_venv
  pip install --upgrade pip wheel >/dev/null

  # GB10 is Blackwell sm_121: cu130 wheels ship sm_120 + PTX that JITs to
  # sm_121, so CUDA 13.0 wheels are the supported path. Must install BEFORE
  # ComfyUI's requirements so nothing pulls a CPU-only torch first.
  log "Installing PyTorch (CUDA 13.0 aarch64 wheels)"
  pip install torch torchvision torchaudio --index-url "$TORCH_INDEX"

  if [[ -d "$INSTALL_DIR/.git" ]]; then
    log "Updating existing ComfyUI at $INSTALL_DIR"
  else
    log "Cloning ComfyUI into $INSTALL_DIR"
    git clone "$REPO_URL" "$INSTALL_DIR"
  fi
  sync_comfyui
  apply_patches

  log "Installing ComfyUI dependencies (requirements.txt) into the venv"
  pip install -r "$INSTALL_DIR/requirements.txt"

  if [[ -f "$INSTALL_DIR/manager_requirements.txt" ]]; then
    log "Installing ComfyUI-Manager dependencies (manager_requirements.txt)"
    pip install -r "$INSTALL_DIR/manager_requirements.txt"
  else
    die "manager_requirements.txt not found in this checkout — the built-in
Manager isn't bundled here. If REPO_URL points at a fork or branch, confirm
it ships ComfyUI-Manager; the stock upstream repo always does."
  fi

  # setuptools pin, torch CUDA verification, SageAttention build+verify,
  # GPU onnxruntime, unified-memory source patches, Manager config — all
  # applied here, in order, self-healing on re-run. See mods/README.md.
  apply_source_patches
  install_self

  log "GPU sanity check"
  python - <<'PY'
import torch
print(f"torch          : {torch.__version__}")
print(f"CUDA (compiled): {torch.version.cuda}")
print(f"Device         : {torch.cuda.get_device_name(0)}")
cap = torch.cuda.get_device_capability(0)
print(f"Capability     : sm_{cap[0]}{cap[1]}")
x = torch.randn(1024, 1024, device="cuda"); y = x @ x
torch.cuda.synchronize(); print("Matmul on GPU  : OK")
PY
  echo
  echo "Note: a warning that capability 12.1 exceeds torch's supported maximum"
  echo "is expected on GB10 and safe to ignore (PTX JITs to sm_121)."

  [[ "$with_service" -eq 1 ]] && cmd_service

  local ip_hint
  ip_hint="$(hostname -I 2>/dev/null | awk '{print $1}')"
  log "Done!"
  cat <<EOF

  Install root:     $BASE_DIR
  Start ComfyUI:    $SELF run
  Health check:     $SELF doctor
  Update later:     $SELF update
  Web UI:           http://${ip_hint:-<spark-ip>}:$PORT
  Models go in:     $INSTALL_DIR/models/checkpoints (etc.)

  DGX Spark notes:
   * If a custom node's requirements replace torch with a CPU build, 'run'
     and 'doctor' will catch it; 'update' repairs it automatically.
   * "GPU OOM" here is system OOM (unified memory). If huge video models make
     the box unresponsive, run: $SELF tune   (disables swap, among other fixes)
   * network_mode = personal_cloud relaxes Manager's security gating so it
     works while serving on 0.0.0.0 — fine on a trusted LAN, but do NOT
     expose port $PORT directly to the internet.
   * Flash Attention is deliberately not installed. FA3 can't target sm_121
     at all; FA2 2.8.3 CAN be compiled from source (TORCH_CUDA_ARCH_LIST=
     "12.0", ~2h build) but SDPA is faster on Blackwell and SageAttention
     covers the rest — the only reason to build FA2 is a custom node that
     hard-imports flash_attn. Ask for it then; skip it otherwise.
EOF
}

# =============================================================================
#  run
# =============================================================================
cmd_run() {
  activate_venv
  cd "$INSTALL_DIR"

  # Quick guard before every launch: catch clobbered torch early
  # (20-torch-repair's mod_prerun; other mods have no prerun and no-op here).
  apply_prerun_mods

  # Field-validated GB10 environment (NVIDIA forum, Feb 2026):
  #  * 4 GB CUDA kernel cache: ~3x faster denoise steps on reruns
  #    (first run compiles PTX->SASS for sm_121, then reuses from disk)
  #  * NCCL P2P off: single GPU, skip the overhead
  #  * TRITON_PTXAS_PATH: torch's bundled ptxas is pre-CUDA-13 and can't
  #    target sm_121, so any custom node calling raw triton.jit() fails with
  #    'no kernel image' unless Triton is pointed at the system CUDA 13
  #    ptxas (triton-lang/triton#10331 — same root cause the SageAttention
  #    build already handles for itself in mod_common.sh).
  #  * Deliberately NOT set (measured harmful on GB10):
  #      TORCH_INDUCTOR_FX_GRAPH_CACHE (stale graphs -> wrong output)
  #      PYTORCH_NO_CUDA_MEMORY_CACHING (fragmentation -> OOM)
  #      torch.compile paths (≈0% gain; GPU is compute-bound)
  export CUDA_CACHE_MAXSIZE="${CUDA_CACHE_MAXSIZE:-4294967296}"
  export NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE:-1}"
  [[ -x /usr/local/cuda/bin/ptxas ]] \
    && export TRITON_PTXAS_PATH="${TRITON_PTXAS_PATH:-/usr/local/cuda/bin/ptxas}"

  # Swap + unified memory = silent system freeze under heavy video loads.
  if [[ -n "$(swapon --noheadings 2>/dev/null)" ]]; then
    warn "swap is ENABLED — heavy workloads can silently freeze the box.
Strongly recommended: $0 tune (disables swap, sets persistence mode)"
  fi

  # DGX Spark tuning:
  #  * Unified memory: don't force everything GPU-side (--gpu-only /
  #    --highvram / --cache-none) — async offload is nearly free here.
  #  * --disable-pinned-memory reduces overhead on the unified fabric.
  #  * bf16 for unet/vae/text-enc is the native fast path on GB10
  #    (opt out with SPARK_BF16=0 if a model misbehaves).
  #  * SageAttention only enabled if it passed the live kernel verification
  #    (marker). To A/B against PyTorch attention: rm $SAGE_MARKER
  local extra_flags=()
  if [[ "${SPARK_BF16:-1}" == "1" ]]; then
    extra_flags+=(--bf16-unet --bf16-vae --bf16-text-enc)
  fi
  # Opt-in: keep models resident when they fit (faster prompt->image
  # iteration on the 128 GB pool). Enable with SPARK_STATIC_VRAM=1.
  if [[ "${SPARK_STATIC_VRAM:-0}" == "1" ]]; then
    extra_flags+=(--disable-dynamic-vram)
  fi
  if [[ -f "$SAGE_MARKER" ]]; then
    if sage_kernel_ok; then
      extra_flags+=(--use-sage-attention)
      info "SageAttention enabled (kernel re-checked OK)"
    else
      # Marker says verified but the live kernel is broken -> something
      # overwrote the sm_121 build (pip shadowing). Rebuild, don't degrade.
      warn "SageAttention was verified but its kernel now FAILS — likely a pip
install overwrote the sm_121 build. Rebuilding before launch..."
      build_and_verify_sage
      extra_flags+=(--use-sage-attention)
    fi
  else
    warn "SageAttention not verified — launching with PyTorch attention.
Run '$0 doctor' to diagnose, or '$0 update' to rebuild."
  fi

  exec python main.py \
    --listen 0.0.0.0 \
    --port "$PORT" \
    --enable-manager \
    --preview-method auto \
    --disable-pinned-memory \
    "${extra_flags[@]}" \
    "$@"
}

# =============================================================================
#  update
# =============================================================================
cmd_update() {
  local upgrade_torch=0
  for arg in "$@"; do
    case "$arg" in
      --torch)    upgrade_torch=1 ;;
      --rollback) cmd_rollback; return ;;
      *) die "Unknown update option: $arg (use --torch or --rollback)" ;;
    esac
  done
  activate_venv

  log "Checking ComfyUI for updates"
  sync_comfyui
  apply_patches
  if [[ "$COMFY_MOVED" -eq 1 || "$PATCHES_ACTIVE" -eq 1 ]]; then
    log "Refreshing python dependencies"
    pip install -r "$INSTALL_DIR/requirements.txt"
    [[ -f "$INSTALL_DIR/manager_requirements.txt" ]] \
      && pip install -r "$INSTALL_DIR/manager_requirements.txt"
  fi

  if [[ "$upgrade_torch" -eq 1 ]]; then
    log "Upgrading PyTorch (cu130) — SageAttention will be rebuilt"
    pip install --upgrade torch torchvision torchaudio --index-url "$TORCH_INDEX"
    rm -f "$SAGE_MARKER"   # torch ABI may have changed; force rebuild
  fi

  # setuptools/torch repair, SageAttention rebuild-if-needed (its own
  # marker/git-rev-drift check, plus SPARK_TORCH_UPGRADED as an extra forced-
  # rebuild trigger below), onnxruntime re-assert, source patches, and
  # Manager config all happen here, in order, self-healing. See mods/README.md.
  export SPARK_TORCH_UPGRADED="$upgrade_torch"
  apply_source_patches
  install_self

  # ---------------------------- Update summary ------------------------------
  local torch_ver
  torch_ver="$(python -c 'import torch; print(torch.__version__)' 2>/dev/null || echo '?')"
  log "Update summary"
  printf '  %-15s %s\n' "ComfyUI:"       "$([[ "$COMFY_MOVED" -eq 1 ]] && echo "updated -> $(git -C "$INSTALL_DIR" rev-parse --short HEAD)" || echo "current ($(git -C "$INSTALL_DIR" rev-parse --short HEAD))")"
  printf '  %-15s %s\n' "Patches:"       "$([[ "${PATCHES_ACTIVE:-0}" -eq 1 ]] && echo "active on branch '$PATCH_BRANCH'" || echo "none")"
  printf '  %-15s %s\n' "SageAttention:" "${SAGE_ACTION:-unknown}"
  printf '  %-15s %s\n' "Mods:"          "${SOURCE_PATCH_STATE:-n/a}"
  printf '  %-15s %s\n' "torch:"         "$torch_ver (pins enforced by Manager)"
  printf '  %-15s %s\n' "onnxruntime:"   "${ORT_STATE:-unknown}"
  echo
  if [[ "$COMFY_MOVED" -eq 1 || "${PATCHES_ACTIVE:-0}" -eq 1 || "${SAGE_ACTION:-}" == rebuilt* || "$upgrade_torch" -eq 1 ]]; then
    echo "Changes applied — restart to pick them up: $0 run"
  else
    echo "Everything is current — nothing to do, no restart needed."
  fi
}

# =============================================================================
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
    local lgc_line=""
    [[ -n "$clock_cap" ]] && lgc_line="ExecStart=/usr/bin/nvidia-smi -lgc 300,$clock_cap"
    sudo tee /etc/systemd/system/comfyui-tune.service >/dev/null <<UNIT
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
cmd_status() {
  # --watch: log GPU temp/power/RAM every 5s (post-mortem evidence for
  # silent hard-reboots: a power spike right before death = overcurrent
  # -> fix with: tune --clock-cap 2100)
  if [[ "${1:-}" == "--watch" || "${1:-}" == "-w" ]]; then
    local logfile="$BASE_DIR/thermal_monitor.log"
    log "Logging every 5s to $logfile (Ctrl-C to stop)"
    while true; do
      echo "$(date +%H:%M:%S) GPU=$(nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader,nounits)°C \
PWR=$(nvidia-smi --query-gpu=power.draw --format=csv,noheader,nounits)W \
RAM=$(free -g | awk '/Mem:/{print $3"/"$2}')G \
SWAP=$(free -g | awk '/Swap:/{print $3}')G" | tee -a "$logfile"
      sleep 5
    done
  fi

  hdr "Process"
  if pgrep -f "main.py --listen" >/dev/null 2>&1; then
    local pid; pid="$(pgrep -f 'main.py --listen' | head -1)"
    echo "  ComfyUI RUNNING (pid $pid) -> http://$(hostname -I 2>/dev/null | awk '{print $1}'):$PORT"
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
  if [[ -d "$INSTALL_DIR/.git" ]]; then
    echo "  ComfyUI: $(git -C "$INSTALL_DIR" log -1 --format='%h %cd %s' --date=short) [branch: $(git -C "$INSTALL_DIR" rev-parse --abbrev-ref HEAD)]"
    if [[ -f "$PATCH_LIST" ]] && grep -qE '^[^#[:space:]]' "$PATCH_LIST"; then
      echo "  patch list: $(grep -cE '^[^#[:space:]]' "$PATCH_LIST") entries in $PATCH_LIST"
    fi
  fi
  if [[ -f "$VENV_DIR/bin/activate" ]]; then
    # shellcheck disable=SC1091
    source "$VENV_DIR/bin/activate"
    python -c "import torch; print(f'  torch: {torch.__version__} (CUDA {torch.version.cuda})')" 2>/dev/null
    [[ -f "$SAGE_MARKER" ]] && echo "  SageAttention: verified" || echo "  SageAttention: NOT verified"
  fi
  local cfg="$INSTALL_DIR/user/__manager/config.ini"
  [[ -f "$cfg" ]] && grep -q 'network_mode *= *personal_cloud' "$cfg" \
    && echo "  Manager: network_mode = personal_cloud" \
    || echo "  Manager: personal_cloud NOT set"
}

# =============================================================================
#  stop / rollback
# =============================================================================
cmd_stop() {
  if systemctl --user is-active comfyui.service >/dev/null 2>&1; then
    systemctl --user stop comfyui.service
    echo "systemd service stopped"
  elif pgrep -f "main.py --listen" >/dev/null 2>&1; then
    pkill -f "main.py --listen"
    echo "ComfyUI process stopped"
  else
    echo "ComfyUI is not running"
  fi
}

cmd_rollback() {
  local rev_file="$VENV_DIR/.last_comfyui_rev"
  [[ -f "$rev_file" ]] || die "No previous revision recorded — rollback is only
available after '$0 update' has moved ComfyUI forward at least once."
  activate_venv
  local rev; rev="$(cat "$rev_file")"
  log "Rolling ComfyUI back to ${rev:0:8}"
  git -C "$INSTALL_DIR" reset --hard "$rev"
  pip install -r "$INSTALL_DIR/requirements.txt"
  [[ -f "$INSTALL_DIR/manager_requirements.txt" ]] \
    && pip install -r "$INSTALL_DIR/manager_requirements.txt"
  repair_torch
  rm -f "$rev_file"
  log "Rolled back. Restart with: $0 run"
}


# =============================================================================
#  doctor — diagnose the silent-drift failure modes specific to GB10
# =============================================================================
#  These are the traps that pass a "successful install" but break at runtime.
#  Each check names the exact fix. Sources: community GB10 field guides.
cmd_doctor() {
  PASS=0; FAIL=0
  activate_venv

  hdr "PyTorch / GPU (CPU-shadow check)"
  if python - <<'PY'
import torch
print(f"  torch {torch.__version__} | compiled CUDA {torch.version.cuda}")
assert (torch.version.cuda or "").startswith("13")
assert torch.cuda.is_available()
cap = torch.cuda.get_device_capability(0)
print(f"  device: {torch.cuda.get_device_name(0)} | sm_{cap[0]}{cap[1]}")
PY
  then
    ok "torch is the cu130 CUDA build and sees the GPU"
  else
    bad "torch is CPU-only or wrong CUDA — a custom node likely re-pinned it.
        Fix: $0 update (repairs torch automatically)"
  fi

  # Whole-venv dependency graph: catches pin violations generically (e.g.
  # setuptools upgraded past torch's <82 pin, conflicting custom-node deps).
  # Filter known-benign noise: on aarch64, some nvidia-*-cu13 wheels declare
  # platform metadata pip misreads as "not supported on this platform" even
  # though torch installed them deliberately and they work — not a real
  # conflict, and 'update' cannot fix it, so surfacing it just misleads.
  local pipcheck real_issues
  pipcheck="$(pip check 2>&1 || true)"
  real_issues="$(echo "$pipcheck" \
    | grep -vE 'is not supported on this platform' \
    | grep -vE '^No broken requirements found' \
    | grep -E '.' || true)"
  if [[ -z "$real_issues" ]]; then
    ok "pip dependency graph is consistent"
    echo "$pipcheck" | grep -q 'not supported on this platform' \
      && info "(ignored benign aarch64 platform-metadata notice for nvidia-*-cu13)"
  else
    bad "pip reports dependency conflicts:"
    echo "$real_issues" | head -5 | sed 's/^/        /'
    echo "        Likely fix: $0 update (realigns torch-pinned deps)"
  fi

  hdr "SageAttention (pip-shadow + kernel-image check)"
  if [[ -f "$SAGE_MARKER" ]]; then
    ok "install-time verification marker present"
  else
    bad "marker missing — 'run' will NOT pass --use-sage-attention. Fix: $0 update"
  fi
  # The nastiest GB10 trap: a wheel silently replaced the local sm_121 build.
  if ! python -c "import sageattention" 2>/dev/null; then
    bad "sageattention not importable — Fix: $0 update"
  elif sage_kernel_ok; then
    ok "live sm_121 kernel runs (local build intact, not shadowed)"
  else
    bad "IMPORTS but kernel FAILS — the classic 'pip install overwrote the
        sm_121 build' shadow. Fix: $0 update (rebuilds from source)"
  fi
  # Inspect the compiled extension: GB10 needs a native sm_121 cubin OR PTX
  # to JIT. sm_120 cubin WITHOUT PTX is the exact 'no kernel image' config.
  local so_file
  so_file="$(python - <<'PY' 2>/dev/null
import glob, os, sageattention
d = os.path.dirname(sageattention.__file__)
hits = glob.glob(os.path.join(d, "**", "*.so"), recursive=True) \
     + glob.glob(os.path.join(os.path.dirname(d), "*sage*", "**", "*.so"), recursive=True) \
     + glob.glob(os.path.join(os.path.dirname(d), "*sage*.so"))
print(hits[0] if hits else "")
PY
)"
  if [[ -n "$so_file" ]] && { command -v cuobjdump >/dev/null 2>&1 || [[ -x /usr/local/cuda/bin/cuobjdump ]]; }; then
    command -v cuobjdump >/dev/null 2>&1 || export PATH="/usr/local/cuda/bin:$PATH"
    local archs ptx
    archs="$(cuobjdump --list-elf "$so_file" 2>/dev/null | grep -o 'sm_[0-9]*' | sort -u | tr '\n' ' ')"
    ptx="$(cuobjdump --list-ptx "$so_file" 2>/dev/null | grep -o 'sm_[0-9]*' | sort -u | tr '\n' ' ')"
    info "embedded cubin: ${archs:-none}    embedded PTX: ${ptx:-none}"
    if [[ "$archs" == *sm_121* ]] || [[ -n "$ptx" ]]; then
      ok "extension has an sm_121 path (native cubin and/or PTX fallback)"
    else
      bad "no sm_121 cubin and no PTX — 'no kernel image' config. Fix: $0 update"
    fi
  fi
  local sage_origin
  sage_origin="$(python - <<'PY' 2>/dev/null
import importlib.metadata as m
try:
    d = m.distribution("sageattention")
    url = (d.read_text("direct_url.json") or "")
    print("local" if ("file://" in url or not url) else "pypi")
except Exception:
    print("unknown")
PY
)"
  [[ "$sage_origin" == "pypi" ]] \
    && warn "sageattention appears to come from a PyPI wheel, not your local
build — high shadow risk. Re-run: $0 update" \
    || info "sageattention distribution origin: $sage_origin"
  # Triton JITs a small C shim per kernel launch; without the dev headers for
  # the venv's exact Python it fails on EVERY call and ComfyUI silently uses
  # PyTorch attention instead — up to ~18x slower, no visible error (field
  # report: NVIDIA forum #375830). The kernel test above passes regardless,
  # so check the headers and the runtime log separately.
  local pyminor devpkg
  pyminor="$(python -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")' 2>/dev/null)"
  devpkg="python${pyminor}-dev"
  if [[ -n "$pyminor" ]] && dpkg -s "$devpkg" >/dev/null 2>&1; then
    ok "$devpkg present (Triton can JIT — no silent per-call fallback)"
  elif [[ -n "$pyminor" ]]; then
    bad "$devpkg MISSING — Triton can't compile its JIT shim, so SageAttention
        fails per-call and ComfyUI silently uses PyTorch attention (~18x
        slower). Fix: sudo apt-get install -y $devpkg   then restart"
  fi
  # And check the actual runtime evidence in ComfyUI's own log. The live log
  # is port-suffixed (user/comfyui_<PORT>.log) despite the startup banner
  # claiming comfyui.log — check both, current session only (not .prev).
  local comfy_log fb_total fb_benign t b
  fb_total=0; fb_benign=0
  for comfy_log in "$INSTALL_DIR/user/comfyui_${PORT}.log" "$INSTALL_DIR/user/comfyui.log"; do
    [[ -f "$comfy_log" ]] || continue
    read -r t b <<< "$(sage_fallback_counts "$comfy_log")"
    fb_total=$((fb_total + t)); fb_benign=$((fb_benign + b))
  done
  if [[ -f "$INSTALL_DIR/user/comfyui_${PORT}.log" || -f "$INSTALL_DIR/user/comfyui.log" ]]; then
    if [[ "$fb_total" -eq 0 ]]; then
      ok "no runtime SageAttention fallbacks in ComfyUI's log"
    elif [[ "$fb_total" -eq "$fb_benign" ]]; then
      info "log shows $fb_benign 'Unsupported head_dim' fallback(s) — benign:"
      info "  that model's attention exceeds SageAttention's head_dim<=128 limit;"
      info "  ComfyUI correctly used PyTorch attention for those layers"
    else
      bad "$((fb_total - fb_benign)) real SageAttention runtime failure(s) in
        ComfyUI's log (user/comfyui_${PORT}.log) — Sage is silently falling
        back to slow PyTorch attention. Check the log's exception text, fix,
        then restart: $0 run"
    fi
  fi

  hdr "onnxruntime (preprocessor GPU check)"
  # DWPose/ControlNet preprocessors silently run on CPU if the sm_121 GPU
  # wheel was never installed or got shadowed by a PyPI 'onnxruntime' dist.
  if ! python -c "import onnxruntime" 2>/dev/null; then
    info "onnxruntime not installed (only needed by DWPose/ControlNet preprocessors)"
  elif onnx_gpu_ok; then
    ok "CUDAExecutionProvider live — preprocessors run on GPU"
  else
    bad "onnxruntime is CPU-ONLY — a PyPI wheel likely shadowed the sm_121 GPU
        wheel (shared import path, no pip conflict). Fix: $0 update"
  fi

  hdr "NVRTC (GPU-FFT custom-node check)"
  # The failure mode: a STALE BUNDLED NVRTC in the torch wheel shadowing the
  # system CUDA 13 one. No bundled copy at all (typical for aarch64 cu130
  # wheels) is the GOOD case — torch then resolves the system libnvrtc.
  local nvrtc_bundled sys_nvrtc
  nvrtc_bundled="$(python - <<'PY' 2>/dev/null
import glob, os, torch
tdir = os.path.dirname(torch.__file__)
sp = os.path.dirname(tdir)
hits = glob.glob(os.path.join(sp, "nvidia", "cuda_nvrtc", "lib", "libnvrtc.so*")) \
     + glob.glob(os.path.join(tdir, "lib", "libnvrtc*.so*"))
print(";".join(sorted({os.path.basename(h) for h in hits})) or "none")
PY
)"
  if [[ "$nvrtc_bundled" == *.13* || "$nvrtc_bundled" == *so.13* ]]; then
    ok "torch bundles a CUDA 13 NVRTC ($nvrtc_bundled)"
  elif [[ "$nvrtc_bundled" == "none" ]]; then
    sys_nvrtc="$(ldconfig -p 2>/dev/null | grep -o 'libnvrtc\.so\.[0-9]*' | sort -u | tr '\n' ' ')"
    if [[ "$sys_nvrtc" == *so.13* ]]; then
      ok "no bundled NVRTC — torch uses the system CUDA 13 one ($sys_nvrtc)"
    else
      bad "no bundled NVRTC and no system libnvrtc.so.13 (found: ${sys_nvrtc:-none}).
        GPU-FFT custom nodes will fail. Install the CUDA 13 runtime libs."
    fi
  else
    warn "bundled NVRTC looks pre-13 ($nvrtc_bundled) — this can shadow the
system CUDA 13 one and crash GPU-FFT custom nodes. If that happens,
symlink the system libnvrtc over the bundled copy."
  fi

  hdr "ptxas (sm_121 capability)"
  if command -v ptxas >/dev/null 2>&1 || [[ -x /usr/local/cuda/bin/ptxas ]]; then
    command -v ptxas >/dev/null 2>&1 || export PATH="/usr/local/cuda/bin:$PATH"
    if ptxas_ge_13; then
      ok "ptxas is CUDA >= 13.0 — sm_121-capable ($(ptxas --version 2>/dev/null | grep -o 'release [0-9.]*' | head -1))"
    else
      bad "ptxas on PATH is older than CUDA 13.0 — cannot target sm_121. This
        causes 'no kernel image'. Fix: put CUDA 13 first on PATH, then $0 update"
    fi
  else
    info "ptxas not found on PATH (only needed when rebuilding kernels)"
  fi

  hdr "Runtime (is the optimization actually active?)"
  if pgrep -f "main.py --listen" >/dev/null 2>&1; then
    if pgrep -af "main.py --listen" | grep -q "use-sage-attention"; then
      ok "running ComfyUI was launched WITH --use-sage-attention"
    else
      bad "ComfyUI is running WITHOUT --use-sage-attention — restart: $0 run"
    fi
  else
    info "ComfyUI not running. After '$0 run', its startup log must say:"
    info "  'Using sage attention'   <- definitive runtime confirmation"
  fi

  # (ComfyUI-Manager config is verified by mod 30-manager-config below.)

  hdr "GPU clocks (stuck-low check)"
  # Field-reported GB10 failure mode (NVIDIA forums, Feb-Jul 2026, multiple
  # independent units): after a prior OOM/power event, SM clocks pin at
  # 513-721 MHz with NO throttle reason and normal temps — invisible to
  # telemetry, not clearable via nvidia-smi; only a full power cycle fixes
  # it. Clocks idle low BY DESIGN, so only a reading under real load means
  # anything: spin a short matmul and sample the max the GPU reaches.
  local max_sm=0 clk spin_pid throttle_mask
  python - <<'PY' >/dev/null 2>&1 &
import time
import torch
x = torch.randn(4096, 4096, device="cuda")
t0 = time.time()
while time.time() - t0 < 2.5:
    x = x @ x
    torch.cuda.synchronize()
PY
  spin_pid=$!
  for _ in 1 2 3 4 5 6 7 8; do
    sleep 0.25
    clk="$(nvidia-smi --query-gpu=clocks.sm --format=csv,noheader,nounits 2>/dev/null | head -1)"
    if [[ "$clk" =~ ^[0-9]+$ ]] && (( clk > max_sm )); then max_sm=$clk; fi
  done
  wait "$spin_pid" 2>/dev/null || true
  if (( max_sm == 0 )); then
    info "could not sample SM clocks under load — skipping stuck-clock check"
  elif (( max_sm < 900 )); then
    throttle_mask="$(nvidia-smi --query-gpu=clocks_throttle_reasons.active --format=csv,noheader 2>/dev/null | head -1)"
    bad "SM clock only reached ${max_sm} MHz under load (throttle reasons:
        ${throttle_mask:-unreadable}) — matches the known GB10 stuck-clock
        state after a prior OOM/power event. nvidia-smi cannot clear it.
        Fix: FULL power cycle — shut down, unplug, wait ~10s, reconnect.
        (Ignore this if you deliberately capped clocks below 900 MHz.)"
  else
    ok "SM clock reached ${max_sm} MHz under load — no stuck-clock state"
  fi

  hdr "Driver / CUDA stack (informational)"
  # Driver & CUDA updates have shipped real GB10 perf gains (e.g. CUDA 13.0u2
  # sped up FP16/BF16/FP8 GEMMs in cuBLAS — exactly what diffusion spends its
  # time on). But upgrades belong in the DGX Dashboard, not this script:
  # NVIDIA's tested path, and driver swaps need a reboot.
  local drv cuda_drv nvcc_ver upgradable
  drv="$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -1)"
  cuda_drv="$(nvidia-smi 2>/dev/null | grep -o 'CUDA Version: [0-9.]*' | awk '{print $3}')"
  nvcc_ver="$( (command -v nvcc >/dev/null 2>&1 && nvcc --version || /usr/local/cuda/bin/nvcc --version 2>/dev/null) | grep -o 'release [0-9.]*' | awk '{print $2}' )"
  info "driver: ${drv:-?}   CUDA (driver): ${cuda_drv:-?}   toolkit (nvcc): ${nvcc_ver:-n/a}"
  upgradable="$(apt list --upgradable 2>/dev/null | grep -Ec '^(nvidia|cuda|libnvidia)' || true)"
  if [[ "${upgradable:-0}" -gt 0 ]]; then
    warn "$upgradable NVIDIA/CUDA packages have pending updates (per last 'apt update').
Updates regularly carry GB10 performance gains — upgrade via the DGX
Dashboard (NVIDIA's recommended path), reboot, then run:
  $0 update && $0 doctor
(update rebuilds SageAttention if the new toolkit changed ptxas/ABI)"
  else
    info "no pending NVIDIA/CUDA apt updates (refresh with: sudo apt update)"
  fi

  hdr "Mods (GB10 fixes & config)"
  if [[ "${SPARK_SOURCE_PATCHES:-1}" != "1" ]]; then
    info "source mods disabled (SPARK_SOURCE_PATCHES=0)"
  else
    local mods; mods="$(_list_mods)"
    if [[ -z "$mods" ]]; then
      info "no mods present in ${MODS_DIR}"
    else
      local name desc
      while IFS= read -r mod_dir; do
        [[ -n "$mod_dir" ]] || continue
        name="$(basename "$mod_dir")"
        desc="$(_run_mod "$mod_dir" describe 2>/dev/null || echo "$name")"
        if _run_mod "$mod_dir" verify >/dev/null 2>&1; then
          ok "$name active — $desc"
        else
          bad "$name NOT active — $desc. Fix: $0 update"
        fi
      done <<< "$mods"
    fi
  fi

  hdr "Unified-memory safety"
  if [[ -z "$(swapon --noheadings 2>/dev/null)" ]]; then
    ok "swap disabled (clean OOM instead of silent freeze)"
  else
    bad "swap ENABLED — heavy loads can freeze the box. Fix: $0 tune"
  fi

  hdr "Summary"
  echo "  $PASS passed, $FAIL failed"
  [[ $FAIL -eq 0 ]] && echo "  No silent-drift issues detected." \
                    || echo "  Run the suggested fixes above, then re-run: $0 doctor"
  [[ $FAIL -eq 0 ]]
}


cmd_service() {
  install_self
  log "Installing systemd user service (comfyui.service)"
  mkdir -p "$HOME/.config/systemd/user"
  cat > "$HOME/.config/systemd/user/comfyui.service" <<UNIT
[Unit]
Description=ComfyUI (DGX Spark)
After=network-online.target

[Service]
ExecStart=$SELF run
Restart=on-failure
RestartSec=5

[Install]
WantedBy=default.target
UNIT
  systemctl --user daemon-reload
  systemctl --user enable --now comfyui.service
  loginctl enable-linger "$USER" || true
  echo "Service installed. Logs: journalctl --user -u comfyui -f"
}

# ------------------------------- Dispatch -----------------------------------
CMD="${1:-}"
shift || true
case "$CMD" in
  install)  cmd_install "$@" ;;
  run)      cmd_run "$@" ;;
  stop)     cmd_stop ;;
  update)   cmd_update "$@" ;;
  doctor)   cmd_doctor ;;
  status)   cmd_status "$@" ;;
  tune)     cmd_tune "$@" ;;
  service)  cmd_service ;;
  # --- hidden backward-compat aliases (old command names still work) ---
  verify)   cmd_doctor ;;
  monitor)  cmd_status --watch ;;
  rollback) cmd_rollback ;;
  bench)    die "'bench' was removed: synthetic attention numbers don't predict
real gains. A/B your actual workflow instead: time it, then
  rm $SAGE_MARKER && $0 stop && $0 run
re-time, and restore with: touch $SAGE_MARKER" ;;
  ""|-h|--help|help) usage ;;
  -v|--version|version) echo "spark-comfyui $VERSION" ;;
  *) die "Unknown command: $CMD (try: install | run | stop | update | doctor | status | tune | service)" ;;
esac
