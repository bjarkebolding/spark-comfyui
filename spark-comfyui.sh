#!/usr/bin/env bash
# =============================================================================
#  spark-comfyui.sh — ComfyUI on NVIDIA DGX Spark (GB10 Grace Blackwell)
#  Version 2026.07.19 | License: MIT
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
#                              Also self-updates spark-comfyui itself first
#                              (git fast-forward, only when its repo has
#                              newer commits; SPARK_SELF_UPDATE=0 disables).
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
#    status [--watch [SEC]]    One-page glance: process, GPU, memory, versions.
#                              --watch shows a live dashboard (sparkline
#                              timeseries: temp/power/clock/util/RAM/CPU,
#                              every 5s or SEC) and appends every sample to
#                              thermal_monitor.log — the evidence trail for
#                              diagnosing silent hard-reboots survives them.
#    tune [--clock-cap MHZ] [--persist]
#                              System stability: disable swap (prevents
#                              unified-memory freezes), persistence mode,
#                              optional clock cap (~2100 fixes overcurrent
#                              hard-reboots). --persist survives reboots.
#    backup [--with-output] [FILE]
#                              Archive the small precious state: workflows,
#                              settings, inputs, patch list, custom-node list,
#                              and a manifest of your models (listed, never
#                              copied — they are huge). --with-output also
#                              archives generated images. Safe while running.
#    restore FILE              Rebuild from a backup archive: installs first
#                              if needed, merges user state back, re-clones
#                              custom nodes, and lists which models you still
#                              need to fetch separately.
#    reset [--yes]             Delete and reinstall ComfyUI, the venv and
#                              SageAttention while keeping models, workflows,
#                              inputs, outputs and custom nodes. The nuclear
#                              option for when doctor's fixes don't stick.
#    service                   Install + start a systemd user service.
#    container build|run|stop|shell
#                              EXPERIMENTAL containerized runtime (the
#                              roadmap; the native path above becomes
#                              legacy once this matures). 'build' bakes an
#                              image (ComfyUI + venv + SageAttention + mods);
#                              'run' launches it with user content
#                              (models, workflows, custom nodes, outputs)
#                              bind-mounted from this install. Custom-node
#                              code runs confined: no host files beyond the
#                              mounts, dropped capabilities.
#
#  Typical day: install once -> run -> update now and then.
#  Something feels wrong? -> doctor tells you what and how to fix it.
#  Re-running install is safe: completed steps are skipped or refreshed.
# =============================================================================
set -euo pipefail

# Date versioning (CalVer): YYYY.MM.DD, with .N appended for a second
# behavior-changing release on the same day. Bumped in the same push as any
# behavior change (pushing to main IS releasing); docs-only pushes don't bump.
VERSION="2026.07.19"

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
# Community sm_121/aarch64/cu13 GPU onnxruntime (no official PyPI wheel
# exists). The #sha256= fragment pins the exact bytes: pip verifies it before
# installing, so a compromised or force-pushed hosting repo fails loudly
# instead of installing silently. Overriding ORT_WHEEL_URL replaces the pin
# too — re-add a fragment for your own wheel if you want the same guarantee.
ORT_WHEEL_URL="${ORT_WHEEL_URL:-https://huggingface.co/Jay0515/onnxruntime-gpu-aarch64-cuda13-sm121/resolve/main/onnxruntime_gpu-1.25.0-cp312-cp312-linux_aarch64.whl#sha256=da487cc1ccd3aa11389efec14c6f0f8b6bd7ca6734423de3b528e578023cb200}"
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

# Containerized runtime (EXPERIMENTAL, `container` subcommands): image and
# container name, both overridable.
CONTAINER_IMAGE="${CONTAINER_IMAGE:-spark-comfyui}"
CONTAINER_NAME="${CONTAINER_NAME:-spark-comfyui}"

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
  # Source-patch mods keep patched upstream files MODIFIED in this tree
  # (e.g. mod 10's get_free_memory edit in comfy/model_management.py). The
  # moment upstream touches such a file, git refuses the branch switch or
  # the ff-only merge below ("local changes would be overwritten") and the
  # whole update dies on a raw git error. Revert ONLY files carrying the
  # spark marker — the mods pass re-applies them right after this sync.
  # A user's own edits (no marker) are deliberately left alone.
  local pf
  while IFS= read -r pf; do
    # if-form, not `grep && checkout`: a trailing markerless file would make
    # the && list (and so the loop) return nonzero under set -e semantics
    # that differ across shells — the if makes "no marker, skip" explicit.
    if grep -qF "# spark-comfyui:" "$pf" 2>/dev/null; then
      git checkout -q -- "$pf"
    fi
  done < <(git diff --name-only)
  # timeout: a wedged network must fail the update cleanly, not hang it.
  # 300s is generous for a fetch that can carry weeks of upstream history.
  timeout 300 git fetch -q origin \
    || die "could not fetch ComfyUI upstream (offline or unreachable) — check
the network and re-run: $0 update"
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
        GIT_TERMINAL_PROMPT=0 git fetch -q origin "+pull/${num}/head:__patch_tmp" </dev/null \
          || { warn "  ! cannot fetch $desc (does it exist?)"; failed=1; continue; } ;;
      branch:*)
        br="${line#branch:}"; desc="origin branch '$br'"
        GIT_TERMINAL_PROMPT=0 git fetch -q origin "+${br}:__patch_tmp" </dev/null \
          || { warn "  ! cannot fetch $desc"; failed=1; continue; } ;;
      remote:*)
        url="${line#remote:}"; br="${url##* }"; url="${url%% *}"
        desc="'$br' from $url"
        # </dev/null + GIT_TERMINAL_PROMPT=0: a prompting fetch (typo'd or
        # private URL -> 401 -> username prompt) must not hang the update or
        # eat the patch-list lines this loop is reading from stdin — same
        # trap cmd_restore's manifest loop guards against.
        GIT_TERMINAL_PROMPT=0 git fetch -q "$url" "+${br}:__patch_tmp" </dev/null \
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
details. This step is required; fix the underlying issue and re-run: $0 ${CMD:-}"
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
(buffered mode). Fix the underlying issue, then re-run: $0 ${CMD:-}"
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
    SPARK_SELF_UPDATED=1 exec "$SELF" update "$@"
  else
    warn "self-update could not fast-forward (uncommitted local edits in
$BASE_DIR?) — continuing with the current version. To update manually:
  git -C $BASE_DIR pull"
  fi
  return 0
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
  self_update "$@"
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
  echo "  spark-comfyui: $VERSION"
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
  # The hard reset wiped the mods' source patches — re-apply them so the
  # rolled-back tree isn't silently missing the GB10 fixes.
  apply_source_patches
  rm -f "$rev_file"
  log "Rolled back. Restart with: $0 run"
}


# =============================================================================
#  reset — regenerate everything, keep user content
# =============================================================================
# Nuke and reinstall ComfyUI/venv/SageAttention without touching the USER_CONTENT
# content. User dirs are mv'd (same filesystem — instant even for 74 GB of
# models) into a sibling hold area, the rest is wiped and reinstalled, then
# the held dirs replace the fresh-from-git skeletons. A .phase marker in the
# hold area makes an interrupted reset resumable: re-running converges.
cmd_reset() {
  local yes=0 arg
  for arg in "$@"; do
    case "$arg" in
      --yes) yes=1 ;;
      *) die "Unknown reset option: $arg (use --yes to skip confirmation)" ;;
    esac
  done

  local hold_dir phase="" d
  hold_dir="$(dirname "$INSTALL_DIR")/.spark-reset-hold"
  if [[ -d "$hold_dir" ]] \
     && [[ -n "$(find "$hold_dir" -mindepth 1 -maxdepth 1 -print -quit 2>/dev/null)" ]]; then
    [[ -f "$hold_dir/.phase" ]] && phase="$(cat "$hold_dir/.phase")"
    log "Resuming an interrupted reset (hold area: $hold_dir${phase:+, phase: $phase})"
  fi

  log "Reset: regenerate the install, keep your content"
  echo "  Deleted and reinstalled fresh:"
  echo "    $INSTALL_DIR"
  echo "    $VENV_DIR"
  echo "    $SAGE_SRC"
  echo "  Preserved (held aside during the reinstall):"
  for d in "${USER_CONTENT[@]}"; do
    echo "    $INSTALL_DIR/$d"
  done
  echo "  The reinstall includes the 10-30 min SageAttention build."
  if [[ "$yes" -ne 1 ]]; then
    [[ -t 0 ]] || die "stdin is not a terminal — pass --yes to confirm the reset"
    local answer=""
    printf "\nType 'reset' to continue: "
    IFS= read -r answer || die "no confirmation received — nothing was changed"
    [[ "$answer" == "reset" ]] || die "confirmation not given — nothing was changed"
  fi

  log "Stopping ComfyUI"
  cmd_stop

  if [[ "$phase" == "wiped" ]]; then
    # Post-wipe resume: the destructive part already happened. A .git dir is
    # no proof the interrupted install finished (clone is its first step, the
    # SageAttention build its longest), so rerun cmd_install unconditionally:
    # it is idempotent and skips or refreshes whatever did complete.
    cmd_install
  else
    log "Holding user content aside in $hold_dir"
    mkdir -p "$hold_dir"
    for d in "${USER_CONTENT[@]}"; do
      [[ -e "$INSTALL_DIR/$d" ]] || continue
      # Never overwrite an existing hold entry: pre-wipe it is the user's
      # data from an interrupted earlier reset, and guessing which of the
      # two copies to keep is not this script's call.
      if [[ -e "$hold_dir/$d" ]]; then
        die "both $hold_dir/$d and $INSTALL_DIR/$d exist — refusing to guess
which copy is yours. Inspect and merge them manually (the hold copy is from
an interrupted earlier reset), then re-run: $0 reset"
      fi
      mv "$INSTALL_DIR/$d" "$hold_dir/"
      echo "  held $d"
    done
    # Wipe guard: nothing user-precious may remain below INSTALL_DIR. This is
    # the line between "reset" and "deleted the models after a failed mv".
    for d in "${USER_CONTENT[@]}"; do
      [[ -e "$INSTALL_DIR/$d" ]] && die "refusing to wipe: $INSTALL_DIR/$d still
exists (the hold move did not complete). Inspect $hold_dir, then re-run: $0 reset"
    done
    log "Wiping $INSTALL_DIR, $VENV_DIR, $SAGE_SRC"
    # The wipe may delete the caller's cwd (running reset from inside the
    # install is natural); git/pip in cmd_install would then fail on getcwd.
    cd "$BASE_DIR"
    rm -rf "$INSTALL_DIR" "$VENV_DIR" "$SAGE_SRC"
    echo wiped > "$hold_dir/.phase"
    cmd_install
  fi

  log "Moving user content back into the fresh install"
  for d in "${USER_CONTENT[@]}"; do
    [[ -e "$hold_dir/$d" ]] || continue
    rm -rf "${INSTALL_DIR:?}/$d"   # fresh-from-git skeleton loses to the user's copy
    mv "$hold_dir/$d" "$INSTALL_DIR/"
    echo "  restored $d"
  done
  # The held dirs carry stock git-tracked files from the old checkout
  # (custom_nodes/websocket_image_save.py, models/configs/*, ...). Put those
  # back at the fresh HEAD so the tree stays clean for update's ff-only
  # merge; checkout -- never touches untracked user files.
  for d in "${USER_CONTENT[@]}"; do
    git -C "$INSTALL_DIR" checkout -q -- "$d" 2>/dev/null || true
  done
  rm -f "$hold_dir/.phase"
  rmdir "$hold_dir"

  log "Reset complete"
  echo "  ComfyUI, the venv and SageAttention were reinstalled fresh;"
  echo "  models, workflows, inputs, outputs and custom nodes were preserved."
  echo "  Start ComfyUI:  $0 run"
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
  [[ -d "$INSTALL_DIR/.git" ]] || die "no install at $INSTALL_DIR — run: $0 install"
  if [[ -z "$out" ]]; then
    mkdir -p "$BASE_DIR/backups"
    out="$BASE_DIR/backups/spark-backup-$(date +%Y%m%d-%H%M%S).tgz"
  fi
  case "$out" in /*) ;; *) out="$PWD/$out" ;; esac

  log "Staging backup"
  local stage; stage="$(mktemp -d)"
  trap 'rm -rf "$stage"' EXIT
  {
    echo "format=1"
    echo "version=$VERSION"
    echo "date=$(date -Is)"
    echo "host=$(hostname)"
    echo "comfyui_commit=$(git -C "$INSTALL_DIR" rev-parse HEAD 2>/dev/null || echo unknown)"
  } > "$stage/meta"

  [[ -f "$PATCH_LIST" ]] && cp -a "$PATCH_LIST" "$stage/comfyui-patches.list"
  [[ -f "$INSTALL_DIR/extra_model_paths.yaml" ]] \
    && cp -a "$INSTALL_DIR/extra_model_paths.yaml" "$stage/extra_model_paths.yaml"

  # Custom nodes: git clones become manifest lines (url + commit, re-cloned on
  # restore); non-git entries are copied whole. "User-installed" = not tracked
  # by ComfyUI's own git (stock files like websocket_image_save.py are).
  : > "$stage/custom-nodes.manifest"
  local entry name url sha
  if [[ -d "$INSTALL_DIR/custom_nodes" ]]; then
    while IFS= read -r entry; do
      name="$(basename "$entry")"
      [[ "$name" == "__pycache__" ]] && continue
      [[ -z "$(git -C "$INSTALL_DIR" ls-files "custom_nodes/$name" 2>/dev/null)" ]] || continue
      if [[ -d "$entry/.git" ]]; then
        url="$(git -C "$entry" remote get-url origin 2>/dev/null || echo unknown)"
        sha="$(git -C "$entry" rev-parse HEAD 2>/dev/null || echo unknown)"
        printf '%s\t%s\t%s\n' "$name" "$url" "$sha" >> "$stage/custom-nodes.manifest"
      else
        mkdir -p "$stage/custom_nodes_plain"
        cp -a "$entry" "$stage/custom_nodes_plain/$name"
      fi
    done < <(find "$INSTALL_DIR/custom_nodes" -mindepth 1 -maxdepth 1 2>/dev/null)
  fi

  # Models are manifested (size + relative path), never copied.
  if [[ -d "$INSTALL_DIR/models" ]]; then
    find "$INSTALL_DIR/models" -type f -printf '%s\t%P\n' | sort -k2 > "$stage/models.manifest"
  else
    : > "$stage/models.manifest"
  fi

  # user/, input/ and output/ are tarred straight from the live tree (no
  # staging copy of possibly-large dirs): tar excludes the logs and caches,
  # --ignore-failed-read plus tolerating exit 1 (a file changed or vanished
  # mid-read) is what makes "safe while ComfyUI is serving" true.
  local members=(-C "$stage" meta models.manifest custom-nodes.manifest)
  [[ -f "$stage/comfyui-patches.list" ]]   && members+=(comfyui-patches.list)
  [[ -f "$stage/extra_model_paths.yaml" ]] && members+=(extra_model_paths.yaml)
  [[ -d "$stage/custom_nodes_plain" ]]     && members+=(custom_nodes_plain)
  members+=(-C "$INSTALL_DIR")
  [[ -d "$INSTALL_DIR/user" ]] && members+=(user)
  [[ -d "$INSTALL_DIR/input" ]] \
    && [[ -n "$(find "$INSTALL_DIR/input" -mindepth 1 -print -quit)" ]] \
    && members+=(input)
  [[ "$with_output" -eq 1 && -d "$INSTALL_DIR/output" ]] && members+=(output)
  local rc=0
  tar -czf "$out" --exclude='__pycache__' --exclude='*.log' \
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

  # Also self-heal a half-gutted machine (checkout present, venv missing):
  # cmd_install is idempotent and refreshes whatever part is there.
  if [[ ! -d "$INSTALL_DIR/.git" || ! -f "$VENV_DIR/bin/activate" ]]; then
    log "No complete install (ComfyUI checkout + venv) — installing first"
    cmd_install
  fi
  activate_venv
  log "Stopping ComfyUI (restoring over its live user/config files)"
  cmd_stop

  log "Merging user state"
  local d
  for d in user input output; do
    [[ -d "$stage/$d" ]] || continue
    mkdir -p "$INSTALL_DIR/$d"
    cp -a "$stage/$d/." "$INSTALL_DIR/$d/"
    echo "  merged $d/"
  done
  local src dst
  for d in extra_model_paths.yaml comfyui-patches.list; do
    src="$stage/$d"
    [[ -f "$src" ]] || continue
    case "$d" in
      extra_model_paths.yaml) dst="$INSTALL_DIR/$d" ;;
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
  mkdir -p "$INSTALL_DIR/custom_nodes"
  local entry name url sha ndir
  if [[ -d "$stage/custom_nodes_plain" ]]; then
    while IFS= read -r entry; do
      name="$(basename "$entry")"
      if [[ -e "$INSTALL_DIR/custom_nodes/$name" ]]; then
        echo "  = $name (present)"
      else
        cp -a "$entry" "$INSTALL_DIR/custom_nodes/$name"
        echo "  + $name (plain copy)"
        if [[ -f "$INSTALL_DIR/custom_nodes/$name/requirements.txt" ]]; then
          pip install -r "$INSTALL_DIR/custom_nodes/$name/requirements.txt" </dev/null \
            || warn "$name: pip install of its requirements failed — the node may not load"
        fi
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
      ndir="$INSTALL_DIR/custom_nodes/$name"
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
      if [[ -f "$ndir/requirements.txt" ]]; then
        pip install -r "$ndir/requirements.txt" </dev/null \
          || warn "$name: pip install of its requirements failed — the node may not load"
      fi
    done < "$stage/custom-nodes.manifest"
  fi

  # Node pip installs can clobber torch (the classic GB10 trap) — the mods
  # pass re-verifies and repairs, same as install/update. Idempotent.
  apply_source_patches

  log "Models check (against the archive's manifest)"
  local missing_count=0 missing_bytes=0 size relpath
  if [[ -f "$stage/models.manifest" ]]; then
    while IFS=$'\t' read -r size relpath; do
      [[ -n "$relpath" ]] || continue
      # A corrupt manifest line must not kill the restore via the size
      # arithmetic below (set -e); treat the size as unknown instead.
      [[ "$size" =~ ^[0-9]+$ ]] || size=0
      [[ -f "$INSTALL_DIR/models/$relpath" ]] && continue
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
#  doctor — diagnose the silent-drift failure modes specific to GB10
# =============================================================================
#  These are the traps that pass a "successful install" but break at runtime.
#  Each check names the exact fix. Sources: community GB10 field guides.
cmd_doctor() {
  PASS=0; FAIL=0
  activate_venv

  # Version first: doctor output doubles as a bug report, and it should say
  # which spark-comfyui produced it. The pending-update probe lives here (and
  # only here) on purpose — it needs a network fetch, which has no business
  # on the run/stop/status hot paths. warn, not bad: a pending release is
  # not a health failure and must not flip doctor's exit code.
  hdr "spark-comfyui (self)"
  if [[ -d "$BASE_DIR/.git" ]]; then
    info "git revision $(git -C "$BASE_DIR" rev-parse --short HEAD 2>/dev/null)"
  else
    info "not a git clone — self-update unavailable"
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
    torch_cuda_diag
    bad "torch cannot use the GPU — the diag lines above name the cause.
        If torch was re-pinned: $0 update (repairs it automatically)"
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

  hdr "comfy-kitchen (NVFP4/FP8 quantization backends)"
  # ComfyUI picks comfy-kitchen's fastest capable backend per call and
  # quietly serves quantized models from the pure-PyTorch 'eager' path when
  # the native CUDA backend is broken/unavailable — everything still works,
  # just massively slower. A registry listing is a claim; run the kernels.
  if ! python -c "import comfy_kitchen" 2>/dev/null; then
    info "comfy-kitchen not installed (ships with current ComfyUI; only used by quantized models)"
  elif kitchen_nvfp4_ok; then
    ok "NVFP4 kernels live on the native CUDA backend (forced + numerically verified)"
  else
    bad "comfy-kitchen's CUDA backend FAILED a live NVFP4 kernel test —
        quantized (NVFP4/FP8) models will silently run on the slow eager
        path. Requires torch cu13 and an SM>=10.0 GPU. Fix: $0 update
        (repairs torch); if it persists, check ComfyUI's startup log for
        'Found comfy_kitchen backend cuda' and its unavailable_reason"
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
Wants=network-online.target
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

# =============================================================================
#  container (EXPERIMENTAL) — the containerized runtime, phase 1
# =============================================================================
# The image holds everything reproducible (ComfyUI at a pinned commit, venv
# with cu130 torch, native sm_121 SageAttention, GPU onnxruntime, build-time
# mods); the USER_CONTENT set is bind-mounted from this install's tree, so
# native and container runs share models/workflows/custom nodes. The build
# has no GPU, so the live kernel gates run in the container entrypoint on
# every start (container/entrypoint.sh), not at build time.

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
  docker build \
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

cmd_container_run() {
  need_docker
  docker image inspect "$CONTAINER_IMAGE:latest" >/dev/null 2>&1 \
    || die "image $CONTAINER_IMAGE:latest not found — run: $0 container build"
  # Bind-mount exactly the USER_CONTENT set. Dirs are created if missing so
  # a container-only setup works without a native install; the yaml entry is
  # a file and only mounted when it exists (docker would create a dir).
  local vols=() entry
  for entry in "${USER_CONTENT[@]}"; do
    if [[ "$entry" == *.yaml ]]; then
      [[ -f "$INSTALL_DIR/$entry" ]] \
        && vols+=(-v "$INSTALL_DIR/$entry:/opt/ComfyUI/$entry:ro")
    else
      mkdir -p "$INSTALL_DIR/$entry"
      vols+=(-v "$INSTALL_DIR/$entry:/opt/ComfyUI/$entry")
    fi
  done
  log "Launching containerized ComfyUI on port $PORT (Ctrl-C stops it)"
  # Hardening: no capabilities, no privilege escalation, only the GPU
  # exposed. The named cache volume keeps pip downloads and compiled sm_121
  # CUDA kernels across container recreation (the container itself is --rm:
  # every launch starts from the immutable image).
  exec docker run --rm --name "$CONTAINER_NAME" \
    --gpus all \
    --shm-size 1g \
    --cap-drop ALL \
    --security-opt no-new-privileges \
    -p "$PORT:8188" \
    -v "$CONTAINER_IMAGE-cache:/home/comfy/.cache" \
    -e SPARK_BF16 \
    -e SPARK_STATIC_VRAM \
    "${vols[@]}" \
    "$CONTAINER_IMAGE:latest" "$@"
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

cmd_container() {
  local sub="${1:-}"
  shift || true
  case "$sub" in
    build) cmd_container_build "$@" ;;
    run)   cmd_container_run "$@" ;;
    stop)  cmd_container_stop ;;
    shell) cmd_container_shell ;;
    *) die "Unknown container subcommand: ${sub:-<none>} (try: container build | run | stop | shell)" ;;
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
  install)  cmd_install "$@" ;;
  run)      cmd_run "$@" ;;
  stop)     cmd_stop ;;
  update)   cmd_update "$@" ;;
  doctor)   cmd_doctor ;;
  status)   cmd_status "$@" ;;
  tune)     cmd_tune "$@" ;;
  backup)   cmd_backup "$@" ;;
  restore)  cmd_restore "$@" ;;
  reset)    cmd_reset "$@" ;;
  service)  cmd_service ;;
  container) cmd_container "$@" ;;
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
  *) die "Unknown command: $CMD (try: install | run | stop | update | doctor | status | tune | backup | restore | reset | service | container)" ;;
esac
