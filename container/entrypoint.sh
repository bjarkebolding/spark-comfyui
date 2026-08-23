#!/usr/bin/env bash
# Container entrypoint: the launch pipeline. The image is immutable, so
# anything that needs the GPU (the live kernel gates), the bind-mounted user
# content (Manager config), or the user's installed custom nodes runs here,
# before every launch. Seven steps, numbered below, in a load-bearing order.
#
# No mods run here. Mods are build-time image modifications and nothing else
# (mods/README.md); the two steps that used to be mods are ensure_torch_cuda
# in container/gb10.sh and container/manager-config.py.
set -euo pipefail

log()  { printf '\n\033[1;32m==> %s\033[0m\n' "$*"; }
warn() { printf '\033[1;33m[warn] %s\033[0m\n' "$*"; }
info() { printf '\033[1;36m[info] %s\033[0m\n' "$*"; }
die()  { printf '\033[1;31m[error] %s\033[0m\n' "$*" >&2; exit 1; }

# The container runs as the CALLER's uid (--user, set by _container_run_args),
# so there is normally no passwd entry for it. getpwuid() then raises and
# anything resolving the current user breaks. HOME, USER and LOGNAME come from
# the image ENV; this supplies the passwd line itself. /etc/passwd is group
# writable in the image for exactly this. A failure here is not fatal: the env
# vars already cover the common callers, so warn and carry on.
if ! getent passwd "$(id -u)" >/dev/null 2>&1; then
  printf 'comfy:x:%s:%s:comfy:%s:/bin/bash\n' \
    "$(id -u)" "$(id -g)" "${HOME:-/home/comfy}" >> /etc/passwd 2>/dev/null \
    || warn "no passwd entry for uid $(id -u) and /etc/passwd is not writable"
fi

# Everything this process writes into the group-0 image trees has to stay group
# writable, or the next caller running as a different uid could not overwrite
# it. Only affects the image trees; bind-mounted content is owned by the caller.
umask 002

: "${INSTALL_DIR:?}" "${VENV_DIR:?}"
cd "$INSTALL_DIR"

# Package installs prefer uv, falling back to pip. This matters because 'run'
# is --rm: the custom-node requirements pass below happens on EVERY launch, and
# uv resolves and downloads in parallel against the same cache volume pip uses.
# uv arrives with ComfyUI-Manager's own requirements, so it is present but not
# guaranteed — a probe, not an assumption. pip stays the fallback for a second
# reason: uv's resolver is stricter, and a node with sloppy pins that pip
# tolerates must not be what stops the server from starting.
UV="$(command -v uv || true)"
INSTALLER="pip"
[[ -n "$UV" ]] && INSTALLER="uv"
py_install() {
  if [[ -n "$UV" ]]; then
    "$UV" pip install -q --python "$VENV_DIR/bin/python" "$@" </dev/null && return 0
    warn "uv could not install this set — retrying the same set with pip"
  fi
  pip install -q "$@" </dev/null
}

# 1. Manager config FIRST. It only writes user/__manager/config.ini, so it
#    depends on nothing below it, and everything below depends on it: the
#    node list needs network_mode to be a known value (not 'offline') and
#    use_uv on, and letting cm-cli create a default config.ini that this
#    step then rewrites would leave two sources of truth for the same file.
#    It lives under the bind-mounted user dir, so it is asserted at run
#    time, never baked into the image.
log "Manager config"
python /opt/spark/manager-config.py apply \
  || warn "Manager config apply failed — continuing, Manager may be gated"

# 2. Registry node list. custom_nodes is bind-mounted, so a node set cannot
#    be baked into the image; reconciling it at start is the only place it
#    can happen. Runs BEFORE the requirements and torch passes below on
#    purpose: cm-cli installs node dependencies into the live venv, which is
#    exactly what clobbers torch, so the guard has to stay downstream of it.
#    Warn, never die, like the onnx guard: a registry outage or an offline
#    box is a missing node, not a reason to refuse to serve.
log "Registry node list"
nodes_list=/opt/spark/comfyui-nodes.list
if [[ ! -f "$nodes_list" ]]; then
  info "no node list mounted — nothing to reconcile"
else
  node_entries=()
  while IFS= read -r line || [[ -n "$line" ]]; do
    line="${line%%#*}"
    line="${line#"${line%%[![:space:]]*}"}"
    line="${line%"${line##*[![:space:]]}"}"
    [[ -z "$line" ]] && continue
    # The entry becomes a cm-cli argument, quoted at the call site, so a
    # metacharacter cannot execute. A junk line should still be named rather
    # than handed to the installer to fail on obscurely.
    if [[ ! "$line" =~ ^[A-Za-z0-9._/:@+-]+$ ]]; then
      warn "ignoring malformed node-list entry: $line"
      continue
    fi
    node_entries+=("$line")
  done < "$nodes_list"

  if (( ${#node_entries[@]} == 0 )); then
    info "node list has no active entries"
  else
    # Is this entry satisfied on disk? The install directory is NOT
    # predictable from the entry text. A bare or versioned id lands under the
    # id (cnr_install joins custom_nodes with node_id), but a URL is resolved
    # against the registry first: gitclone_install calls get_cnr_by_repo and
    # delegates to install_by_id, so
    # https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite.git installs
    # into comfyui-videohelpersuite, NOT ComfyUI-VideoHelperSuite. Guessing
    # the repo basename reported a healthy install as a failure on every
    # start, which then triggered the cache refresh below every start too
    # (field-verified 2026-08-05: a 2m18s penalty per launch, plus a warning
    # about a node that was sitting right there).
    #
    # So match on what is actually there: the directory name, or the project
    # name in the node's pyproject.toml. A CNR install arrives as a zip and
    # has no .git to read a remote from, but it always carries a pyproject.
    # Compared case-insensitively, which is what the id-vs-repo-name
    # difference usually amounts to.
    node_present() {
      local want="$1" d base pname
      case "$want" in
        *://*|*.git) want="${want%.git}"; want="${want##*/}" ;;
        *)           want="${want%%@*}" ;;
      esac
      for d in "$INSTALL_DIR"/custom_nodes/*/; do
        [[ -d "$d" ]] || continue
        d="${d%/}"; base="${d##*/}"
        [[ "${base,,}" == "${want,,}" ]] && return 0
        [[ -f "$d/pyproject.toml" ]] || continue
        pname="$(sed -n 's/^[[:space:]]*name[[:space:]]*=[[:space:]]*["'"'"']\([^"'"'"']*\)["'"'"'].*/\1/p' \
                   "$d/pyproject.toml" | head -1)"
        [[ -n "$pname" && "${pname,,}" == "${want,,}" ]] && return 0
      done
      return 1
    }

    # How many node directories exist right now. A rise across an install is
    # the last-resort proof that something landed, for the rare node whose
    # registry id resembles neither its repo name nor its pyproject name.
    node_dir_count() {
      find "$INSTALL_DIR/custom_nodes" -mindepth 1 -maxdepth 1 -type d 2>/dev/null | wc -l
    }

    # Fast path: an all-present list needs no cm-cli at all — no Manager
    # import, no registry round trip, nothing on a steady-state start.
    # VERSIONED entries (id@1.2.3, id@nightly) always go to cm-cli, which is
    # the only thing that can tell WHICH version is on disk. Bare ids and
    # URLs are satisfied by presence, which is also all cm-cli does with them
    # (an already-installed node resolves to 'skip'), so a URL entry now
    # short-circuits here instead of paying a round trip on every start.
    node_todo=()
    for entry in "${node_entries[@]}"; do
      if [[ "$entry" == *@* ]] || ! node_present "$entry"; then
        node_todo+=("$entry")
      fi
    done
    if (( ${#node_todo[@]} == 0 )); then
      info "${#node_entries[@]} listed node(s) already present — nothing to install"
    else
      info "reconciling ${#node_todo[@]} of ${#node_entries[@]} listed node(s) via cm-cli"
      # One call per entry so a failure names the node instead of hiding in a
      # batch. COMFYUI_PATH is mandatory (cm-cli exits immediately without
      # it); </dev/null keeps a prompting git clone from eating the loop's
      # stdin; timeout keeps a hung registry off the launch path.
      #
      # The exit code is NOT a gate, because cm-cli does not have one.
      # --exit-on-fail is still passed for the day upstream honours it, but it
      # is a no-op in ComfyUI-Manager 4.2.2: the install command forwards
      # exit_on_fail= through for_each_nodes, while install_node reads
      # kwargs['raise_on_fail'], so nothing ever raises, nothing lands in
      # for_each_nodes' failed[] list, and the command exits 0 for a
      # nonexistent id AND for a bogus URL (both field-verified 2026-08-05).
      # Golden rule 3, with no fallback available: the live check is whether
      # the node is actually on disk afterwards.
      # cm-cli is extremely chatty: an interpreter banner per invocation, FETCH
      # DATA lines, a tqdm bar, and a '## ComfyUI-Manager: EXECUTE => [...]'
      # block for every transitive pip dependency. On a first launch that
      # buried the one fact a user wants ("did my node install?") under about
      # forty lines of someone else's internals.
      #
      # So capture it and stay quiet while it works, which is golden rule 5.
      # The output is NOT discarded: on failure the caller prints the whole
      # thing, because that is the moment it is worth reading. Manager also
      # keeps its own copy under user/__manager (file_logging is on, set by
      # manager-config.py), so the detail survives even a successful run.
      #
      # NODE_LAST_LOG carries the path out to the caller. Deliberately not a
      # subshell: this must set it in the caller's scope.
      NODE_LAST_LOG=""
      node_install_once() {
        local before after log
        before="$(node_dir_count)"
        log="$(mktemp)"
        COMFYUI_PATH="$INSTALL_DIR" timeout 600 cm-cli install --exit-on-fail "$1" \
          </dev/null >"$log" 2>&1 || true
        if node_present "$1"; then rm -f "$log"; NODE_LAST_LOG=""; return 0; fi
        after="$(node_dir_count)"
        if (( after > before )); then rm -f "$log"; NODE_LAST_LOG=""; return 0; fi
        NODE_LAST_LOG="$log"
        return 1
      }

      # Print what cm-cli said, but only when it matters. Trimmed to the tail:
      # the useful part of a cm-cli failure is always at the end, and the head
      # is the banner noise we suppressed in the first place.
      node_dump_log() {
        [[ -n "$NODE_LAST_LOG" && -s "$NODE_LAST_LOG" ]] || return 0
        echo "    ---- cm-cli output ----"
        tail -25 "$NODE_LAST_LOG" | sed 's/^/    /'
        echo "    -----------------------"
        rm -f "$NODE_LAST_LOG"; NODE_LAST_LOG=""
      }
      node_fail=0
      cache_warmed=0
      node_i=0
      for entry in "${node_todo[@]}"; do
        node_i=$((node_i+1))
        node_t0=$SECONDS
        info "[$node_i/${#node_todo[@]}] $entry"
        # Resolve a BARE id to a version BEFORE calling cm-cli, not after.
        #
        # cm-cli can only turn a bare id into a version via its own registry
        # cache, which is empty on a new box, so that first call is doomed
        # there — and it fails LOUDLY, printing "ERROR: An error occurred while
        # installing ..." from inside cm-cli before we ever get to recover.
        # A new user's first launch therefore showed a red ERROR followed by a
        # quiet recovery, which reads as a broken install (field-reported
        # 2026-08-09 during a new-box test). Asking the registry for the one
        # version we need costs one request (measured 0.49s) and removes both
        # the noise and the wasted call.
        #
        # A pinned id@version skips the cache entirely: resolve_node_spec
        # returns immediately and cnr_install makes a single install request.
        # On a WARM box cm-cli would resolve the same bare id to this same
        # version out of cnr_map, which is built from this same listing, so
        # pinning changes which call is made and not which version lands.
        #
        # Bare ids only. Anything with @ or / or : already carries what cm-cli
        # needs. Every failure falls through to the unpinned attempt below, so
        # this can save time but never change the outcome, and the on-disk
        # check in node_install_once remains the only success gate.
        if [[ "$entry" != *@* && "$entry" != *:* && "$entry" != */* ]]; then
          if pinned_ver="$(timeout 30 python /opt/spark/resolve-node.py "$entry" 2>/dev/null)" \
             && [[ -n "$pinned_ver" ]]; then
            info "        version $pinned_ver (registry, no catalogue fetch)"
            if node_install_once "$entry@$pinned_ver"; then
              info "        installed in $((SECONDS-node_t0))s"
              continue
            fi
            warn "        pinned install did not take — retrying unpinned"
          fi
        fi
        if node_install_once "$entry"; then
          info "        installed in $((SECONDS-node_t0))s"
          continue
        fi
        # A COLD box has no Manager registry cache, and the catalogue bundled
        # in the pip package does not carry every registry node. Field-hit
        # 2026-08-05 after a data/ wipe: the bundled list holds 3587 entries
        # and zero matches for a node that installs fine once the cache is
        # warm, so the first start after a wipe silently got no nodes.
        # Refresh once per start, then retry. Reactive rather than always-on,
        # so a warm box pays nothing, and it also heals a cache too old to
        # know about a recently published node.
        if (( ! cache_warmed )); then
          cache_warmed=1
          # Heartbeat, not silence. The fetch takes a few minutes on a cold
          # box and this is the launch path, so a run with no output for that
          # long is indistinguishable from a hang (field-hit 2026-08-05: it
          # was reported as stuck while working normally). A \r spinner is
          # wrong here: 'run' is docker run WITHOUT -t, so stdout is a pipe
          # and only newline-terminated lines render sanely in the log.
          info "registry cache is cold — fetching the node catalogue"
          info "  (a few minutes on a first run; every later start skips this)"
          COMFYUI_PATH="$INSTALL_DIR" timeout 600 cm-cli update-cache </dev/null >/dev/null 2>&1 &
          cache_pid=$!
          cache_secs=0
          while kill -0 "$cache_pid" 2>/dev/null; do
            sleep 20
            cache_secs=$((cache_secs+20))
            kill -0 "$cache_pid" 2>/dev/null && info "  still fetching ... ${cache_secs}s"
          done
          wait "$cache_pid" || true
          info "  catalogue fetched in ${cache_secs}s — retrying $entry"
          if node_install_once "$entry"; then
            info "        installed in $((SECONDS-node_t0))s"
            continue
          fi
        fi
        warn "        FAILED to install '$entry' — continuing without it"
        node_dump_log
        node_fail=1
      done
      if (( node_fail )); then
        warn "one or more listed nodes are missing; the next start retries them"
      fi
    fi
  fi
fi

# 3. Custom-node requirements. Manager clones nodes into the mounted
#    custom_nodes dir, but their pip deps land in the container's writable
#    layer and vanish on recreation, so a fresh container must install them.
#    A RESTARTED container (service restart policy, reboots) still has
#    them; the marker lives in /tmp — the same writable layer, so it dies
#    exactly when the installed packages die and can never wrongly skip on
#    a fresh container. All requirement files go to pip in one invocation
#    (one resolver pass instead of one per node); a failure falls back to
#    per-node installs to isolate the culprit. A failing node is a warning,
#    not a dead server — same policy as restore.
#
#    This globs what is ON DISK, which is deliberately NOT the same set as
#    comfyui-nodes.list: the list is additive (what must be installed), and
#    a node stays after its line is removed, plus Manager-UI installs never
#    appear in it at all. The count alone read as a contradiction in the
#    field, so the nodes are named.
log "Custom-node requirements"
shopt -s nullglob
req_files=("$INSTALL_DIR"/custom_nodes/*/requirements.txt)
shopt -u nullglob
marker=/tmp/spark-node-reqs.sha256
if (( ${#req_files[@]} == 0 )); then
  info "no custom nodes with requirements"
elif [[ -f "$marker" ]] \
     && [[ "$(sha256sum "${req_files[@]}")" == "$(cat "$marker")" ]]; then
  info "requirements unchanged since this container's last start — skipping"
else
  rm -f "$marker"
  req_names=()
  pip_args=()
  for req in "${req_files[@]}"; do
    req_names+=("$(basename "$(dirname "$req")")")
    pip_args+=(-r "$req")
  done
  req_names_txt="$(printf '%s, ' "${req_names[@]}")"
  info "installing requirements for ${#req_files[@]} custom node(s) (via $INSTALLER)"
  info "  ${req_names_txt%, }"
  if py_install "${pip_args[@]}"; then
    sha256sum "${req_files[@]}" > "$marker"
  else
    warn "combined install failed — retrying per node to isolate it"
    req_fail=0
    for req in "${req_files[@]}"; do
      node="$(basename "$(dirname "$req")")"
      info "custom node $node: installing requirements"
      py_install -r "$req" \
        || { warn "installing requirements for custom node $node failed — the node may not load"; req_fail=1; }
    done
    # No marker after a failure: every start retries until the set heals.
    (( req_fail )) || sha256sum "${req_files[@]}" > "$marker"
  fi
fi

# The GPU probes and guards used by the two steps below. Not the mod
# contract: no mod runs at launch (see mods/README.md).
# shellcheck disable=SC1091
source /opt/spark/gb10.sh

# 4. Torch guard AFTER every install above — the node list and the
#    requirements pass are exactly what clobbers torch. Paired with the onnx
#    guard below: same placement, same reason, and since 2026-08-08 the same
#    shape (both are ensure_* functions in gb10.sh). This one dies
#    rather than warns, because a torch without CUDA is not a slow server, it
#    is no server at all.
log "Torch CUDA guard"
ensure_torch_cuda || die "torch CUDA check failed — see the diag lines above. If the wheel
set itself is broken, rebuild the image: spark-comfyui.sh update"

# 5. onnxruntime GPU guard, same placement logic as the torch guard: a custom
#    node that pip installs onnxruntime shadows the sm_121 GPU wheel through
#    the shared import path, with no pip conflict to notice. DWPose and the
#    ControlNet preprocessors then silently run on CPU. The healthy case is
#    one provider query (~0.06s measured), so this is cheap on every start;
#    only a genuinely shadowed install pays the reinstall, which is a 57 MB
#    wheel and took ~3s when tested against a real shadow.
#    Warn, never die, unlike torch above: CPU onnxruntime is a slowdown, not
#    a broken server, so it must not stop a launch. The '||' also suspends
#    set -e inside the function, which is what keeps a failed pip from
#    killing the entrypoint.
log "onnxruntime GPU guard"
ensure_onnx_gpu \
  || warn "onnxruntime GPU guard did not complete — DWPose and ControlNet
preprocessors may fall back to CPU this session (spark-comfyui.sh doctor re-checks)"

# 6. SageAttention live kernel gate. The image build compiled it blind (no
#    GPU exists at build time); this is where golden rule 3 now lives.
log "SageAttention kernel gate"
if sage_kernel_ok; then
  sage_flag=(--use-sage-attention)
  info "SageAttention enabled (kernel verified live)"
else
  die "SageAttention kernel FAILED on this GPU — refusing to launch degraded.
Rebuild the image: spark-comfyui.sh update"
fi

# 7. Launch. Flags mirror the native cmd_run; exposure is controlled by the
#    host's port mapping, so --listen 0.0.0.0 here is scoped to the
#    container's own network namespace.
extra_flags=()
# bf16 is the GB10 fast path. The VAE half is split out because LTX-2.3's
# audio VAE never casts the input waveform to the VAE dtype, so --bf16-vae
# fails it with "Input type (float) and bias type (c10::BFloat16)". That hits
# the stock LoadAudio node too, i.e. every LTX-2.3 audio workflow. SPARK_BF16_VAE=0
# drops that one flag and keeps the unet and text-encoder speedups.
if [[ "${SPARK_BF16:-1}" == "1" ]]; then
  extra_flags+=(--bf16-unet --bf16-text-enc)
  if [[ "${SPARK_BF16_VAE:-1}" == "1" ]]; then
    extra_flags+=(--bf16-vae)
  else
    info "bf16 VAE disabled (SPARK_BF16_VAE=0) — needed for LTX-2.3 audio workflows"
  fi
fi
if [[ "${SPARK_STATIC_VRAM:-0}" == "1" ]]; then
  extra_flags+=(--disable-dynamic-vram)
fi
# Optional VRAM reserve (GB) kept free for the OS / a co-resident CUDA
# process. Unset by default (ComfyUI's own default headroom applies); set
# SPARK_RESERVE_VRAM at run time to harden against the overcommit freeze when
# pushing large models. Numeric guard so a stray value can't inject flags.
if [[ "${SPARK_RESERVE_VRAM:-}" =~ ^[0-9]+([.][0-9]+)?$ ]]; then
  extra_flags+=(--reserve-vram "$SPARK_RESERVE_VRAM")
elif [[ -n "${SPARK_RESERVE_VRAM:-}" ]]; then
  warn "ignoring SPARK_RESERVE_VRAM='$SPARK_RESERVE_VRAM' (not a number of GB)"
fi

log "Launching ComfyUI"
exec python main.py \
  --listen 0.0.0.0 \
  --port 8188 \
  --enable-manager \
  --preview-method auto \
  --disable-pinned-memory \
  "${sage_flag[@]}" \
  "${extra_flags[@]}" \
  "$@"
