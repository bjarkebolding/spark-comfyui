# shellcheck shell=bash
# =============================================================================
#  lib/recipes.sh — recipe list/show/check/install/capture
#
#  A recipe is everything a workflow needs in order to run: the workflow
#  file, the custom nodes it wants, and every model with its destination,
#  size and sha256. Recipes live in recipes/ and are tracked, because a
#  recipe is the shareable, executable form of "here is how I got this
#  working".
#
#  SOURCED by spark-comfyui.sh, not run. There is no dispatch here and no
#  execute bit; the commands are
#  ./spark-comfyui.sh recipe list|show|check|install|capture. Everything
#  under lib/ is internal — the user-facing surface is the one executable.
#
#  It lives out here rather than inside spark-comfyui.sh because it only
#  POPULATES an install: it never touches the image, the container or the
#  host, and would work unchanged if ComfyUI had been installed some other
#  way. Everything operational stays in the entry point. It uses the entry
#  point's shared contracts (resolve_mounts, _mount_path, seed_nodes_list)
#  and its log helpers, and owns nothing else.
#
#  The JSON work is in sibling lib/*.py files: recipe_read.py decides
#  whether a path can escape models/, so it belongs somewhere it can be
#  linted and tested, not in a heredoc.
# =============================================================================

_recipe_read() {
  python3 "$LIB_DIR/recipe_read.py" "$1"
}

# A recipe is ONE self-contained file, $RECIPES_DIR/<name>.json, with the
# workflow graph embedded. One file is the whole point: it can be pasted into
# a gist or an issue and installed from there.
# The name lands in a path, so it gets the same treatment as the node names in
# restore: it must not be able to point anywhere else.
_recipe_path() {
  case "$1" in
    ""|.|..|*/*|-*) die "invalid recipe name: '$1'
(a plain name, no path — see: $0 recipe list)" ;;
  esac
  printf '%s' "$RECIPES_DIR/${1%.json}.json"
}

_recipe_size() { numfmt --to=iec "$1" 2>/dev/null || printf '%sB' "$1"; }

_recipe_list() {
  [[ -d "$RECIPES_DIR" ]] || { info "no recipes directory ($RECIPES_DIR)"; return 0; }
  local json name parsed found=0
  hdr "Recipes"
  for json in "$RECIPES_DIR"/*.json; do
    [[ -f "$json" ]] || continue
    found=1
    name="$(basename "$json" .json)"
    # A broken recipe must be visible in the listing, not silently absent.
    if ! parsed="$(_recipe_read "$json" 2>&1)"; then
      printf '  %-24s \033[1;31mINVALID\033[0m\n' "$name"
      continue
    fi
    printf '  %-24s %s\n' "$name" \
      "$(printf '%s\n' "$parsed" | awk -F'\t' '$1=="META" && $2=="name" {print $3}')"
  done
  (( found )) || info "no recipes yet — author one with: $0 recipe capture WORKFLOW --name NAME"
  echo
  echo "  Detail:  $0 recipe show NAME"
  echo "  Dry run: $0 recipe check NAME"
}

_recipe_show() {
  local name="${1:-}"
  [[ -n "$name" ]] || die "usage: $0 recipe show NAME"
  local json parsed
  json="$(_recipe_path "$name")"
  [[ -f "$json" ]] || die "no such recipe: $name (looked for $json)"
  parsed="$(_recipe_read "$json")" || die "recipe '$name' is not valid — see the error above"

  hdr "$name"
  local tag f1 f2 f3 f4 f5 total=0 nmodels=0
  while IFS=$'\t' read -r tag f1 f2 f3 f4 f5; do
    case "$tag" in
      META)  if [[ -n "$f2" ]]; then printf '  %-12s %s\n' "$f1:" "$f2"; fi ;;
      NODE)  printf '  %-12s %s\n' "node:" "$f1" ;;
      FLAG)  printf '  %-12s %s\n' "run flag:" "$f1" ;;
      MODEL) nmodels=$((nmodels+1)); total=$((total+f4))
             printf '  %-12s %s/%s (%s)\n' "model:" "$f2" "$f1" "$(_recipe_size "$f4")" ;;
    esac
  done <<< "$parsed"
  echo
  printf '  %s model(s), %s total\n' "$nmodels" "$(_recipe_size "$total")"
}

# Fetch one model: resume, verify, then move into place. Never leaves a
# partial or unverified file where the loader would find it.
_recipe_fetch() {
  local url="$1" out="$2" want_sha="$3"
  local size="${4:-0}" all_done="${5:-0}" all_total="${6:-0}"
  local part="$out.part" got_sha t0
  t0=$(date +%s)
  local -a auth=()
  # Tokens come from the environment only. A recipe is committed to a repo,
  # so a credential must never be able to live in one.
  case "$url" in
    https://huggingface.co/*) [[ -n "${HF_TOKEN:-}" ]] \
      && auth=(--header="Authorization: Bearer $HF_TOKEN") ;;
    https://civitai.com/*)    [[ -n "${CIVITAI_TOKEN:-}" ]] \
      && auth=(--header="Authorization: Bearer $CIVITAI_TOKEN") ;;
  esac
  # wget stays fully quiet and its stderr is captured, shown only if the
  # download fails. --no-verbose still logs one line per file containing HF's
  # signed CDN url, which is about a thousand characters of noise. On a
  # terminal we render progress ourselves by polling the .part file, which is
  # the only way to show the whole job as well as the current file.
  local -a progress=(-q)

  # wget, not curl, and specifically for -c. Measured 2026-08-05 against a
  # server that drops mid-transfer: wget re-evaluates the continue offset on
  # every try, so --tries resumes (Range went 0, 4000000, 8000000, 12000000
  # in one invocation). curl does NOT: --retry ignores a dropped connection
  # entirely (error 18 is not in its transient set) and --retry-all-errors
  # restarts from zero even with -C -, which on a 20 GB model throws the
  # whole transfer away. wget is also the safer dependency here: it is
  # priority=standard from ubuntu-standard, whereas curl is priority=optional
  # and present on this box only because an NVIDIA package pulled it in.
  #
  # --read-timeout turns a stalled connection into a retryable failure rather
  # than a download that hangs forever holding up a launch.
  local rc=0 errf start now cur wpid
  errf="$(mktemp)"
  start=$(date +%s)
  # Backgrounded either way so the two paths differ only in whether anything
  # is drawn. Its own retry and resume are untouched; we just watch it grow.
  wget -c --tries=5 --waitretry=5 --timeout=30 --read-timeout=120 \
    "${progress[@]}" "${auth[@]}" -O "$part" "$url" >/dev/null 2>"$errf" &
  wpid=$!
  if [[ -t 1 ]]; then
    while kill -0 "$wpid" 2>/dev/null; do
      cur=$(stat -c %s "$part" 2>/dev/null || echo 0)
      now=$(date +%s)
      printf '\r\033[K%s' \
        "$(_recipe_render "$cur" "$size" "$((now-start))" "$((all_done+cur))" "$all_total")"
      sleep 0.5
    done
  fi
  wait "$wpid" || rc=$?
  [[ -t 1 ]] && printf '\r\033[K'
  # wget's own words matter on failure; they are all the user has to go on.
  (( rc )) && sed 's/^/     /' "$errf" >&2
  rm -f "$errf"
  if (( rc )); then
    # A zero-length .part is litter, not progress: a refused request still
    # creates the file. Drop it so a retry starts clean and 'check' is not
    # confused by a partial that holds nothing.
    [[ -s "$part" ]] || rm -f "$part"
    warn "download failed: $(basename "$out")"
    # Ask the server WHY rather than infer it from an exit code. wget exit 8
    # is "server issued an error response", which lumps 404, 403 and 500
    # together, so an earlier version cheerfully told users a missing file
    # was gated. -q also silences wget's own diagnostic, so there is nothing
    # to fall back on. One HEAD request on the failure path buys an accurate
    # answer, and reuses the preflight classifier so there is a single
    # definition of what each status means.
    _recipe_preflight \
      "$(basename "$out")"$'\t'-$'\t'"$url"$'\t'"$size"$'\t'"$want_sha" || true
    [[ -s "$part" ]] \
      && echo "     $( _recipe_size "$(stat -c %s "$part")" ) kept at .part — re-running resumes"
    return 1
  fi
  # Say so. Hashing 20 GB takes ~15 seconds and silence there reads as a hang.
  [[ -t 1 ]] && printf '     verifying sha256 of %s ...' "$(_recipe_size "$size")"
  got_sha="$(sha256sum "$part" | awk '{print $1}')"
  [[ -t 1 ]] && printf '\r\033[K'
  if [[ "$got_sha" != "$want_sha" ]]; then
    warn "sha256 MISMATCH, refusing to install
    expected $want_sha
    got      $got_sha
  left at $part — delete it to start over"
    return 1
  fi
  # Only now is it a real model. Same filesystem, so the rename is atomic and
  # a half-downloaded file can never look installed.
  mv -f "$part" "$out"
  local secs=$(( $(date +%s) - t0 ))
  printf '     \033[1;32m✓\033[0m %s verified in %dm%02ds\n' \
    "$(_recipe_size "$size")" "$((secs/60))" "$((secs%60))"
}

# One progress line, one awk per tick, same shape as _watch_row in the entry
# point. Shows this file AND the whole job, which wget cannot do because it
# only ever knows about the file in front of it.
# $1 done  $2 total  $3 elapsed_s  $4 all_done  $5 all_total  $6 bar width
_recipe_render() {
  awk -v d="$1" -v t="$2" -v e="$3" -v ad="$4" -v at="$5" -v w="${6:-28}" '
    function human(b) {
      if (b >= 1073741824) return sprintf("%.1fG", b/1073741824)
      if (b >= 1048576)    return sprintf("%.0fM", b/1048576)
      if (b >= 1024)       return sprintf("%.0fK", b/1024)
      return sprintf("%dB", b)
    }
    function dur(s) {
      if (s <= 0 || s > 359999) return "--"
      h = int(s/3600); m = int((s%3600)/60); ss = int(s%60)
      if (h) return sprintf("%dh%02dm", h, m)
      if (m) return sprintf("%dm%02ds", m, ss)
      return sprintf("%ds", ss)
    }
    BEGIN {
      # Glyphs into an array, never substr: substr is byte-oriented and would
      # shred these multibyte blocks under mawk (same reason as _watch_row).
      pct  = (t > 0 ? d/t : 0)
      n    = int(pct*w + 0.5)
      bar  = ""
      for (i = 0; i < w; i++) bar = bar (i < n ? "\342\226\210" : "\342\226\221")
      rate = (e > 0 ? d/e : 0)
      eta  = (rate > 0 ? (t-d)/rate : -1)
      apct = (at > 0 ? ad/at*100 : 0)
      printf "     \033[1;36m%s\033[0m %3d%%  %7s  %6.1f MB/s  eta %-7s  total %2d%%",
             bar, pct*100, human(d), rate/1048576, dur(eta), apct
    }'
}

# Ask the server about every model we are about to fetch, before fetching any
# of them. A handful of HEAD requests beats discovering at 15 GB that the next
# file is gated.
#
# Deliberately NOT "warn if the token is unset": almost every HuggingFace file
# is public, so that would be a false alarm on nearly every recipe, which
# golden rule 5 forbids. Golden rule 3 instead — a live check decides. The
# same request also catches a url that has rotted (404) and a recipe whose
# size has drifted from the file the server now serves, both of which a token
# check would miss entirely.
#
# Quiet when everything is fine; prints only problems. Returns non-zero if any.
_recipe_preflight() {
  local -a bad=()
  local entry file dest url size sha out rc code len var
  for entry in "$@"; do
    IFS=$'\t' read -r file dest url size sha <<< "$entry"
    rc=0
    out="$(wget --spider --server-response --tries=1 --timeout=20 "$url" 2>&1)" || rc=$?
    code="$(printf '%s\n' "$out" | grep -oE 'HTTP/[0-9.]+ [0-9]{3}' | tail -1 | awk '{print $2}')"
    len="$(printf '%s\n' "$out" | grep -iE '^ *Content-Length:' | tail -1 | awk '{print $2}')"
    case "${code:-none}" in
      200|206)
        if [[ -n "$len" && "$len" != "$size" ]]; then
          printf '  \033[1;31m[SIZE]\033[0m  %s\n           server serves %s, recipe says %s — the recipe is stale\n' \
            "$file" "$(_recipe_size "$len")" "$(_recipe_size "$size")"
          bad+=("$file")
        fi ;;
      401|403)
        var=""
        case "$url" in
          https://huggingface.co/*) var=HF_TOKEN ;;
          https://civitai.com/*)    var=CIVITAI_TOKEN ;;
        esac
        if [[ -n "$var" && -z "${!var:-}" ]]; then
          printf '  \033[1;31m[GATED]\033[0m %s\n           HTTP %s. Accept the licence on the site, then set %s\n' \
            "$file" "$code" "$var"
        else
          printf '  \033[1;31m[GATED]\033[0m %s\n           HTTP %s, and the token in use was not accepted\n' \
            "$file" "$code"
        fi
        bad+=("$file") ;;
      404|410)
        printf '  \033[1;31m[GONE]\033[0m  %s\n           HTTP %s — the url in the recipe has rotted\n' "$file" "$code"
        bad+=("$file") ;;
      *)
        printf '  \033[1;31m[NET]\033[0m   %s\n           unreachable (wget exit %s)\n' "$file" "$rc"
        bad+=("$file") ;;
    esac
  done
  (( ${#bad[@]} == 0 ))
}

# Is a node id already an active entry in the node list? Compares the base id
# so 'foo' and 'foo@1.2.3' are recognised as the same node.
_recipe_node_listed() {
  [[ -f "$NODES_LIST" ]] || return 1
  awk -v want="${1%%@*}" '
    { sub(/#.*/, ""); gsub(/^[ \t]+|[ \t]+$/, "")
      if ($0 == "") next
      base = $0; sub(/@.*/, "", base)
      if (base == want) found = 1 }
    END { exit found ? 0 : 1 }
  ' "$NODES_LIST"
}

# check and install share one engine: install is check plus the actions, so
# what you are told is always exactly what would happen.
# $1 = check|install, then NAME, then flags.
_recipe_apply() {
  local mode="$1"; shift
  local name="${1:-}"; shift || true
  [[ -n "$name" ]] || die "usage: $0 recipe $mode NAME"
  local yes=0 force=0 verify=0 arg
  for arg in "$@"; do
    case "$arg" in
      --yes|-y)  yes=1 ;;
      --force)   force=1 ;;
      --verify)  verify=1 ;;
      *) die "Unknown recipe $mode option: $arg (valid: --yes, --force, --verify)" ;;
    esac
  done

  local json parsed
  json="$(_recipe_path "$name")"
  [[ -f "$json" ]] || die "no such recipe: $name (looked for $json)"
  parsed="$(_recipe_read "$json")" || die "recipe '$name' is not valid — see the error above"

  resolve_mounts
  local models_dir
  models_dir="$(_mount_path models)"

  local tag f1 f2 f3 f4 f5
  local -a nodes=() flags=() todo=()
  local rname="" rnote="" workflow=""
  local have=0 want=0 bad_size=0 todo_bytes=0
  hdr "$name"
  while IFS=$'\t' read -r tag f1 f2 f3 f4 f5; do
    case "$tag" in
      META)
        case "$f1" in
          name)     rname="$f2" ;;
          run_note) rnote="$f2" ;;
          workflow) workflow="$f2" ;;
        esac ;;
      NODE) nodes+=("$f1") ;;
      FLAG) flags+=("$f1") ;;
      MODEL)
        want=$((want+1))
        local target="$models_dir/$f2/$f1" actual
        if [[ -f "$target" ]]; then
          actual="$(stat -c %s "$target" 2>/dev/null || echo 0)"
          if [[ "$actual" != "$f4" ]]; then
            printf '  \033[1;31m[SIZE]\033[0m %s/%s — on disk %s, recipe says %s\n' \
              "$f2" "$f1" "$(_recipe_size "$actual")" "$(_recipe_size "$f4")"
            bad_size=$((bad_size+1))
            if (( force )); then todo+=("$f1"$'\t'"$f2"$'\t'"$f3"$'\t'"$f4"$'\t'"$f5"); todo_bytes=$((todo_bytes+f4)); fi
          elif (( verify )) && [[ "$(sha256sum "$target" | awk '{print $1}')" != "$f5" ]]; then
            printf '  \033[1;31m[HASH]\033[0m %s/%s — size matches, sha256 does not\n' "$f2" "$f1"
            bad_size=$((bad_size+1))
            if (( force )); then todo+=("$f1"$'\t'"$f2"$'\t'"$f3"$'\t'"$f4"$'\t'"$f5"); todo_bytes=$((todo_bytes+f4)); fi
          else
            have=$((have+1))
            printf '  \033[1;32m[have]\033[0m %s/%s\n' "$f2" "$f1"
          fi
        else
          printf '  \033[1;33m[want]\033[0m %s/%s (%s)\n' "$f2" "$f1" "$(_recipe_size "$f4")"
          todo+=("$f1"$'\t'"$f2"$'\t'"$f3"$'\t'"$f4"$'\t'"$f5")
          todo_bytes=$((todo_bytes+f4))
        fi ;;
    esac
  done <<< "$parsed"

  echo
  [[ -n "$rname" ]] && echo "  $rname"
  printf '  models: %s of %s present' "$have" "$want"
  (( ${#todo[@]} )) && printf ', %s to download (%s)' "${#todo[@]}" "$(_recipe_size "$todo_bytes")"
  printf '\n'
  # A size or hash mismatch is reported and the file is LEFT ALONE. Models are
  # expensive user data; replacing one silently because a recipe disagrees is
  # not a call this tool gets to make. --force opts in.
  if (( bad_size && ! force )); then
    warn "$bad_size model(s) on disk do not match the recipe and were left untouched
  re-download them with: $0 recipe $mode $name --force"
  fi
  (( verify )) || info "sizes checked, not hashes — full verification: $0 recipe check $name --verify"

  local n missing_nodes=()
  for n in "${nodes[@]}"; do
    _recipe_node_listed "$n" || missing_nodes+=("$n")
  done
  (( ${#missing_nodes[@]} )) \
    && printf '  nodes: %s to add to %s\n' "${#missing_nodes[@]}" "$(basename "$NODES_LIST")"

  # Preflight before anything is downloaded, and before the confirmation
  # prompt, so what you agree to is known-reachable rather than hoped-for.
  local preflight_ok=1
  if (( ${#todo[@]} )); then
    echo
    info "checking ${#todo[@]} source(s)"
    _recipe_preflight "${todo[@]}" || preflight_ok=0
  fi

  if [[ "$mode" == check ]]; then
    echo
    if (( preflight_ok )); then
      echo "  Install it: $0 recipe install $name"
    else
      warn "the problems above must be fixed before this recipe can install"
    fi
    return 0
  fi
  (( preflight_ok )) \
    || die "refusing to start: the sources above are not all fetchable
Nothing was downloaded. Fix them and re-run."

  # ---- install ----
  if (( ${#todo[@]} )); then
    if (( ! yes )); then
      [[ -t 0 ]] || die "stdin is not a terminal — re-run with: $0 recipe install $name --yes"
      local answer
      read -r -p "  Download the above ($(_recipe_size "$todo_bytes"))? [y/N] " answer
      [[ "$answer" == y || "$answer" == Y ]] || die "aborted — nothing downloaded"
    fi
    log "Downloading ${#todo[@]} model(s), $(_recipe_size "$todo_bytes")"
    local entry file dest url size sha fails=0 idx=0 all_done=0 job_t0
    job_t0=$(date +%s)
    for entry in "${todo[@]}"; do
      IFS=$'\t' read -r file dest url size sha <<< "$entry"
      idx=$((idx+1))
      mkdir -p "$models_dir/$dest" \
        || die "cannot create $models_dir/$dest — check the models mount"
      printf '\n  \033[1m[%d/%d]\033[0m %s\n         %s → %s\n' \
        "$idx" "${#todo[@]}" "$file" "$(_recipe_size "$size")" "$dest/"
      if _recipe_fetch "$url" "$models_dir/$dest/$file" "$sha" \
           "$size" "$all_done" "$todo_bytes"; then
        all_done=$((all_done+size))
      else
        fails=$((fails+1))
      fi
    done
    local job_secs=$(( $(date +%s) - job_t0 ))
    printf '\n  %s of %s in %dm%02ds\n' \
      "$(_recipe_size "$all_done")" "$(_recipe_size "$todo_bytes")" \
      "$((job_secs/60))" "$((job_secs%60))"
    (( fails )) && warn "$fails model(s) did not install — re-run to resume"
  fi

  if (( ${#missing_nodes[@]} )); then
    log "Custom nodes"
    seed_nodes_list
    for n in "${missing_nodes[@]}"; do
      printf '%s\n' "$n" >> "$NODES_LIST"
      echo "  + $n"
    done
    echo "  they install on the next start: $0 run"
  fi

  if [[ -n "$workflow" ]]; then
    log "Workflow"
    local user_dir wf_dst wf_tmp
    user_dir="$(_mount_path user)"
    wf_dst="$user_dir/default/workflows/$name.json"
    mkdir -p "$(dirname "$wf_dst")"
    # Extracted to a temp file first: writing straight to wf_dst would
    # clobber an existing workflow before we could compare and save it aside.
    wf_tmp="$(mktemp)"
    if python3 "$LIB_DIR/recipe_read.py" "$json" --extract-workflow "$wf_tmp"; then
      if [[ -f "$wf_dst" ]] && ! cmp -s "$wf_tmp" "$wf_dst"; then
        cp -a "$wf_dst" "$wf_dst.bak"
        warn "a different $name.json was already there — saved aside as $name.json.bak"
      fi
      mv -f "$wf_tmp" "$wf_dst"
      chmod 644 "$wf_dst"
      echo "  $wf_dst ($workflow)"
    else
      rm -f "$wf_tmp"
      warn "could not extract the workflow from the recipe"
    fi
  fi

  # Advice only. Installing a recipe must never change how anything launches:
  # these flags are global, and one workflow's requirement is not a reason to
  # alter every other workflow's launch.
  if (( ${#flags[@]} )); then
    log "This workflow needs"
    printf '  %s %s run\n' "${flags[*]}" "$0"
    [[ -n "$rnote" ]] && echo "  ($rnote)"
  fi
  log "Recipe '$name' applied"
}

# Author a recipe from a workflow that already works on this box. Without
# this, every model entry is hand-written; with it, dest/size/sha256 come
# from the files on disk and only the URLs need filling in.
_recipe_capture() {
  local workflow="" name="" arg
  while (( $# )); do
    case "$1" in
      --name) shift; name="${1:-}" ;;
      --name=*) name="${1#*=}" ;;
      -*) die "Unknown recipe capture option: $1" ;;
      *)  workflow="$1" ;;
    esac
    shift || true
  done
  [[ -n "$workflow" ]] \
    || die "usage: $0 recipe capture WORKFLOW [--name NAME]
WORKFLOW is a saved workflow's name, or a path to a .json file."

  resolve_mounts
  # Take a workflow NAME, not just a path. People think in the names they see
  # in the ComfyUI sidebar, and requiring the full path to a file the tool
  # already knows how to find is friction for no reason. A path still works
  # and is tried first, so nothing that used to work stops working.
  local wf_dir
  wf_dir="$(_mount_path user)/default/workflows"
  local cand found=""
  for cand in "$workflow" "$workflow.json" "$wf_dir/$workflow" "$wf_dir/$workflow.json"; do
    [[ -f "$cand" ]] && { found="$cand"; break; }
  done
  if [[ -z "$found" ]]; then
    warn "no workflow '$workflow' — not a file, and not saved under $wf_dir"
    if compgen -G "$wf_dir/*.json" >/dev/null; then
      echo "  saved workflows:"
      local w
      for w in "$wf_dir"/*.json; do echo "    $(basename "$w" .json)"; done
    fi
    die "name one of the above, or give a path to a .json file"
  fi
  workflow="$found"

  # Default the recipe name to the workflow's, so the common case is one
  # argument. --name still overrides it.
  [[ -n "$name" ]] || name="$(basename "$workflow" .json)"
  local out
  out="$(_recipe_path "$name")"
  [[ -e "$out" ]] && die "recipe '$name' already exists at $out
(remove it first, or pass --name — capture never overwrites)"
  mkdir -p "$RECIPES_DIR"
  local models_dir
  models_dir="$(_mount_path models)"

  # Manager's model-list.json is a 509-entry catalogue with exactly the
  # fields a recipe needs (filename/save_path/url/size). Read it out of the
  # image so it is always the version actually installed. Optional: capture
  # works without it, the URLs just stay blank.
  # Via a temp FILE, not the environment: the catalogue is ~3 MB and passing
  # it as an env var overflows the exec argument limit (field-hit at first
  # run: "Argument list too long").
  local catalog=""
  if docker image inspect "$CONTAINER_IMAGE:latest" >/dev/null 2>&1; then
    catalog="$(mktemp)"
    # shellcheck disable=SC2064
    trap "rm -f '$catalog'" RETURN
    docker run --rm --entrypoint cat "$CONTAINER_IMAGE:latest" \
      /opt/comfyui-env/lib/python3.12/site-packages/comfyui_manager/model-list.json \
      > "$catalog" 2>/dev/null || catalog=""
  fi

  log "Capturing '$name' from $(basename "$workflow")"
  if ! MODELS_DIR="$models_dir" RECIPE_NAME="$name" CATALOG_FILE="$catalog" \
       RECIPES_DIR="$RECIPES_DIR" \
       python3 "$LIB_DIR/recipe_capture.py" "$workflow" "$out"
  then
    rm -f "$out"
    die "capture failed — nothing written"
  fi
  chmod 644 "$out"
  echo "  wrote $out ($(_recipe_size "$(stat -c %s "$out")"))"
}

cmd_recipe() {
  local sub="${1:-}"
  shift || true
  case "$sub" in
    list)     _recipe_list ;;
    show)     _recipe_show "$@" ;;
    check)    _recipe_apply check "$@" ;;
    install)  _recipe_apply install "$@" ;;
    capture)  _recipe_capture "$@" ;;
    ""|-h|--help)
      echo "  $0 recipe list                      what is available"
      echo "  $0 recipe show NAME                 models, nodes and flags"
      echo "  $0 recipe check NAME [--verify]     what is missing, nothing downloaded"
      echo "  $0 recipe install NAME [--yes] [--force]"
      echo "  $0 recipe capture WORKFLOW --name NAME"
      echo
      echo "  Gated downloads: HF_TOKEN / CIVITAI_TOKEN from the environment." ;;
    *) die "unknown recipe subcommand: $sub
(valid: list, show, check, install, capture — see: $0 recipe --help)" ;;
  esac
}

