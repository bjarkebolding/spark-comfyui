# shellcheck shell=bash
# =============================================================================
#  mod: 10-unified-memory-free
#  Makes ComfyUI's get_free_memory() report the host-available unified pool
#  instead of the CUDA free query, which under-reports on GB10 when another
#  CUDA process (e.g. vLLM) is resident.
#  Patches: comfy/model_management.py
# =============================================================================
MOD_TAG="mem_unified"
MOD_FILE="comfy/model_management.py"

mod_describe() {
  echo "unified-memory-aware get_free_memory() (fixes offload cliff with co-resident CUDA procs)"
}

mod_apply() {
  py_patch_file "$MOD_FILE" "$MOD_TAG" "$MOD_DIR/transform.py"
}

mod_verify() {
  py_marker_present "$MOD_FILE" "$MOD_TAG"
}
