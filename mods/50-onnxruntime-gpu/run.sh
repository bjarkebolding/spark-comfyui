# shellcheck shell=bash
# =============================================================================
#  mod: 50-onnxruntime-gpu
#  DWPose/ControlNet preprocessors run on onnxruntime. PyPI ships no
#  aarch64+cu13 GPU wheel, so without the community sm_121 wheel they
#  silently fall back to CPU. Also guards the shadow trap: a later
#  'pip install onnxruntime' (e.g. pulled in by a custom node) can overwrite
#  the GPU wheel via the shared import path with no pip conflict — re-run on
#  every update to catch that. Not critical: today's failure mode is a CPU
#  fallback with a warning, not a broken install, so it stays soft like the
#  source-patch mods. Streamed: ensure_onnx_gpu prints its own progress/log
#  lines (and can download a ~220 MB wheel on first install) — buffering
#  that until mod_apply returns would both hide progress AND corrupt the
#  buffered "first line is the status" protocol with its extra echoed lines,
#  so status is reported via mod_export instead.
# =============================================================================
# shellcheck disable=SC2034  # read by _run_mod's 'flags' verb in the main script
MOD_STREAM=1

mod_describe() {
  echo "GPU onnxruntime (sm_121) for DWPose/ControlNet preprocessors"
}

mod_apply() {
  local was_active=1
  mod_verify || was_active=0
  ensure_onnx_gpu
  mod_export "ORT_STATE=${ORT_STATE:-unknown}"
  if [[ "$was_active" -eq 1 ]]; then
    mod_export "STATUS=present ${ORT_STATE:-unknown}"
  else
    mod_export "STATUS=applied ${ORT_STATE:-unknown}"
  fi
}

mod_verify() {
  onnx_gpu_ok
}
