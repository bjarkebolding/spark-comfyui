# syntax=docker/dockerfile:1
# spark-comfyui container image.
#
# The image holds everything reproducible: ComfyUI at a pinned commit (plus
# the user's comfyui-patches.list merged on top, when present), the venv
# with cu130 torch, a native sm_121 SageAttention build, the GPU onnxruntime
# wheel, and the build-time mods. User content is bind-mounted at run time
# per the mount contract (spark-mounts.conf, resolve_mounts in the script).
#
# No GPU exists at image-build time, so the live kernel gates (golden rule 3)
# run in the entrypoint on every start instead of here.
#
# Stages are named so 'container update --torch' can bust exactly the torch
# stage with --no-cache-filter=torch: everything from the torch install down
# (SageAttention included) rebuilds, the apt/base stage stays cached. The
# chain is linear; this is not a slimming multi-stage (that is a phase 4
# item).

# Pinned to the 13.0 patch line on purpose: the host driver (r580) is a
# CUDA 13.0 driver and torch's cu130 wheels are 13.0 builds. 13.1+ images
# ship a newer ptxas whose embedded PTX the driver JIT may reject ("no
# kernel image", the GB10 classic) for zero benefit — sm_121 is fully
# supported since 13.0.0. Bumping past 13.0.x is a deliberate pin change
# with a field test, same policy as SAGE_REF.
#
# Both base images are pinned by DIGEST as well as by tag. The tag says what
# we mean, the digest says which bytes we tested: a tag is mutable and both
# of these are re-pushed regularly, so without the digest two builds a month
# apart are not the same image. These are INDEX (multi-platform) digests, so
# they resolve on aarch64 and on anything else. Refresh either one with:
#   docker buildx imagetools inspect <ref> | awk '/^Digest:/{print $2}'
# and treat it like SAGE_REF: a deliberate change with a build behind it,
# never a drive-by edit. A stale pin is not a security hole on its own —
# every package we actually install comes from a fresh apt/pip fetch at build
# time — but it does mean the untouched base layers age, so refresh both when
# you bump CUDA or ComfyUI for real.
ARG CUDA_IMAGE=nvidia/cuda:13.0.3-devel-ubuntu24.04@sha256:b7ae301dea2c162444795462ce17a05f6a516e5a75944b57af5b88540a1a2266
# The runtime stage's own base. Named so it is pinned in one place with the
# other one, instead of hiding as a literal 150 lines down.
ARG RUNTIME_IMAGE=ubuntu:24.04@sha256:33ceb71981b602c1a7443a53469e4dba065f7503eab3078a2d7a57a2ab987517
ARG TORCH_INDEX=https://download.pytorch.org/whl/cu130
# The torch trio is PINNED, and this was the last unpinned input in the file.
# On 2026-09-05 a remote PyTorch release (2.14.0) broke every cold build with
# no change on our side: SageAttention could not compile against it. That was
# the loud failure. The quiet one is worse, because torch is the most
# behaviour-defining dependency here and every measured number in CLAUDE.md
# (the attention A/B, the aimdo and get_free_memory findings, NVFP4) was taken
# on 2.13.0. An unannounced bump hands the user a different machine from the
# one those numbers describe, which is exactly the silent degradation golden
# rule 5 forbids.
#
# Pinned does NOT mean frozen. Emptying these three restores floating
# behaviour, which is precisely what 'update --torch' does, and 'doctor'
# reports when a newer torch exists so the decision gets made rather than
# forgotten. Moving the pin is a deliberate edit with a field test behind it,
# same policy as SAGE_REF.
ARG TORCH_VERSION=2.14.0
ARG TORCHVISION_VERSION=0.29.0
ARG TORCHAUDIO_VERSION=2.11.0
ARG REPO_URL=https://github.com/Comfy-Org/ComfyUI.git
ARG COMFY_SHA=master
ARG SAGE_REPO=https://github.com/thu-ml/SageAttention.git
ARG SAGE_REF=d1a57a546c3d395b1ffcbeecc66d81db76f3b4b5
# Official PyPI onnxruntime-gpu. PyPI GPU packages have been CUDA 13 builds
# since 1.27 and aarch64 wheels appeared with them, which retires the
# community sm_121 wheel this used to point at: one file, one HuggingFace
# user, no fallback. Verified 2026-08-29 on GB10 with a real CUDA-only
# session, identical per-run time and bit-identical output to the wheel it
# replaces. Pinned as a URL rather than a version so the #sha256= fragment
# still makes pip verify the exact bytes; that also pins the platform tag,
# so a future Python bump fails the build loudly instead of silently
# resolving some other wheel.
ARG ORT_WHEEL_URL=https://files.pythonhosted.org/packages/fa/96/1be0b9711a614861fbf5c0c85c4da3aa2321f9da0b4734a717eb606f70ab/onnxruntime_gpu-1.29.0-cp312-cp312-manylinux_2_34_aarch64.whl#sha256=545d2966dd11208bbc98e67be4d92e54ecd793acc45c0532cc7bfd36f50692fa
ARG TORCH_CUDA_ARCH_LIST=12.1+PTX
# There is deliberately NO UID build arg, and adding one is a regression.
# The image bakes no host identity: it is built as uid 1000 internally and
# RUN AS THE CALLER at launch (--user in _container_run_args), with /opt and
# /home/comfy group-0 writable so any uid can write the venv and the ComfyUI
# tree. Baking the host's uid would put one identity in two places, the image
# and the host directories, and every drift between them is a permission
# failure. Field-reported 2026-08-23 by a GX10 user whose account was uid 1001
# because a second cluster user had taken 1000.

# --------------------------------------------------------------------------
FROM ${CUDA_IMAGE} AS base
ENV DEBIAN_FRONTEND=noninteractive
# The X client libs are for ComfyUI's stock GLSL nodes (comfy_extras/
# nodes_glsl.py): the bundled ANGLE libGLESv2 links libX11/libXext/libxcb
# even headless, and without them the node import fails (field-found on the
# first container launch, 2026-07-19).
RUN apt-get update && apt-get install -y --no-install-recommends \
      git ca-certificates python3 python3-venv python3-dev python3-pip \
      libx11-6 libxext6 libxcb1 \
    && rm -rf /var/lib/apt/lists/*

# ubuntu24.04 ships an 'ubuntu' user on uid 1000; replace it with comfy. The
# BUILD always runs as uid 1000, which is why the BuildKit cache mounts below
# can name it literally. The shipped image does not depend on that number: see
# the group-0 pass at the end of the final stage.
RUN userdel -r ubuntu 2>/dev/null || true \
 && useradd -m -u 1000 comfy \
 && mkdir -p /opt/spark && chown -R comfy:comfy /opt

USER comfy
WORKDIR /opt

# --------------------------------------------------------------------------
FROM base AS torch
ARG TORCH_INDEX
# Network resilience for every pip call in the image, at build time and at
# launch. pip's own default socket timeout is 15s, which is thin for the
# wheels involved here (cu130 torch is >1 GB, the GPU onnxruntime ~57 MB): one
# slow moment on a long download becomes a failed build rather than a retry.
#
# These lived in spark-comfyui.sh as host-side exports until 2026.08.08. That
# was dead code the moment the native path went: nothing on the host runs pip
# any more, and neither 'docker build' nor 'docker run' forwards host
# environment, so the protection the comment described had quietly not existed
# since the container cut. Verified in-image before moving them here: pip
# resolves timeout=120 and retries from exactly these names.
#
# PIP_RETRIES is set to pip's current default on purpose — it pins the value
# rather than inheriting whatever a future pip decides, the same reasoning as
# every other pin in this file.
ENV PIP_DEFAULT_TIMEOUT=120 \
    PIP_RETRIES=5
# venv first; torch BEFORE ComfyUI's requirements so nothing can pull a
# CPU-only torch first (same golden rule as the native install).
RUN python3 -m venv /opt/comfyui-env
ENV VIRTUAL_ENV=/opt/comfyui-env
ENV PATH=/opt/comfyui-env/bin:${PATH}
ARG TORCH_VERSION
ARG TORCHVISION_VERSION
ARG TORCHAUDIO_VERSION
# The ${VAR:+==${VAR}} form is what makes an EMPTY arg mean "unpinned", so
# 'update --torch' can hand back the floating behaviour by passing nothing.
RUN --mount=type=cache,target=/home/comfy/.cache/pip,uid=1000 \
    pip install --upgrade pip wheel \
 && pip install \
      "torch${TORCH_VERSION:+==${TORCH_VERSION}}" \
      "torchvision${TORCHVISION_VERSION:+==${TORCHVISION_VERSION}}" \
      "torchaudio${TORCHAUDIO_VERSION:+==${TORCHAUDIO_VERSION}}" \
      --index-url ${TORCH_INDEX}

# --------------------------------------------------------------------------
FROM torch AS sage
ARG SAGE_REPO
ARG SAGE_REF
ARG TORCH_CUDA_ARCH_LIST
# Native sm_121 SageAttention, pinned to SAGE_REF (compile only; the live
# kernel verification happens in the entrypoint, where a GPU exists).
#
# CXX_APPEND_FLAGS / NVCC_APPEND_FLAGS are SageAttention's OWN documented
# escape hatch (setup.py reads both and appends them AFTER its hardcoded
# flags), and the last -std wins for gcc and nvcc. That matters because
# setup.py hardcodes -std=c++17 while torch 2.14.0 raised its floor to C++20:
# torch/csrc/api/include/torch/all.h and ATen/ATen.h both #error out below
# 202002L, where 2.13.0 only wanted 201703L. Field-reported 2026-09-05 and
# reproduced here.
#
# Do NOT "fix" this by sed-ing setup.py. Patching upstream source buys a
# maintenance burden that this env knob does not, and it breaks silently if
# SageAttention ever restructures the file. Setting the standard explicitly
# is also what torch itself does: torch/utils/cpp_extension.py has compiled
# extensions with -std=c++20 since at least 2.13.
RUN --mount=type=cache,target=/home/comfy/.cache/pip,uid=1000 \
    git clone ${SAGE_REPO} /opt/SageAttention \
 && git -C /opt/SageAttention checkout ${SAGE_REF} \
 && pip install ninja packaging \
 && cd /opt/SageAttention \
 && CXX_APPEND_FLAGS="-std=c++20" NVCC_APPEND_FLAGS="-std=c++20" \
    TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST} MAX_JOBS=$(nproc) \
      pip install --no-build-isolation --no-deps .

# --------------------------------------------------------------------------
FROM sage AS final
ARG REPO_URL
ARG COMFY_SHA
ARG ORT_WHEEL_URL
ARG SAGE_REF
ARG TORCH_INDEX

# ComfyUI at the exact commit cmd_build resolved (the ARG value is
# the cache key: a new upstream commit rebuilds from this layer down, the
# torch and SageAttention stages above stay cached).
RUN git clone ${REPO_URL} /opt/ComfyUI \
 && git -C /opt/ComfyUI checkout ${COMFY_SHA}

# The user's patch list (PRs/branches merged on top of the pinned commit).
# The [t] glob is the standard optional-COPY trick: zero matches is not an
# error, so a build without a patch list works. A changed list busts the
# cache from here down, which is exactly the right rebuild scope.
COPY --chown=comfy:comfy container/build-patches.sh comfyui-patches.lis[t] /opt/spark/
RUN bash /opt/spark/build-patches.sh

RUN --mount=type=cache,target=/home/comfy/.cache/pip,uid=1000 \
    pip install -r /opt/ComfyUI/requirements.txt \
 && pip install -r /opt/ComfyUI/manager_requirements.txt

# Community sm_121 GPU onnxruntime; the #sha256= fragment makes pip verify
# the exact bytes.
RUN --mount=type=cache,target=/home/comfy/.cache/pip,uid=1000 \
    pip install "${ORT_WHEEL_URL}"

# The mod pass: setuptools-compat and unified-memory-free. Both need no GPU
# and no user content, which is now the definition of a mod rather than a
# property two of them happened to share.
COPY --chown=comfy:comfy mods/ /opt/spark/mods/
# --chmod=755 is load-bearing, not tidiness. COPY reproduces the build-context
# mode verbatim and the runtime stage carries it through, so a tree that
# arrived without the execute bit (a downloaded zip, a copy that dropped
# modes) baked a non-executable entrypoint into an image that still built
# with exit 0. The failure then surfaced at run time as an OCI "permission
# denied" naming only a path inside the container. Field-reported 2026-08-05.
# The image must not inherit that bit from the user's disk.
COPY --chown=comfy:comfy --chmod=755 \
     container/build-mods.sh container/entrypoint.sh \
     container/manager-config.py container/gb10.sh container/resolve-node.py \
     /opt/spark/
ENV INSTALL_DIR=/opt/ComfyUI
ENV VENV_DIR=/opt/comfyui-env
RUN bash /opt/spark/build-mods.sh

# Group 0 rights over everything the runtime stage carries. The container runs
# as the CALLER's uid, which is unknown at build time, so an arbitrary uid must
# still be able to write the venv (the entrypoint pip-installs custom-node
# requirements on EVERY start) and the ComfyUI tree (__pycache__, temp/).
# Group 0 is the one group every container process is given, via --group-add 0.
# g=u grants the group exactly the owner's rights; g+s on directories makes
# files created later inherit group 0 rather than the caller's primary group.
#
# This MUST live here and not in the runtime stage. A recursive chmod there
# would rewrite every file of a 6.2 GB venv into a fresh layer and roughly
# double the shipped image. Here it costs a build-stage layer that is never
# shipped, because runtime is a fresh FROM ubuntu:24.04 that COPYs from this
# stage, and a COPY writes new files at the mode of the source.
USER root
RUN chgrp -R 0 /opt \
 && chmod -R g=u /opt \
 && find /opt -type d -exec chmod g+s {} +
USER comfy

# --------------------------------------------------------------------------
# Slim runtime stage (phase 4). Everything above runs in the devel context;
# the shipped image is this one: plain ubuntu + the venv + ComfyUI + the
# spark scripts. What the devel base provided and the runtime still needs:
#  - ptxas (31 MB, copied): Triton JITs kernels at runtime and torch's
#    bundled ptxas is pre-CUDA-13 (can't target sm_121, triton#10331).
#  - NVRTC: NOT copied — the cu130 wheels bundle nvidia-cuda-nvrtc in the
#    venv nowadays (verified in-image 2026-07-20).
# build-essential and python3-dev stay: the entrypoint pip-installs
# custom-node requirements at start, and an sdist dep needs a compiler.
# The SageAttention build tree is NOT copied (246 MB of build residue; the
# compiled package lives in the venv).
FROM ${RUNTIME_IMAGE} AS runtime
ARG COMFY_SHA
ARG SAGE_REF
ARG TORCH_INDEX
ARG ORT_WHEEL_URL
ENV DEBIAN_FRONTEND=noninteractive
# Two separate reasons this slim runtime image carries graphics libraries at
# all, neither of which is about drawing anything:
#   libx11-6 libxext6 libxcb1  ComfyUI's stock GLSL nodes dlopen a bundled
#                              ANGLE that links libX11/libXext/libxcb even
#                              headless.
#   libgl1 libglib2.0-0t64     opencv-python links libGL.so.1, libglib-2.0
#                              and libgthread-2.0. It is one of the most
#                              common custom-node dependencies, so without
#                              these any node importing cv2 dies with
#                              "libGL.so.1: cannot open shared object file"
#                              and ComfyUI carries on without it (field-hit
#                              2026-08-05 with comfyui-videohelpersuite).
#                              Exactly these two packages: they cover the
#                              three libraries ldd actually reported missing,
#                              and nothing else was.
# NOTE the t64 suffix. Ubuntu 24.04's time_t transition renamed the glib
# package, and plain 'libglib2.0-0' has NO installation candidate here, so
# the name every search result gives you fails the build.
# git is for Manager's node installs inside the container.
RUN apt-get update && apt-get install -y --no-install-recommends \
      git ca-certificates python3 python3-venv python3-dev python3-pip \
      build-essential \
      libx11-6 libxext6 libxcb1 \
      libgl1 libglib2.0-0t64 \
    && rm -rf /var/lib/apt/lists/*
# /etc/passwd is left group writable so the entrypoint can add an entry for
# whatever uid it is running as. Without one, getpwuid() raises and anything
# resolving the current user breaks. It grants nothing: the process is already
# non-root, the container drops ALL capabilities and sets no-new-privileges,
# and there is no setuid binary for a forged entry to reach.
RUN userdel -r ubuntu 2>/dev/null || true \
 && useradd -m -u 1000 comfy \
 && mkdir -p /opt && chown comfy:0 /opt && chmod g=u /opt \
 && chmod g+w /etc/passwd

COPY --from=final --chown=comfy:0 /opt/comfyui-env /opt/comfyui-env
COPY --from=final --chown=comfy:0 /opt/ComfyUI /opt/ComfyUI
COPY --from=final --chown=comfy:0 /opt/spark /opt/spark
COPY --from=base /usr/local/cuda/bin/ptxas /usr/local/cuda/bin/ptxas

# COPY creates the DESTINATION DIRECTORY itself at a default 0755 and ignores
# the source directory's mode, so the three roots above land without the group
# rights the final stage set, while everything inside them keeps them. Caught
# by the arbitrary-uid test 2026-08-23: files were 664 group root, but the tree
# roots were 755, so an arbitrary uid could not create anything in them.
# Three directories, so this is not the recursive chmod the final stage exists
# to avoid; it adds a layer of three inodes.
RUN chmod g+ws /opt/comfyui-env /opt/ComfyUI /opt/spark

# The cache volume mounts at /home/comfy/.cache; the dir must exist in the
# image OWNED BY comfy, or docker creates the mountpoint (and initializes a
# fresh named volume) as root and pip/uv/CUDA all fail to write. Group 0 is
# what makes it survive a change of caller: docker copies ownership from this
# mountpoint the first time it populates the volume and never revisits it, so
# a uid-specific owner here would strand the cache for every other user.
RUN mkdir -p /home/comfy/.cache/pip /home/comfy/.cache/uv /home/comfy/.cache/nv-compute \
 && chown -R comfy:0 /home/comfy \
 && chmod -R g=u /home/comfy \
 && find /home/comfy -type d -exec chmod g+s {} +

# The fallback identity, for anything that runs this image without --user:
# doctor's throwaway gate containers do exactly that. The launch path overrides
# it with the caller's uid.
USER comfy

# Runtime environment, mirroring the native-era cmd_run: 4 GB CUDA kernel
# cache (redirected into the cache volume so compiled sm_121 kernels
# survive container recreation), NCCL P2P off, Triton pointed at the
# copied CUDA 13 ptxas.
# uv (the entrypoint's preferred installer for custom-node requirements) is
# pointed at the same cache volume, so node deps survive container recreation
# exactly like pip's downloads do. UV_LINK_MODE=copy because the cache volume
# and the venv are different filesystems: uv would try to hardlink, fail, warn,
# and fall back to copying anyway on every single install.
# LD_LIBRARY_PATH points at torch's own pip-installed CUDA libraries, and it
# is not a convenience. Without it, onnxruntime's CUDA provider only loads in
# a process that imported torch FIRST, because that import is what pulls
# libcublasLt.so.13 and friends into the process. ComfyUI does import torch
# first, so the server was fine, but anything reaching onnxruntime on its own
# (a node shelling out to a subprocess, our own gate) got a session that
# silently fell back to CPU while get_available_providers() still said CUDA.
# Measured 2026-08-29: with this set, an ORT CUDA session builds and runs
# correctly with no torch import at all. Verified in the same pass that torch,
# the SageAttention kernel gate, the NVFP4 gate and the stdlib C extensions
# are all unaffected, i.e. nothing here shadows a system library.
ENV LD_LIBRARY_PATH=/opt/comfyui-env/lib/python3.12/site-packages/nvidia/cu13/lib:/opt/comfyui-env/lib/python3.12/site-packages/nvidia/cudnn/lib
ENV HOME=/home/comfy \
    USER=comfy \
    LOGNAME=comfy \
    VIRTUAL_ENV=/opt/comfyui-env \
    PATH=/opt/comfyui-env/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin \
    INSTALL_DIR=/opt/ComfyUI \
    VENV_DIR=/opt/comfyui-env \
    TORCH_INDEX=${TORCH_INDEX} \
    ORT_WHEEL_URL=${ORT_WHEEL_URL} \
    CUDA_CACHE_PATH=/home/comfy/.cache/nv-compute \
    CUDA_CACHE_MAXSIZE=4294967296 \
    NCCL_P2P_DISABLE=1 \
    TRITON_PTXAS_PATH=/usr/local/cuda/bin/ptxas \
    UV_CACHE_DIR=/home/comfy/.cache/uv \
    UV_LINK_MODE=copy \
    PIP_DEFAULT_TIMEOUT=120 \
    PIP_RETRIES=5 \
    UV_HTTP_TIMEOUT=120
# The three timeout/retry values above are the launch-path half of the same
# concern as the torch stage: 'run' is --rm, so the custom-node requirements
# pass happens on EVERY start, and a stalled connection there holds up a
# server that is not serving yet. uv is the preferred installer on that path
# and does not read PIP_*; its own default is 30s. UV_HTTP_TIMEOUT is a real
# knob in uv 0.12.3, confirmed the only way that proves a variable is read —
# a deliberately invalid value makes uv refuse to run and name the variable.

# Consumed by backup (comfyui_commit in meta) and doctor on hosts that have
# no ComfyUI checkout of their own.
#
# The LABEL is the ONLY place SAGE_REF survives into the shipped image. It was
# also an ENV until 2026-08-08, left behind when mod 40 (its sole reader) was
# deleted with the native path. TORCH_INDEX and ORT_WHEEL_URL above are NOT in
# that category and must stay: ensure_torch_cuda and ensure_onnx_gpu read them
# out of the environment at run time.
LABEL org.spark-comfyui.comfy-sha=${COMFY_SHA} \
      org.spark-comfyui.sage-ref=${SAGE_REF}

EXPOSE 8188

# Liveness, for 'service' mode above all: a hung ComfyUI still looks "Up" to
# docker, whose restart policy only ever reacts to an exit. This does NOT make
# anything self-heal (restart policies ignore health by design) — it upgrades
# "the process exists" into "the server answers", which is what 'status' and
# 'doctor' report. /system_stats is a stock endpoint and needs no auth.
# start-period is generous because the entrypoint reconciles the registry
# node list, installs custom-node requirements and runs the GPU gates before
# main.py ever binds the port; a large node set legitimately takes minutes,
# and health stays "starting" (never "unhealthy") for that whole window.
# Raised 300s -> 600s when the node list landed: a COLD first start now
# downloads and installs nodes too, and a slow link must not be reported as
# a sick server. Nothing self-heals on health anyway, so the only cost of a
# wider window is later detection of a genuine hang.
HEALTHCHECK --interval=30s --timeout=5s --start-period=600s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8188/system_stats', timeout=4)" || exit 1

ENTRYPOINT ["/opt/spark/entrypoint.sh"]
