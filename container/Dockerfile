# syntax=docker/dockerfile:1
# spark-comfyui container image (phase 1, experimental).
#
# The image holds everything reproducible: ComfyUI at a pinned commit, the
# venv with cu130 torch, a native sm_121 SageAttention build, the GPU
# onnxruntime wheel, and the build-time mods. User content (the USER_CONTENT
# set: models, user, input, output, custom_nodes, extra_model_paths.yaml) is
# bind-mounted at run time by cmd_container_run.
#
# No GPU exists at image-build time, so the live kernel gates (golden rule 3)
# run in the entrypoint on every start instead of here.
#
# Single stage on the devel image for now; slimming via a multi-stage build
# is a later phase.

# Pinned to the 13.0 patch line on purpose: the host driver (r580) is a
# CUDA 13.0 driver and torch's cu130 wheels are 13.0 builds. 13.1+ images
# ship a newer ptxas whose embedded PTX the driver JIT may reject ("no
# kernel image", the GB10 classic) for zero benefit — sm_121 is fully
# supported since 13.0.0. Bumping past 13.0.x is a deliberate pin change
# with a field test, same policy as SAGE_REF.
ARG CUDA_IMAGE=nvidia/cuda:13.0.3-devel-ubuntu24.04
FROM ${CUDA_IMAGE}

ARG TORCH_INDEX=https://download.pytorch.org/whl/cu130
ARG REPO_URL=https://github.com/Comfy-Org/ComfyUI.git
ARG COMFY_SHA=master
ARG SAGE_REPO=https://github.com/thu-ml/SageAttention.git
ARG SAGE_REF=d1a57a546c3d395b1ffcbeecc66d81db76f3b4b5
ARG ORT_WHEEL_URL=https://huggingface.co/Jay0515/onnxruntime-gpu-aarch64-cuda13-sm121/resolve/main/onnxruntime_gpu-1.25.0-cp312-cp312-linux_aarch64.whl#sha256=da487cc1ccd3aa11389efec14c6f0f8b6bd7ca6734423de3b528e578023cb200
ARG TORCH_CUDA_ARCH_LIST=12.1+PTX
ARG UID=1000

ENV DEBIAN_FRONTEND=noninteractive
# The X client libs are for ComfyUI's stock GLSL nodes (comfy_extras/
# nodes_glsl.py): the bundled ANGLE libGLESv2 links libX11/libXext/libxcb
# even headless, and without them the node import fails (field-found on the
# first container launch, 2026-07-19).
RUN apt-get update && apt-get install -y --no-install-recommends \
      git ca-certificates python3 python3-venv python3-dev python3-pip \
      libx11-6 libxext6 libxcb1 \
    && rm -rf /var/lib/apt/lists/*

# ubuntu24.04 ships an 'ubuntu' user on uid 1000; replace it with comfy so
# files written to the bind-mounted user content keep the host owner's uid.
RUN userdel -r ubuntu 2>/dev/null || true \
 && useradd -m -u ${UID} comfy \
 && mkdir -p /opt/spark && chown -R comfy:comfy /opt

USER comfy
WORKDIR /opt

# venv first; torch BEFORE ComfyUI's requirements so nothing can pull a
# CPU-only torch first (same golden rule as the native install).
RUN python3 -m venv /opt/comfyui-env
ENV VIRTUAL_ENV=/opt/comfyui-env
ENV PATH=/opt/comfyui-env/bin:${PATH}
RUN --mount=type=cache,target=/home/comfy/.cache/pip,uid=1000 \
    pip install --upgrade pip wheel \
 && pip install torch torchvision torchaudio --index-url ${TORCH_INDEX}

# Native sm_121 SageAttention, pinned to SAGE_REF (compile only; the live
# kernel verification happens in the entrypoint, where a GPU exists).
RUN --mount=type=cache,target=/home/comfy/.cache/pip,uid=1000 \
    git clone ${SAGE_REPO} /opt/SageAttention \
 && git -C /opt/SageAttention checkout ${SAGE_REF} \
 && pip install ninja packaging \
 && cd /opt/SageAttention \
 && TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST} MAX_JOBS=$(nproc) \
      pip install --no-build-isolation --no-deps .

# ComfyUI at the exact commit cmd_container_build resolved (the ARG value is
# the cache key: a new upstream commit rebuilds from this layer down, the
# torch and SageAttention layers above stay cached).
RUN git clone ${REPO_URL} /opt/ComfyUI \
 && git -C /opt/ComfyUI checkout ${COMFY_SHA}
RUN --mount=type=cache,target=/home/comfy/.cache/pip,uid=1000 \
    pip install -r /opt/ComfyUI/requirements.txt \
 && pip install -r /opt/ComfyUI/manager_requirements.txt

# Community sm_121 GPU onnxruntime; the #sha256= fragment makes pip verify
# the exact bytes.
RUN --mount=type=cache,target=/home/comfy/.cache/pip,uid=1000 \
    pip install "${ORT_WHEEL_URL}"

# Build-time mod pass: 05-setuptools-compat and 10-unified-memory-free need
# no GPU and no user content. 20/40/50 are verified by the entrypoint; 30
# writes to the mounted user dir and also runs there.
COPY --chown=comfy:comfy mods/ /opt/spark/mods/
COPY --chown=comfy:comfy container/build-mods.sh container/entrypoint.sh /opt/spark/
ENV INSTALL_DIR=/opt/ComfyUI
ENV VENV_DIR=/opt/comfyui-env
RUN bash /opt/spark/build-mods.sh

# The cache volume mounts at /home/comfy/.cache; the dir must exist in the
# image OWNED BY comfy, or docker creates the mountpoint (and initializes a
# fresh named volume) as root and pip/uv/CUDA all fail to write. Placed late
# so editing it never invalidates the expensive layers above.
RUN mkdir -p /home/comfy/.cache/pip /home/comfy/.cache/uv /home/comfy/.cache/nv-compute

# Runtime environment, mirroring the native cmd_run: 4 GB CUDA kernel cache
# (redirected into the cache volume so compiled sm_121 kernels survive
# container recreation), NCCL P2P off, Triton pointed at CUDA 13's ptxas.
ENV SAGE_SRC=/opt/SageAttention \
    SAGE_REF=${SAGE_REF} \
    SAGE_MARKER=/opt/comfyui-env/.sage_ok \
    TORCH_INDEX=${TORCH_INDEX} \
    ORT_WHEEL_URL=${ORT_WHEEL_URL} \
    CUDA_CACHE_PATH=/home/comfy/.cache/nv-compute \
    CUDA_CACHE_MAXSIZE=4294967296 \
    NCCL_P2P_DISABLE=1 \
    TRITON_PTXAS_PATH=/usr/local/cuda/bin/ptxas

EXPOSE 8188
ENTRYPOINT ["/opt/spark/entrypoint.sh"]
