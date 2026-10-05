ARG BASE_IMAGE
FROM ${BASE_IMAGE}
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
      openssh-server git git-lfs tmux htop rsync curl util-linux \
    && rm -rf /var/lib/apt/lists/* && mkdir -p /var/run/sshd \
    && rm -f /etc/ssh/ssh_host_*

# SAM2 in its own venv (CUDA 12.4 to match the base image)
RUN python3 -m venv /opt/venvs/sam2 && \
    /opt/venvs/sam2/bin/pip install --upgrade pip && \
    /opt/venvs/sam2/bin/pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124 && \
    git clone https://github.com/facebookresearch/sam2.git /opt/sam2 && \
    git -C /opt/sam2 checkout 2b90b9f5ceec907a1c18123530e92e794ad901a4 && \
    SAM2_BUILD_CUDA=0 /opt/venvs/sam2/bin/pip install -e /opt/sam2

COPY reconstruction/modules/v2d_sam2/lib /workspace/v2d_sam2/lib
RUN /opt/venvs/sam2/bin/pip install \
      -e "/workspace/v2d_common[io]" -e /workspace/v2d_mv \
      flask decord huggingface_hub && \
    /opt/venvs/sam2/bin/pip install --no-deps -e /workspace/v2d_sam2/lib && \
    /opt/venvs/sam2/bin/python -m v2d.sam2.lib.video_to_masks --help

# transformers 5.3 requires huggingface_hub <2; CUDA 12 hosts need a compatible
# Warp build rather than the CUDA 13 wheels in recent unpinned releases.
RUN pip install "huggingface_hub==1.33.0" "warp-lang==1.9.1" hf_transfer kaggle

# SAM3D Objects uses MoGe v1 / utils3d APIs that differ from CARI4D's MoGe v2.
# Reuse the base CUDA libraries, but install these Python overrides only here.
COPY runpod-image/sam3d-requirements.txt /opt/sam3d-requirements.txt
RUN python3 -m venv --system-site-packages /opt/venvs/sam3d && \
    /opt/venvs/sam3d/bin/pip install -r /opt/sam3d-requirements.txt && \
    /opt/venvs/sam3d/bin/pip install --no-deps \
      "git+https://github.com/microsoft/MoGe.git@a8c37341bc0325ca99b9d57981cc3bb2bd3e255b" \
      "git+https://github.com/EasternJournalist/utils3d.git@3913c65d81e05e47b9f367250cf8c0f7462a0900" \
      "git+https://github.com/facebookresearch/sam-3d-objects.git@f91db411c50efee93d8db7aeb323885650f6f722" && \
    MAX_JOBS=4 /opt/venvs/sam3d/bin/pip install flash-attn==2.7.4.post1 --no-build-isolation

COPY reconstruction/modules/v2d_sam3d/lib /workspace/v2d_sam3d/lib
ENV LIDRA_SKIP_INIT=1
ENV HYDRA_FULL_ERROR=1
RUN /opt/venvs/sam3d/bin/pip install --no-deps -e /workspace/v2d_sam3d/lib && \
    /opt/venvs/sam3d/bin/python -m v2d.sam3d.lib.image_to_mesh --help

COPY runpod-image/start.sh /start.sh
RUN sed -i 's/\r$//' /start.sh && chmod +x /start.sh

ENTRYPOINT []
CMD ["/start.sh"]
