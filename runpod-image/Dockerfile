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
    SAM2_BUILD_CUDA=0 /opt/venvs/sam2/bin/pip install -e /opt/sam2

RUN pip install -U "huggingface_hub[cli]" hf_transfer kaggle

COPY runpod-image/start.sh /start.sh
RUN sed -i 's/\r$//' /start.sh && chmod +x /start.sh

ENTRYPOINT []
CMD ["/start.sh"]
