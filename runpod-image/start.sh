#!/bin/bash
set -e
if ! mountpoint -q /vol; then
  echo "Network volume is not mounted at /vol; refusing container-disk writes." >&2
  exit 1
fi
# SSH keys from template env vars (PUBLIC_KEY, PUBLIC_KEY_2, PUBLIC_KEY_3)
mkdir -p /root/.ssh && chmod 700 /root/.ssh
printf "%s\n%s\n%s\n" "$PUBLIC_KEY" "$PUBLIC_KEY_2" "$PUBLIC_KEY_3" | grep -v '^$' >> /root/.ssh/authorized_keys || true
chmod 600 /root/.ssh/authorized_keys
echo "DIAG keylen=${#PUBLIC_KEY} key2len=${#PUBLIC_KEY_2} key3len=${#PUBLIC_KEY_3}"
ssh-keygen -A
service ssh start || /usr/sbin/sshd

# Make pod env vars (HF_TOKEN, etc.) visible in SSH sessions
(
  umask 077
  for name in $(compgen -e); do
    case "$name" in
      HF_*|KAGGLE_*|GITHUB_*|RUNPOD_*|GIT_USER_*|PATH|LD_LIBRARY_PATH|CUDA*|TORCH_HOME|PIP_CONSTRAINT|PYTHONPATH|PYOPENGL_PLATFORM|OPENCV_IO_ENABLE_OPENEXR|LIDRA_SKIP_INIT|HYDRA_FULL_ERROR)
        printf 'export %s=%q\n' "$name" "${!name}" ;;
    esac
  done > /etc/rp_environment
)
chmod 600 /etc/rp_environment
grep -q rp_environment /root/.bashrc || echo 'source /etc/rp_environment' >> /root/.bashrc

# Port 8080: simple browser for outputs on the network volume
mkdir -p /vol/outputs
nohup python3 -m http.server 8080 --directory /vol/outputs >/tmp/http.log 2>&1 &

sleep infinity
