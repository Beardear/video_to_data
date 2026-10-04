#!/bin/bash
set -e
mkdir -p ~/.ssh && chmod 700 ~/.ssh
[ -n "$PUBLIC_KEY" ] && echo "$PUBLIC_KEY" >> ~/.ssh/authorized_keys && chmod 600 ~/.ssh/authorized_keys
ssh-keygen -A
service ssh start || /usr/sbin/sshd

env | grep -E '^(HF_|KAGGLE_|GITHUB_|RUNPOD_|PATH=|LD_LIBRARY_PATH=|CUDA|TORCH_HOME)' \
  | sed 's/=\(.*\)/="\1"/; s/^/export /' > /etc/rp_environment
grep -q rp_environment ~/.bashrc || echo 'source /etc/rp_environment' >> ~/.bashrc

mkdir -p /vol/outputs
nohup python3 -m http.server 8080 --directory /vol/outputs >/tmp/http.log 2>&1 &

sleep infinity
