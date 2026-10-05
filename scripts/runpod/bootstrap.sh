#!/bin/bash
# Run at the start of every pod session from the docker-image worktree.
set -e
if ! mountpoint -q /vol; then
  echo "Network volume is not mounted at /vol; refusing container-disk writes." >&2
  exit 1
fi
pip install -q -U "brotli>=1.2" || true
mkdir -p /vol/{data,weights,outputs/logs,submission,cache,secrets}

if [ -n "$GIT_USER_NAME" ]; then git config --global user.name "$GIT_USER_NAME"; fi
if [ -n "$GIT_USER_EMAIL" ]; then git config --global user.email "$GIT_USER_EMAIL"; fi
git config --global --add safe.directory /vol/video_to_data

if [ -f /vol/secrets/kaggle.json ]; then
  mkdir -p ~/.kaggle && ln -sf /vol/secrets/kaggle.json ~/.kaggle/kaggle.json
  chmod 600 /vol/secrets/kaggle.json
fi

grep -q "cd /vol/video_to_data" ~/.bashrc || cat >> ~/.bashrc <<'RC'
cd /vol/video_to_data 2>/dev/null
alias sam2py=/opt/venvs/sam2/bin/python
RC
echo "bootstrap done"
