#!/bin/bash
# Run at the start of every pod session: bash /vol/video_to_data/scripts/runpod/bootstrap.sh
pip install -q -U "brotli>=1.2" || true
mkdir -p /vol/{data,weights,outputs/logs,submission,cache,secrets}

[ -n "$GIT_USER_NAME" ]  && git config --global user.name  "$GIT_USER_NAME"
[ -n "$GIT_USER_EMAIL" ] && git config --global user.email "$GIT_USER_EMAIL"
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
