#!/bin/bash
# One-time setup of a fresh network volume. Run inside tmux.
set -e
bash /vol/video_to_data/scripts/runpod/bootstrap.sh
hf download nvidia/video_to_data_challenge --repo-type dataset \
  --include "track_1/*" --local-dir /vol/data
# TODO: add the weights download once the command is confirmed
echo "volume initialized"
