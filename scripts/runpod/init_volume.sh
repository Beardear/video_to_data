#!/bin/bash
# One-time setup of a fresh network volume. Run inside tmux.
set -e
script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
bash "$script_dir/bootstrap.sh"
: "${HF_TOKEN:?Set HF_TOKEN from your RunPod secret before downloading weights}"
hf download nvidia/video_to_data_challenge --repo-type dataset \
  --revision 5f68335f3acc802033d1e80728c1633197521de8 \
  --include "track_1/**" --local-dir /vol/data
python -m v2d.cari4d.lib.download_weights --output_dir /vol/weights/cari4d
/opt/venvs/sam2/bin/python -m v2d.sam2.lib.download_weights --output_dir /vol/weights/sam2
/opt/venvs/sam3d/bin/python -m v2d.sam3d.lib.download_weights --output_dir /vol/weights/sam3d
echo "volume initialized"
