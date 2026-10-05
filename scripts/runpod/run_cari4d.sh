#!/bin/bash
# Use the baked inference library with CPU threading sized for a GPU Pod.
set -e
if ! mountpoint -q /vol; then
  echo "Network volume is not mounted at /vol; refusing container-disk writes." >&2
  exit 1
fi
source /etc/rp_environment

# A container can see hundreds of host CPUs while its quota is much smaller.
# Keep small CPU tensor operations from spawning a host-sized thread pool.
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
exec python -m v2d.cari4d.lib.run_inference "$@"
