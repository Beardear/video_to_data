"""Decode submitted parameters with the official model; measure conversion jitter."""

from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path

import numpy as np
import torch

from v2d.common.artifacts import atomic_json
from v2d.common.track1_conversion import track1_added_acceleration, track1_scoring


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("original-joints", "submission", "submission-kit", "model", "output", "device"):
        parser.add_argument(f"--{name}", required=True)
    parser.add_argument("--episode-index", type=int, required=True)
    parser.add_argument("--expected-frames", type=int, required=True)
    parser.add_argument("--batch-size", type=int, required=True)
    parser.add_argument("--precision", choices=("float32", "float64"), required=True)
    parser.add_argument("--reference-cm", type=float, required=True)
    args = parser.parse_args()
    if args.batch_size <= 0:
        raise ValueError("batch-size must be positive")
    kit, output = Path(args.submission_kit), Path(args.output)
    scoring = track1_scoring(kit, args.episode_index, args.expected_frames)
    spec = importlib.util.spec_from_file_location("official_track1_fitter", kit / "tools/track1/mesh_to_mhr_params.py")
    official = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(official)
    model = official.MHR(args.model, args.device, args.batch_size, args.precision)
    # Decode the exact float32 arrays to be published, not the pre-cast fitter output.
    with np.load(args.submission, allow_pickle=False) as data:
        pose = torch.as_tensor(data["pose"], dtype=model.dtype, device=model.device)
        identity = torch.as_tensor(np.concatenate([data["scales"], data["shape"]])[None],
                                   dtype=model.dtype, device=model.device)
    converted = np.empty((args.expected_frames, 127, 3), dtype=np.float64)
    if tuple(pose.shape) != (args.expected_frames, 136):
        raise ValueError("Submission poses must cover the full original timeline")
    for start in range(0, args.expected_frames, args.batch_size):
        stop = min(start + args.batch_size, args.expected_frames)
        _, joints = model.run(pose[start:stop], identity)
        converted[start:stop] = joints.cpu().numpy()
    np.save(output.with_suffix(".joints.npy"), converted)
    report = track1_added_acceleration(np.load(args.original_joints, allow_pickle=False), converted,
                                      scoring, reference_cm=args.reference_cm)
    # Values above the reference are diagnostics, not publication failures.
    atomic_json(output, report)


if __name__ == "__main__":
    main()
