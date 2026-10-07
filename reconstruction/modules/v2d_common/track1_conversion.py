"""CPU contracts for conversion-added acceleration, not ground-truth ACC-H."""

from __future__ import annotations

import ast
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(frozen=True)
class Track1Scoring:
    joint_indices: tuple[int, ...]
    frames: tuple[int, ...]

    def __post_init__(self) -> None:
        if (len(self.joint_indices) != 22 or len(set(self.joint_indices)) != 22
                or any(type(i) is not int or not 0 <= i < 127 for i in self.joint_indices)):
            raise ValueError("Expected 22 unique official MHR joint indices")
        if (not self.frames or any(type(i) is not int or i < 0 for i in self.frames)
                or tuple(sorted(set(self.frames))) != self.frames):
            raise ValueError("Scored frames must be nonempty, sorted, unique nonnegative integers")

    def stretches(self) -> list[np.ndarray]:
        frames = np.asarray(self.frames, dtype=np.int64)
        return list(np.split(frames, np.flatnonzero(np.diff(frames) != 1) + 1))

    def centers(self) -> list[int]:
        return [int(f) for run in self.stretches() for f in run[1:-1]]


def track1_body_joint_indices(metrics_path: Path) -> tuple[int, ...]:
    """Read the kit's literal joint contract without importing its code."""
    values = [ast.literal_eval(node.value) for node in ast.parse(metrics_path.read_text()).body
              if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name)
                  and t.id == "MHR_TABLE3_BODY_JOINT_INDICES" for t in node.targets)]
    if len(values) != 1:
        raise ValueError("Missing or ambiguous official body-joint definition")
    indices = tuple(values[0])
    Track1Scoring(indices, (0,))
    return indices


def track1_scoring(kit: Path, episode_index: int, expected_frames: int) -> Track1Scoring:
    import pandas as pd

    sample = pd.read_parquet(kit / "data/track_1_sample_submission.parquet", columns=["row_id"])
    ids = sample["row_id"]
    if ids.empty or ids.isna().any() or ids.duplicated().any():
        raise ValueError("Official sample must contain unique nonempty row IDs")
    keys = ids.str.extract(r"^t1_(\d{6})_(\d{6})_([0-7])_(\d{6})$")
    if keys.isna().any().any():
        raise ValueError("Official sample has invalid Track 1 row IDs")
    keys = keys.astype(int)
    selected = keys.loc[keys[0] == episode_index]
    # Pose, rotation and translation must describe the same scored timeline.
    frame_sets = [set(selected.loc[selected[2] == role, 1]) for role in (0, 3, 4)]
    if not frame_sets[0] or any(s != frame_sets[0] for s in frame_sets[1:]):
        raise ValueError("Official scored pose/object timelines are missing or inconsistent")
    frames = tuple(sorted(frame_sets[0]))
    if frames[-1] >= expected_frames:
        raise ValueError("Official scored frames exceed the reconstructed timeline")
    return Track1Scoring(track1_body_joint_indices(kit / "v2dlb/mhr_metrics.py"), frames)


def track1_acceleration_summary(
    per_frame_mean_cm: Any, scoring: Track1Scoring, reference_cm: float,
) -> dict[str, Any]:
    """Summarize second differences against a diagnostic reference, without a gate."""
    if (type(reference_cm) not in (int, float) or not np.isfinite(reference_cm)
            or reference_cm <= 0):
        raise ValueError("Added-acceleration reference must be finite and positive")
    centers = scoring.centers()
    if not centers:
        raise ValueError("Added acceleration requires a scored stretch of at least three frames")
    values = np.asarray(per_frame_mean_cm, dtype=np.float64)
    if (values.shape != (len(centers),) or not np.isfinite(values).all() or (values < 0).any()):
        raise ValueError("Added acceleration must be finite and nonnegative for each scored triplet")
    stretches, offset = [], 0
    for run in scoring.stretches():
        count = max(0, len(run) - 2)
        stretches.append({"start_frame": int(run[0]), "end_frame": int(run[-1]),
                          "evaluated_triplets": count,
                          "added_acc_h_cm": float(values[offset:offset + count].mean()) if count else None})
        offset += count
    mean = float(values.mean())
    return {"schema": "v2d.track1.added_acceleration.v2", "units": "cm/frame^2",
            "policy": "report_only",
            "definition": "mean_norm_second_difference_of_converted_minus_original_joints",
            "alignment": "none_same_world_frame", "is_kaggle_acc_h": False,
            "joint_indices": list(scoring.joint_indices), "scored_frames": list(scoring.frames),
            "center_frames": centers, "per_frame_mean_cm": values.tolist(), "stretches": stretches,
            "added_acc_h_cm": mean, "reference_cm": float(reference_cm),
            "comparison": "<", "reference_is_competition_rule": False,
            "within_reference": mean < reference_cm}


def track1_added_acceleration(
    original_joints: Any, converted_joints: Any, scoring: Track1Scoring, *, reference_cm: float,
) -> dict[str, Any]:
    """Arrays are [T, 127, 3] in metres, in the identical world/axis convention.

    No similarity alignment, temporal smoothing, FPS scaling, or gap bridging.
    Weight each valid frame triplet equally, rather than each stretch equally.
    """
    original, converted = (np.asarray(a, dtype=np.float64) for a in (original_joints, converted_joints))
    if (original.ndim != 3 or original.shape[1:] != (127, 3) or converted.shape != original.shape
            or not np.isfinite(original).all() or not np.isfinite(converted).all()
            or scoring.frames[-1] >= len(original)):
        raise ValueError("Expected matching finite [T,127,3] joint arrays covering the scored frames")
    residual = converted[:, scoring.joint_indices] - original[:, scoring.joint_indices]
    values = []
    for run in scoring.stretches():
        if len(run) >= 3:
            acceleration = residual[run[2:]] - 2 * residual[run[1:-1]] + residual[run[:-2]]
            values.extend((100 * np.linalg.norm(acceleration, axis=-1).mean(axis=1)).tolist())
    return track1_acceleration_summary(values, scoring, reference_cm)


def track1_validate_acceleration(
    report: dict[str, Any], scoring: Track1Scoring, reference_cm: float,
) -> None:
    """Reject malformed diagnostics; a value above the reference remains valid."""
    if not isinstance(report, dict):
        raise ValueError("Missing or inconsistent added-acceleration report")
    expected = track1_acceleration_summary(report.get("per_frame_mean_cm"), scoring, reference_cm)
    if report != expected:
        raise ValueError("Missing or inconsistent added-acceleration report")


def track1_vertex_spikes(per_frame_error_mm: Any, scoring: Track1Scoring) -> dict[str, Any]:
    """Report reviewer-defined vertex-error spikes; does not gate publication.

    Use original frame indices and up to five neighbors on each side, excluding
    the frame itself. Neighbors outside the scored span still provide context;
    report separately which detected spikes fall inside the scored frames.
    """
    errors = np.asarray(per_frame_error_mm, dtype=np.float64)
    if (errors.ndim != 1 or len(errors) < 2 or scoring.frames[-1] >= len(errors)
            or not np.isfinite(errors).all() or (errors < 0).any()):
        raise ValueError("Spikes require finite nonnegative errors for the full scored timeline and neighbors")
    window, ratio, floor_mm = 5, 3.0, 1.0
    medians = np.asarray([
        np.median(np.concatenate((errors[max(0, t - window):t], errors[t + 1:t + window + 1])))
        for t in range(len(errors))
    ])
    spikes = np.flatnonzero((errors > ratio * medians) & (errors > floor_mm)).tolist()
    scored = set(scoring.frames)
    scored_spikes = [t for t in spikes if t in scored]
    return {"schema": "v2d.track1.vertex_spikes.v1", "units": "mm",
            "policy": "report_only", "threshold_is_competition_rule": False,
            "definition": "error > 3 * neighbor_median AND error > 1 mm",
            "window_radius_frames": window, "exclude_center": True,
            "neighbors": "full_original_timeline_clipped_at_episode_boundaries",
            "ratio": ratio, "floor_mm": floor_mm, "comparison": ">",
            "local_median_mm": medians.tolist(), "scored_frames": list(scoring.frames),
            "spike_frames": spikes, "spike_count": len(spikes),
            "scored_spike_frames": scored_spikes, "scored_spike_count": len(scored_spikes),
            "unscored_spike_frames": [t for t in spikes if t not in scored],
            "no_scored_spikes": not scored_spikes}
