"""Track 1 export contracts for a complete CARI4D wild-inference episode.

Canonical MHR parameters are obtained by the submission kit's mesh fitter;
internal continuous poses and per-frame identities are not submission arrays.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np

from .object_pose_frame import OBJECT_POSE_FRAME_REVISION, resolve_object_pose_frame
from .schema import MHR_PARAM_DIMS


@dataclass(frozen=True)
class Track1ObjectMotion:
    rotation: np.ndarray
    translation: np.ndarray


def _array(value: Any, shape: tuple[int, ...], name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float32)
    if array.shape != shape or not np.isfinite(array).all():
        raise ValueError(f"{name} must be finite with shape {shape}, got {array.shape}")
    return array


def track1_object_motion(poses: Any, frames: int) -> Track1ObjectMotion:
    poses = _array(poses, (frames, 4, 4), "object poses")
    if not np.allclose(poses[:, 3], [0, 0, 0, 1], rtol=0, atol=1e-6):
        raise ValueError("Object poses must be homogeneous rigid transforms")
    rotation = poses[:, :3, :3]
    if not np.allclose(rotation @ rotation.transpose(0, 2, 1), np.eye(3), rtol=0, atol=1e-4):
        raise ValueError("Object rotations must be orthonormal")
    if not np.allclose(np.linalg.det(rotation), 1, rtol=0, atol=1e-4):
        raise ValueError("Object rotations must be proper (determinant +1)")
    return Track1ObjectMotion(rotation.copy(), poses[:, :3, 3].copy())


def track1_validate_bundle(bundle: Mapping[str, Any], expected_frames: int) -> Track1ObjectMotion:
    if expected_frames <= 0:
        raise ValueError("expected_frames must be positive")
    if bundle.get("schema") != "cari4d.mhr_wild_inference.v1":
        raise ValueError("Expected a CARI4D wild-inference bundle")
    names = [f"{i:06d}" for i in range(expected_frames)]
    if bundle.get("frames") != names:
        raise ValueError("Bundle frames must cover every video frame in order, starting at 0")
    frame_meta = bundle.get("frame_meta")
    if not isinstance(frame_meta, list) or len(frame_meta) != expected_frames:
        raise ValueError("Missing complete frame_meta")
    for index, row in enumerate(frame_meta):
        if row.get("src_frame") != index or row.get("frame") != names[index] or row.get("kid") != 0:
            raise ValueError(f"Frame metadata does not match video frame {index}")
    if bundle.get("postopt", {}).get("frame_indices") != list(range(expected_frames)):
        raise ValueError("Expected a refined bundle covering the entire episode")
    metadata = bundle.get("metadata", {})
    if metadata.get("object_pose_frame_revision") != OBJECT_POSE_FRAME_REVISION:
        raise ValueError("Missing current object-pose-frame metadata")
    frame = resolve_object_pose_frame(metadata)
    # This adapter accepts the wild pipeline's already aligned mesh. It must not
    # silently pair a raw SAM3D mesh with poses in the training/template frame.
    if not np.allclose(frame.mesh_to_training, np.eye(4), rtol=0, atol=1e-6):
        raise ValueError("Export requires the wild pipeline's aligned object mesh")
    pr = bundle["pr"]
    for key, dimension in MHR_PARAM_DIMS.items():
        _array(pr.get(key), (expected_frames, dimension), key)
    return track1_object_motion(pr["pose_abs"], expected_frames)


def track1_submission_arrays(
    fitted: Mapping[str, Any], motion: Track1ObjectMotion, *, max_vertex_error_mm: float,
) -> dict[str, np.ndarray]:
    """Validate the official fitter's outputs before publishing submission files.

    The threshold bounds the worst frame's mean vertex displacement introduced
    by conversion, not reconstruction accuracy against held-out ground truth.
    """
    if not np.isfinite(max_vertex_error_mm) or max_vertex_error_mm <= 0:
        raise ValueError("max_vertex_error_mm must be finite and positive")
    count = len(motion.rotation)
    valid = np.asarray(fitted["valid_input"])
    if valid.shape != (count,) or valid.dtype != np.bool_ or not valid.all():
        raise ValueError("Fitter interpolated or omitted invalid input frames")
    errors = _array(fitted["per_frame_vertex_error_mm"], (count,), "fit errors")
    if (errors < 0).any() or errors.max() > max_vertex_error_mm:
        raise ValueError(f"Conversion worst-frame mean error {errors.max():.6f} mm "
                         f"exceeds the {max_vertex_error_mm:g} mm limit; inspect the fit")
    return {
        "pose": _array(fitted["pose"], (count, 136), "pose"),
        "scales": _array(fitted["scales"], (68,), "scales"),
        "shape": _array(fitted["shape"], (45,), "shape"),
        "object_rotation": _array(motion.rotation, (count, 3, 3), "object_rotation"),
        "object_translation": _array(motion.translation, (count, 3), "object_translation"),
        # The mesh already carries its physical scale. Applying it again would
        # scale the object twice. P_scene = R @ P_aligned_mesh + t, in metres.
        "object_scale": np.asarray(1.0, dtype=np.float32),
    }
