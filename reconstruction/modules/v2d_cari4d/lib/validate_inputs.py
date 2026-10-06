# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Decode and inspect prepared CARI4D inputs without loading an ML model."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
import sys


SOURCE_ROOT = Path(__file__).resolve().parent / "cari4d"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))
from lib_mhr.artifacts import artifact_record, atomic_json


@dataclass(frozen=True)
class InputIssue:
    severity: str
    code: str
    message: str


def validate_inputs(
    video_path: str, mask_h5_path: str, object_mesh_path: str, report_path: str, *,
    expected_frames: int, expected_fps: float, source_video_path: str | None = None,
    mesh_scale_report_path: str | None = None, require_scale_provenance: bool = False,
) -> Path:
    """Write PASS/FAIL evidence; a caller must inspect verdict before inference.

    PASS proves timeline, mask encoding, mesh structure, and (when required)
    recorded scale provenance. It does not prove segmentation or metric accuracy.
    Empty object masks after frame zero are recorded as occlusion warnings.
    """
    import av
    import h5py
    import numpy as np
    from lib_mhr.contact import load_object_mesh

    if type(expected_frames) is not int or expected_frames <= 0:
        raise ValueError("expected_frames must be a positive integer")
    if isinstance(expected_fps, bool) or not math.isfinite(expected_fps) or expected_fps <= 0:
        raise ValueError("expected_fps must be finite and positive")
    video, masks, mesh_path, report_path = map(lambda p: Path(p).resolve(),
                                             (video_path, mask_h5_path, object_mesh_path, report_path))
    suffix = ".0.color.mp4"
    if not video.name.endswith(suffix):
        raise ValueError(f"Prepared video must end with {suffix}")
    sequence = video.name[:-len(suffix)]
    paths = {"video": video, "masks": masks, "mesh": mesh_path}
    if source_video_path is not None:
        paths["source_video"] = Path(source_video_path).resolve()
    if mesh_scale_report_path is not None:
        paths["mesh_scale_report"] = Path(mesh_scale_report_path).resolve()
    if report_path in paths.values():
        raise ValueError("The validation report must not overwrite an input")
    identities = {role: artifact_record(path) for role, path in paths.items()}
    issues: list[InputIssue] = []
    def fail(code: str, message: str) -> None:
        issues.append(InputIssue("error", code, message))

    def warn(code: str, message: str) -> None:
        issues.append(InputIssue("warning", code, message))
    if source_video_path is not None and identities["source_video"]["sha256"] != identities["video"]["sha256"]:
        fail("source_video_mismatch", "Prepared video is not a byte-identical copy of the dataset video")
    empty: dict[str, list[int]] = {"human": [], "object": []}
    coverage: dict[str, list[float]] = {"human": [], "object": []}
    observed = 0
    width = height = 0
    fps = None
    previous_time = None
    expected_keys = {f"{i:06d}-k0.{kind}" for i in range(expected_frames)
                     for kind in ("person_mask.png", "obj_rend_mask.png")}
    with av.open(str(video)) as container, h5py.File(masks, "r") as handle:
        if len(container.streams.video) != 1:
            raise ValueError("Expected exactly one video stream")
        stream = container.streams.video[0]
        width, height = stream.codec_context.width, stream.codec_context.height
        fps = float(stream.average_rate) if stream.average_rate else None
        if fps is None or not math.isclose(fps, expected_fps, rel_tol=0, abs_tol=1e-3):
            fail("video_fps", f"Expected {expected_fps} fps, found {fps}")
        if set(handle.keys()) != {sequence} or not isinstance(handle.get(sequence), h5py.Group):
            fail("mask_sequence", f"Expected only the mask group {sequence}")
        group = handle.get(sequence)
        if isinstance(group, h5py.Group) and set(group.keys()) != expected_keys:
            fail("mask_timeline", "Mask keys must cover both roles for exactly the expected video frames")
        # Attributes are emitted by the current packer; older compatible masks
        # may omit them. Present metadata must agree with decoded content.
        expected_attrs = {"schema": "v2d.cari4d.wild_masks.v1", "sequence": sequence,
                          "frame_count": expected_frames, "width": width, "height": height}
        for key, expected in expected_attrs.items():
            if key in handle.attrs and handle.attrs[key] != expected:
                fail("mask_metadata", f"Mask attribute {key} disagrees with input: {handle.attrs[key]!r}")
        for index, frame in enumerate(container.decode(stream)):
            observed += 1
            rgb = frame.to_ndarray(format="rgb24")
            if rgb.shape != (height, width, 3):
                fail("video_dimensions", f"Video dimensions changed at frame {index}")
            if frame.time is None or (previous_time is not None and frame.time <= previous_time):
                fail("video_timestamps", f"Missing or non-increasing timestamp at frame {index}")
            elif previous_time is not None and not math.isclose(
                frame.time - previous_time, 1 / expected_fps, rel_tol=0, abs_tol=1e-4
            ):
                fail("video_timestamps", f"Frame cadence differs from metadata at frame {index}")
            previous_time = frame.time
            for role, ending in (("human", "person_mask.png"), ("object", "obj_rend_mask.png")):
                key = f"{sequence}/{index:06d}-k0.{ending}"
                if key not in handle:
                    continue  # Missing/extra keys already produce a timeline error.
                dataset = handle[key]
                if not isinstance(dataset, h5py.Dataset) or dataset.shape != (height, width):
                    fail("mask_dimensions", f"{role} mask has incorrect dimensions at frame {index}")
                    continue
                mask = np.asarray(dataset[()])
                if (mask.dtype.kind not in "biu" or not np.isin(mask, [0, 1, 255]).all()
                        or (np.any(mask == 1) and np.any(mask == 255))):
                    fail("mask_encoding", f"{role} mask is not a binary mask at frame {index}")
                    continue
                fraction = float(np.count_nonzero(mask) / mask.size)
                coverage[role].append(fraction)
                if fraction == 0:
                    empty[role].append(index)
                elif fraction == 1:
                    fail("full_frame_mask", f"{role} mask covers the entire image at frame {index}")
    if observed != expected_frames:
        fail("video_frame_count", f"Expected {expected_frames} frames, decoded {observed}")
    if empty["human"]:
        fail("empty_human_masks", f"Human is absent in {len(empty['human'])} mask frames")
    if 0 in empty["object"] or len(empty["object"]) == expected_frames:
        fail("object_initialization", "FoundationPose needs a nonempty object mask at frame zero")
    elif empty["object"]:
        warn("object_occlusion", f"Object mask is empty in {len(empty['object'])} frames; inspect tracking")
    mesh = load_object_mesh(mesh_path)
    vertices, faces = np.asarray(mesh.vertices), np.asarray(mesh.faces)
    mesh_valid = (vertices.ndim == 2 and vertices.shape[1:] == (3,) and len(vertices) > 0
                  and np.isfinite(vertices).all() and faces.ndim == 2 and faces.shape[1:] == (3,)
                  and len(faces) > 0 and faces.min() >= 0 and faces.max() < len(vertices))
    mesh_info = {"vertices": len(vertices), "faces": len(faces)}
    if not mesh_valid:
        fail("mesh_geometry", "Mesh has invalid vertices or face indices")
    elif not np.isfinite(mesh.area) or mesh.area <= 0:
        fail("mesh_surface", "Mesh has no finite nondegenerate surface")
    else:
        mesh_info.update(extents_m=mesh.extents.tolist(), area_m2=float(mesh.area),
                         watertight=bool(mesh.is_watertight))
        if not mesh.is_watertight:
            warn("mesh_open_surface", "Mesh is not watertight; inspect object reconstruction")
    scale_verified = False
    if mesh_scale_report_path is not None:
        scale = json.loads(paths["mesh_scale_report"].read_text())
        if not isinstance(scale, dict):
            raise ValueError("Scale record must be a JSON object")
        factors = scale.get("applied_scale")
        factors_valid = (isinstance(factors, list) and len(factors) == 3
                         and all(type(v) in (int, float) and math.isfinite(v) and v > 0 for v in factors))
        reference_frame = scale.get("reference_frame")
        scale_verified = (scale.get("schema") == "v2d.track1.mesh_scale.v1"
                          and scale.get("units") == "metres"
                          and scale.get("mesh_sha256") == identities["mesh"]["sha256"]
                          and scale.get("source_video_sha256") == identities["video"]["sha256"]
                          and type(reference_frame) is int and 0 <= reference_frame < expected_frames
                          and factors_valid
                          and scale.get("method") in {"depth_alignment", "sam3d_pointmap", "measured"})
        if not scale_verified:
            fail("mesh_scale_provenance", "Scale record must bind this mesh and video to a valid frame, positive scale, units, and method")
    elif require_scale_provenance:
        fail("mesh_scale_provenance", "A scale provenance record is required")
    else:
        warn("mesh_scale_unverified", "Mesh units and physical scale have not been established by a record")
    if identities != {role: artifact_record(path) for role, path in paths.items()}:
        fail("input_changed", "An input changed during validation; rerun before inference")
    report = {"schema": "v2d.cari4d.input_validation.v1", "sequence": sequence,
              "verdict": "FAIL" if any(i.severity == "error" for i in issues) else "PASS",
              "inputs": identities, "video": {"expected_frames": expected_frames, "decoded_frames": observed,
              "width": width, "height": height, "fps": fps}, "empty_mask_frames": empty,
              "mask_coverage": {role: {"min": min(values), "max": max(values)} if values else None
                                for role, values in coverage.items()}, "mesh": mesh_info,
              "scale_provenance_verified": scale_verified, "issues": [asdict(i) for i in issues],
              "not_checked": ["segmentation_semantic_accuracy", "physical_scale_ground_truth", "gpu_runtime"]}
    atomic_json(report_path, report)
    return report_path


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("video_path", "mask_h5_path", "object_mesh_path", "report_path"):
        parser.add_argument(f"--{name}", required=True)
    parser.add_argument("--expected_frames", type=int, required=True)
    parser.add_argument("--expected_fps", type=float, required=True)
    parser.add_argument("--source_video_path")
    parser.add_argument("--mesh_scale_report_path")
    parser.add_argument("--require_scale_provenance", action="store_true")
    return parser


if __name__ == "__main__":
    report = validate_inputs(**vars(_parser().parse_args()))
    print(report)
    raise SystemExit(0 if json.loads(report.read_text())["verdict"] == "PASS" else 2)
