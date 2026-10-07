# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Export a refined CARI4D episode via the official Track 1 MHR mesh fitter."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile


SOURCE_ROOT = Path(__file__).resolve().parent / "cari4d"
SAM3D_SOURCE_ROOT = Path("/workspace/v2d_sam3d_body/lib")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _python_source_digest(root: Path) -> str:
    """Include uncommitted edits when identifying the decoder/adapter source."""
    files = {str(path.relative_to(root)): _sha256(path) for path in sorted(root.rglob("*.py"))}
    if not files:
        raise FileNotFoundError(f"Required Python source tree is missing or empty: {root}")
    return hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest()


def export_track1(
    result_dir: str, submission_kit: str, weights_path: str, output_dir: str, *,
    episode_index: int, expected_frames: int, business_commit: str,
    image_build_commit: str, image_digest: str, device: str = "cuda",
    decode_batch_size: int = 16, fit_model_batch_size: int = 128,
    fit_precision: str = "float64", max_vertex_error_mm: float = 1.0,
    conversion_error_policy: str = "reject",
    max_added_acc_h_cm: float = 0.02,
) -> Path:
    """Publish a new episode directory only after conversion passes its checks.

    The directory is directly consumable by the official packer for a matching
    episode roster. Existing destinations are refused: exports are never reused
    across source/config changes. This does not submit anything to Kaggle.
    """
    import numpy as np
    from v2d.common.track1_conversion import track1_scoring, track1_require_acceleration

    if str(SOURCE_ROOT) not in sys.path:
        sys.path.insert(0, str(SOURCE_ROOT))
    from lib_mhr.contact import load_object_mesh
    from lib_mhr.track1 import Track1ObjectMotion, track1_submission_arrays

    if episode_index < 0 or min(expected_frames, decode_batch_size, fit_model_batch_size) <= 0:
        raise ValueError("Episode index must be nonnegative and counts must be positive")
    if fit_precision not in ("float32", "float64"):
        raise ValueError("fit_precision must be float32 or float64")
    if not np.isfinite(max_vertex_error_mm) or max_vertex_error_mm <= 0:
        raise ValueError("max_vertex_error_mm must be finite and positive")
    if conversion_error_policy not in {"reject", "report"}:
        raise ValueError("conversion_error_policy must be reject or report")
    if (type(max_added_acc_h_cm) not in (int, float) or not np.isfinite(max_added_acc_h_cm)
            or max_added_acc_h_cm <= 0):
        raise ValueError("max_added_acc_h_cm must be finite and positive")
    for name, value in (("business_commit", business_commit), ("image_build_commit", image_build_commit)):
        if not re.fullmatch(r"[0-9a-f]{40}", value):
            raise ValueError(f"{name} must be a full 40-character commit SHA")
    if not re.fullmatch(r"(?:[^\s]+@)?sha256:[0-9a-f]{64}", image_digest):
        raise ValueError("image_digest must identify the immutable image sha256 digest")
    result, kit, weights, output = map(lambda p: Path(p).resolve(),
                                       (result_dir, submission_kit, weights_path, output_dir))
    if output.exists():
        raise FileExistsError(f"Use a new export directory: {output}")
    sequence = f"episode_{episode_index:06d}"
    pipeline_report = result / "pipeline_report.json"
    bundle = result / "inference/refined.pth"
    mesh = result / "export" / sequence / "object_mesh/output_aligned.glb"
    converter = kit / "tools/track1/mesh_to_mhr_params.py"
    model = weights / "sam3d_body/checkpoints/sam-3d-body-dinov3/assets/mhr_model.pt"
    decoder = SOURCE_ROOT / "tools/export_track1_vertices.py"
    acceleration_checker = SOURCE_ROOT / "tools/check_track1_acceleration.py"
    import v2d.common.track1_conversion as conversion_contract
    inputs = {"refined_bundle": bundle, "aligned_object_mesh": mesh,
              "pipeline_report": pipeline_report, "mhr_model": model,
              "official_converter": converter, "decoder_script": decoder,
              "adapter": Path(__file__), "export_contract": SOURCE_ROOT / "lib_mhr/track1.py",
              "acceleration_checker": acceleration_checker,
              "acceleration_contract": Path(conversion_contract.__file__),
              "official_metrics": kit / "v2dlb/mhr_metrics.py",
              "official_sample": kit / "data/track_1_sample_submission.parquet"}
    for path in inputs.values():
        if not path.is_file() or path.stat().st_size == 0:
            raise FileNotFoundError(f"Required export input is missing or empty: {path}")
    pipeline = json.loads(pipeline_report.read_text())
    if (pipeline.get("schema") != "v2d.cari4d.wild_inference.v1"
            or pipeline.get("sequence") != sequence or pipeline.get("verdict") != "PASS"):
        raise ValueError("Pipeline report must identify this successfully reconstructed episode")
    inference_settings = pipeline.get("defaults", {})
    if pipeline.get("run_identity") is not None:
        identity_path = result / "run_identity.json"
        identity = json.loads(identity_path.read_text())
        if (identity.get("schema") != "v2d.cari4d.run_identity.v1"
                or identity.get("sha256") != pipeline["run_identity"].get("sha256")):
            raise ValueError("Inference identity does not match its pipeline report")
        inference_settings = identity["settings"]
        inputs["inference_identity"] = identity_path
    object_mesh = load_object_mesh(mesh)
    if (len(object_mesh.vertices) == 0 or len(object_mesh.faces) == 0
            or not np.isfinite(object_mesh.vertices).all() or not np.isfinite(object_mesh.area)
            or object_mesh.area <= 0):
        raise ValueError("Aligned mesh has no finite, nondegenerate surface")
    input_hashes = {name: _sha256(path) for name, path in inputs.items()}
    scoring = track1_scoring(kit, episode_index, expected_frames)
    if not scoring.centers():
        raise ValueError("Added acceleration requires a continuous scored three-frame stretch")
    source_roots = {"cari4d_lib": Path(__file__).resolve().parent,
                    "sam3d_body_lib": SAM3D_SOURCE_ROOT}
    source_hashes = {name: _python_source_digest(root) for name, root in source_roots.items()}
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join((str(SOURCE_ROOT), str(SAM3D_SOURCE_ROOT),
                                       env.get("PYTHONPATH", "")))
    output.parent.mkdir(parents=True, exist_ok=True)
    work = Path(tempfile.mkdtemp(prefix=f".{output.name}-", dir=output.parent))
    try:
        # Separate processes release all decoder VRAM before the fitter starts.
        decode_command = [sys.executable, str(decoder), "--bundle", str(bundle),
                          "--weights", str(weights), "--output", str(work),
                          "--expected-frames", str(expected_frames),
                          "--batch-size", str(decode_batch_size), "--device", device]
        subprocess.run(decode_command, cwd=SOURCE_ROOT, env=env, check=True)
        fit_command = [sys.executable, str(converter), "--model", str(model),
                       "--input", str(work / "human_vertices.npy"),
                       "--output", str(work / "human_fit.npz"), "--device", device,
                       "--precision", fit_precision, "--model-batch", str(fit_model_batch_size)]
        subprocess.run(fit_command, cwd=kit, env=env, check=True)
        with np.load(work / "human_fit.npz", allow_pickle=False) as fitted:
            with np.load(work / "object_motion.npz", allow_pickle=False) as motion:
                arrays = track1_submission_arrays(fitted, Track1ObjectMotion(
                    motion["rotation"], motion["translation"]), max_vertex_error_mm=max_vertex_error_mm,
                    conversion_error_policy=conversion_error_policy)
            fit_report = json.loads(str(fitted["report"].item()))
            errors = fitted["per_frame_vertex_error_mm"].copy()
        destination = work / "published"
        destination.mkdir()
        npz = destination / f"{sequence}.npz"
        np.savez(npz, **arrays)
        acceleration_path = work / "added_acceleration.json"
        subprocess.run([
            sys.executable, str(acceleration_checker), "--original-joints", str(work / "human_joints.npy"),
            "--submission", str(npz), "--submission-kit", str(kit), "--model", str(model),
            "--output", str(acceleration_path), "--device", device, "--episode-index", str(episode_index),
            "--expected-frames", str(expected_frames), "--batch-size", str(decode_batch_size),
            "--precision", fit_precision, "--threshold-cm", str(max_added_acc_h_cm),
        ], cwd=SOURCE_ROOT, env=env, check=True)
        acceleration = json.loads(acceleration_path.read_text())
        track1_require_acceleration(acceleration, scoring, max_added_acc_h_cm)
        mesh_output = destination / f"{sequence}_object.glb"
        shutil.copyfile(mesh, mesh_output)
        # Reject concurrently changed inputs; do not publish mixed-version data.
        if (any(_sha256(path) != input_hashes[name] for name, path in inputs.items())
                or any(_python_source_digest(root) != source_hashes[name]
                       for name, root in source_roots.items())):
            raise ValueError("An export input changed during conversion; rerun in a new directory")
        report = {
            "schema": "v2d.cari4d.track1_export.v1", "sequence": sequence,
            "frames": expected_frames, "business_commit": business_commit,
            "inference_run_identity": pipeline.get("run_identity"),
            "inference_settings": inference_settings,
            "image_build_commit": image_build_commit, "image_digest": image_digest,
            "input_sha256": input_hashes,
            "export_source_sha256": source_hashes,
            "decoder_identity": json.loads((work / "decoder.json").read_text()),
            "settings": {"device": device, "decode_batch_size": decode_batch_size,
                         "fit_model_batch_size": fit_model_batch_size, "fit_precision": fit_precision,
                         "max_vertex_error_mm": max_vertex_error_mm,
                         "conversion_error_policy": conversion_error_policy,
                         "max_added_acc_h_cm": max_added_acc_h_cm},
            "acceptance": {"structural_checks": "PASS", "accepted_for_packing": True,
                           "conversion_error_policy": conversion_error_policy,
                           "within_reference_tolerance": bool(np.all(errors <= max_vertex_error_mm)),
                           "frames_above_reference_tolerance": np.flatnonzero(errors > max_vertex_error_mm).tolist(),
                           "reference_tolerance_mm": max_vertex_error_mm,
                           "reference_tolerance_is_competition_rule": False,
                           "added_acceleration_check": "PASS"},
            "conversion": {"method": "official_mesh_to_mhr_params", "report": fit_report,
                           "per_frame_mean_vertex_error_mm": errors.tolist(),
                           "mean_vertex_error_mm": float(errors.mean()),
                           "worst_frame_mean_vertex_error_mm": float(errors.max()),
                           "added_acceleration": acceleration,
                           "identity_policy": "one fitted shape and scale vector per episode"},
            "coordinates": "CARI4D wild camera/world, metres; no extra flip or scaling",
            "object_scale": 1.0, "kaggle_scored": False,
            "output_sha256": {npz.name: _sha256(npz), mesh_output.name: _sha256(mesh_output)},
        }
        (destination / f"{sequence}_export.json").write_text(
            json.dumps(report, indent=2, allow_nan=False) + "\n")
        if output.exists():
            raise FileExistsError(output)
        destination.rename(output)
    except Exception:
        print(f"Track 1 export failed; unpublished diagnostics retained at {work}", file=sys.stderr)
        raise
    else:
        shutil.rmtree(work)
    return output


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("result_dir", "submission_kit", "weights_path", "output_dir",
                 "business_commit", "image_build_commit", "image_digest"):
        parser.add_argument(f"--{name}", required=True)
    parser.add_argument("--episode_index", type=int, required=True)
    parser.add_argument("--expected_frames", type=int, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--decode_batch_size", type=int, default=16)
    parser.add_argument("--fit_model_batch_size", type=int, default=128)
    parser.add_argument("--fit_precision", choices=("float32", "float64"), default="float64")
    parser.add_argument("--max_vertex_error_mm", type=float, default=1.0,
                        help="Reference tolerance for per-frame mean conversion displacement, in mm")
    parser.add_argument("--conversion_error_policy", choices=("reject", "report"), default="reject",
                        help="Reject above the reference tolerance, or publish with the measured errors recorded")
    parser.add_argument("--max_added_acc_h_cm", type=float, default=0.02,
                        help="Require conversion-added joint acceleration below this value in cm/frame^2")
    return parser


if __name__ == "__main__":
    print(export_track1(**vars(_parser().parse_args())))
