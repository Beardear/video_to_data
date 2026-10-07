# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Collect accepted episode exports and invoke the unchanged official packer.

Default scope is the complete official roster. An explicit episode subset is a
format check only. This command never uploads or submits anything to Kaggle.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
from typing import Any

from v2d.common.artifacts import artifact_record, atomic_json
from v2d.common.track1_conversion import Track1Scoring, track1_body_joint_indices, track1_require_acceleration
from v2d.pipelines.track1_preflight import Track1Episode, track1_episodes, track1_metadata_identity


@dataclass(frozen=True)
class AcceptedExport:
    episode_index: int
    npz: Path
    mesh: Path
    report: Path
    provenance: dict[str, Any]
    identities: dict[str, dict[str, Any]]


def _accepted_export(root: Path, episode: Track1Episode, commit: str, scoring: Track1Scoring) -> AcceptedExport:
    import numpy as np

    sequence = f"episode_{episode.episode_index:06d}"
    directory = root / sequence
    paths = {"npz": directory / f"{sequence}.npz", "mesh": directory / f"{sequence}_object.glb",
             "report": directory / f"{sequence}_export.json"}
    identities = {name: artifact_record(path) for name, path in paths.items()}
    report = json.loads(paths["report"].read_text())
    if (report.get("schema") != "v2d.cari4d.track1_export.v1" or report.get("sequence") != sequence
            or report.get("frames") != episode.expected_frames or report.get("business_commit") != commit):
        raise ValueError(f"{sequence}: export does not match episode metadata and the requested business commit")
    acceptance = report.get("acceptance", {})
    if acceptance.get("structural_checks") != "PASS" or acceptance.get("accepted_for_packing") is not True:
        raise ValueError(f"{sequence}: an explicit accepted export report is required; diagnostics cannot be packed")
    for name in ("npz", "mesh"):
        if report.get("output_sha256", {}).get(paths[name].name) != identities[name]["sha256"]:
            raise ValueError(f"{sequence}: {name} changed after export")
    settings = report.get("settings", {})
    acceleration = report.get("conversion", {}).get("added_acceleration", {})
    track1_require_acceleration(acceleration, scoring, settings.get("max_added_acc_h_cm"))
    if acceptance.get("added_acceleration_check") != "PASS":
        raise ValueError(f"{sequence}: added acceleration was not accepted")
    policy = settings.get("conversion_error_policy")
    threshold = settings.get("max_vertex_error_mm")
    errors = np.asarray(report.get("conversion", {}).get("per_frame_mean_vertex_error_mm"), dtype=float)
    if (policy not in {"report", "reject"} or policy != acceptance.get("conversion_error_policy")
            or type(threshold) not in (int, float) or not np.isfinite(threshold) or threshold <= 0
            or errors.shape != (episode.expected_frames,) or not np.isfinite(errors).all()
            or (errors < 0).any() or (policy == "reject" and errors.max() > threshold)):
        raise ValueError(f"{sequence}: inconsistent conversion acceptance")
    with np.load(paths["npz"], allow_pickle=False) as arrays:
        shapes = {"pose": (episode.expected_frames, 136), "scales": (68,), "shape": (45,),
                  "object_rotation": (episode.expected_frames, 3, 3),
                  "object_translation": (episode.expected_frames, 3), "object_scale": ()}
        if set(arrays.files) != set(shapes):
            raise ValueError(f"{sequence}: expected exactly the six submission arrays")
        for name, shape in shapes.items():
            if arrays[name].shape != shape or not np.isfinite(arrays[name]).all():
                raise ValueError(f"{sequence}: {name} must be finite and cover all original frames")
        rotation = arrays["object_rotation"]
        if (not np.allclose(rotation @ rotation.transpose(0, 2, 1), np.eye(3), rtol=0, atol=1e-4)
                or not np.allclose(np.linalg.det(rotation), 1, rtol=0, atol=1e-4)
                or float(arrays["object_scale"]) != 1):
            raise ValueError(f"{sequence}: invalid aligned-mesh object motion")
    provenance = {name: report[name] for name in (
        "business_commit", "image_build_commit", "image_digest", "export_source_sha256", "settings",
        "inference_settings", "decoder_identity")}
    inference = provenance["inference_settings"]
    if inference.get("expected_frames") not in (None, episode.expected_frames):
        raise ValueError(f"{sequence}: inference frame count disagrees with dataset metadata")
    provenance["inference_settings"] = {key: value for key, value in inference.items() if key != "expected_frames"}
    for name in ("official_converter", "mhr_model", "official_metrics", "official_sample", "acceleration_contract"):
        provenance[f"{name}_sha256"] = report["input_sha256"][name]
    return AcceptedExport(episode.episode_index, paths["npz"], paths["mesh"], paths["report"],
                          provenance, identities)


def _kit_identity(kit: Path) -> dict[str, dict[str, Any]]:
    files = [kit / "eval_reconstruction.py"]
    for folder in (kit / "tools", kit / "v2dlb"):
        files.extend(sorted(folder.rglob("*.py")))
    return {str(path.relative_to(kit)): artifact_record(path) for path in files}


def pack_track1(dataset_root: str, export_root: str, submission_kit: str, output_dir: str, *,
                code_commit_url: str, episodes: list[int] | None = None) -> Path:
    """Publish a directory containing a verified Parquet and its complete provenance."""
    import numpy as np
    import pandas as pd

    match = re.fullmatch(r"https://github\.com/[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+/commit/([0-9a-f]{40})", code_commit_url)
    if not match:
        raise ValueError("code_commit_url must be a GitHub URL with a full 40-character commit SHA")
    dataset, exports, kit, output = map(lambda p: Path(p).resolve(),
                                       (dataset_root, export_root, submission_kit, output_dir))
    if output.exists():
        raise FileExistsError(f"Use a new packing output_dir: {output}")
    dataset_identity = track1_metadata_identity(dataset)
    metadata = {e.episode_index: e for e in track1_episodes(dataset)}
    sample_path = kit / "data/track_1_sample_submission.parquet"
    sample_identity = artifact_record(sample_path)
    sample = pd.read_parquet(sample_path, columns=["row_id"])
    if sample.empty or sample["row_id"].duplicated().any() or sample["row_id"].isna().any():
        raise ValueError("Official sample must contain unique, nonempty row IDs")
    keys = sample["row_id"].str.extract(r"^t1_(\d{6})_(\d{6})_([0-7])_(\d{6})$")
    if keys.isna().any().any():
        raise ValueError("Official sample has invalid Track 1 row IDs")
    keys = keys.astype(int)
    roster = set(keys[0])
    if roster != set(metadata):
        raise ValueError("The full official sample roster and dataset episode roster disagree")
    if episodes is not None and (not episodes or len(set(episodes)) != len(episodes)
                                 or any(type(i) is not int or i not in roster for i in episodes)):
        raise ValueError("episodes must be a nonempty, unique subset of the official episode roster")
    selected = roster if episodes is None else set(episodes)
    sample = sample.loc[keys[0].isin(selected)].reset_index(drop=True)
    for index in sorted(selected):
        frames = keys.loc[(keys[0] == index) & (keys[1] != 999999), 1]
        if frames.empty or frames.max() >= metadata[index].expected_frames:
            raise ValueError(f"episode {index}: official scored frames exceed dataset metadata")
    joint_indices = track1_body_joint_indices(kit / "v2dlb/mhr_metrics.py")
    accepted = [_accepted_export(exports, metadata[i], match.group(1), Track1Scoring(
        joint_indices, tuple(sorted(set(keys.loc[(keys[0] == i) & (keys[2] == 0), 1])))))
        for i in sorted(selected)]
    if any(item.provenance != accepted[0].provenance for item in accepted[1:]):
        raise ValueError("Exports mix business/image versions, converter sources, or export settings")
    converter_hash = artifact_record(kit / "tools/track1/mesh_to_mhr_params.py")["sha256"]
    if accepted[0].provenance["official_converter_sha256"] != converter_hash:
        raise ValueError("Exported fits and the requested submission kit use different converter sources")
    for name, path in (("official_metrics", kit / "v2dlb/mhr_metrics.py"), ("official_sample", sample_path)):
        if accepted[0].provenance[f"{name}_sha256"] != artifact_record(path)["sha256"]:
            raise ValueError("Added-acceleration check and packing use different official scoring contracts")
    kit_identity = _kit_identity(kit)
    output.parent.mkdir(parents=True, exist_ok=True)
    work = Path(tempfile.mkdtemp(prefix=f".{output.name}-", dir=output.parent))
    try:
        staged = work / "episodes"
        staged.mkdir()
        for episode in accepted:
            for role, path in (("npz", episode.npz), ("mesh", episode.mesh), ("report", episode.report)):
                destination = staged / path.name
                shutil.copyfile(path, destination)
                if artifact_record(destination)["sha256"] != episode.identities[role]["sha256"]:
                    raise ValueError(f"Export changed while collecting {path}")
        sample_file = work / "sample.parquet"
        sample.to_parquet(sample_file, index=False)
        parquet = work / "submission.parquet"
        command = [sys.executable, str(kit / "eval_reconstruction.py"), "--episodes", str(staged),
                   "--sample", str(sample_file), "--commit", code_commit_url, "--out", str(parquet)]
        with (work / "pack.log").open("w") as log:
            subprocess.run(command, cwd=kit, stdout=log, stderr=subprocess.STDOUT, check=True)
        table = pd.read_parquet(parquet)
        if (table.columns.tolist() != ["row_id", "x", "y", "z", "code_commit_url"]
                or not table["row_id"].equals(sample["row_id"])
                or not np.isfinite(table[["x", "y", "z"]].to_numpy()).all()
                or not (table["code_commit_url"] == code_commit_url).all()):
            raise ValueError("Packed output does not exactly match the requested official rows and commit")
        if (artifact_record(sample_path) != sample_identity or _kit_identity(kit) != kit_identity
                or track1_metadata_identity(dataset) != dataset_identity):
            raise ValueError("Official kit, roster, or dataset metadata changed while packing")
        for episode in accepted:
            for role, path in (("npz", episode.npz), ("mesh", episode.mesh), ("report", episode.report)):
                if artifact_record(path) != episode.identities[role]:
                    raise ValueError(f"Export changed while packing: {path}")
        record = {
            "schema": "v2d.track1.packing.v1", "verdict": "PASS",
            "scope": "full_official_roster" if episodes is None else "subset_format_check",
            "episodes": sorted(selected), "rows": len(table), "code_commit_url": code_commit_url,
            "official_sample": sample_identity, "official_kit_sources": kit_identity,
            "dataset_metadata": dataset_identity,
            "packer_adapter": artifact_record(Path(__file__)),
            "export_provenance": accepted[0].provenance,
            "exports": {str(e.episode_index): e.identities for e in accepted},
            "submission_sha256": artifact_record(parquet)["sha256"],
            "kaggle_submitted": False, "kaggle_scored": False,
        }
        atomic_json(work / "packing_report.json", record)
        if output.exists():
            raise FileExistsError(output)
        work.rename(output)
    except Exception:
        print(f"Packing failed; unpublished diagnostics retained at {work}", file=sys.stderr)
        raise
    return output / "submission.parquet"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("dataset_root", "export_root", "submission_kit", "output_dir", "code_commit_url"):
        parser.add_argument(f"--{name}", required=True)
    parser.add_argument("--episodes", nargs="+", type=int, help="Explicit subset format check; never a full submission")
    return parser


if __name__ == "__main__":
    print(pack_track1(**vars(_parser().parse_args())))
