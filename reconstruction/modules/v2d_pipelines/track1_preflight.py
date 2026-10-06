# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Inventory Track 1 metadata and prepared input files without importing ML code.

Run as ``python -m v2d.pipelines.track1_preflight`` after installing the host
package, or execute this file directly with Python 3.10+. JSON goes to stdout.
Exit codes: 0 = required files present, 2 = preparation needed, 1 = invalid input.
File presence does not establish valid masks, geometry, or GPU readiness.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence


INPUT_SCHEMA = "v2d.track1.inputs.v1"
REPORT_SCHEMA = "v2d.track1.preflight.v1"
INPUT_ROLES = ("video_path", "mask_h5_path", "object_mesh_path")


@dataclass(frozen=True)
class Track1Episode:
    episode_index: int
    sequence_id: str
    camera: str
    object_name: str
    object_prompt: str
    tasks: tuple[str, ...]
    expected_frames: int
    fps: float
    source_video: Path


@dataclass(frozen=True)
class InputFileCheck:
    path: str | None
    status: str


def _integer(value: Any, label: str, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise ValueError(f"{label}: expected integer >= {minimum}")
    return value


def _text(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label}: expected nonempty text")
    return value


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected a JSON object")
    return value


def _indexed_records(records: Any, key: str, label: str) -> dict[int, dict[str, Any]]:
    if not isinstance(records, list):
        raise ValueError(f"{label}: expected a list")
    indexed = {}
    for row in records:
        if not isinstance(row, dict):
            raise ValueError(f"{label}: expected object records")
        index = _integer(row.get(key), f"{label}.{key}")
        if index in indexed:
            raise ValueError(f"{label}: duplicate {key} {index}")
        indexed[index] = row
    return indexed


def _read_jsonl(path: Path, key: str) -> dict[int, dict[str, Any]]:
    records = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    return _indexed_records(records, key, str(path))


def _dataset_path(root: Path, relative: str) -> Path:
    path = (root / relative).resolve()
    if not path.is_relative_to(root):
        raise ValueError(f"Dataset path must stay within {root}: {relative}")
    return path


def track1_episodes(dataset_root: Path) -> tuple[Track1Episode, ...]:
    """Read the complete released episode roster, checking metadata consistency."""
    root = dataset_root.resolve()
    info = _read_json(root / "meta/info.json")
    if info.get("codebase_version") != "v2.1":
        raise ValueError("Expected a LeRobot v2.1 Track 1 dataset")
    count = _integer(info.get("total_episodes"), "total_episodes", 1)
    total_frames = _integer(info.get("total_frames"), "total_frames", 1)
    chunk_size = _integer(info.get("chunks_size"), "chunks_size", 1)
    fps = info.get("fps")
    if type(fps) not in (int, float) or not math.isfinite(fps) or fps <= 0:
        raise ValueError("fps must be finite and positive")
    video_template = _text(info.get("video_path"), "video_path")
    features = info.get("features")
    if not isinstance(features, dict):
        raise ValueError("features must be a JSON object")
    episodes = _read_jsonl(root / "meta/episodes.jsonl", "episode_index")
    metadata = _read_jsonl(_dataset_path(root, _text(
        info.get("episodes_metadata"), "episodes_metadata")), "episode_index")
    tasks = _read_jsonl(root / "meta/tasks.jsonl", "task_index")
    task_texts = {_text(row.get("task"), "task") for row in tasks.values()}
    expected = set(range(count))
    if set(episodes) != expected or set(metadata) != expected:
        raise ValueError("Episode metadata must cover exactly 0..total_episodes-1")
    result = []
    for index in sorted(episodes):
        row, meta = episodes[index], metadata[index]
        frames = _integer(row.get("length"), f"episode {index} length", 1)
        descriptions = row.get("tasks")
        if not isinstance(descriptions, list) or not descriptions:
            raise ValueError(f"episode {index}: expected action descriptions")
        descriptions = tuple(_text(task, f"episode {index} task") for task in descriptions)
        if not set(descriptions).issubset(task_texts):
            raise ValueError(f"episode {index}: action missing from tasks.jsonl")
        video_key = _text(meta.get("video_key"), f"episode {index} video_key")
        feature = features.get(video_key)
        if not isinstance(feature, dict) or feature.get("dtype") != "video":
            raise ValueError(f"episode {index}: unknown video feature {video_key}")
        relative = video_template.format(
            episode_chunk=index // chunk_size, episode_index=index, video_key=video_key)
        result.append(Track1Episode(
            episode_index=index,
            sequence_id=_text(meta.get("sequence_id"), f"episode {index} sequence_id"),
            camera=_text(meta.get("camera"), f"episode {index} camera"),
            object_name=_text(meta.get("object"), f"episode {index} object"),
            object_prompt=_text(meta.get("object_prompt"), f"episode {index} object_prompt"),
            tasks=descriptions, expected_frames=frames, fps=float(fps),
            source_video=_dataset_path(root, relative),
        ))
    if sum(episode.expected_frames for episode in result) != total_frames:
        raise ValueError("Episode lengths do not add up to total_frames")
    if len({episode.source_video for episode in result}) != count:
        raise ValueError("Video template maps multiple episodes to the same file")
    return tuple(result)


def _prepared_inputs(path: Path | None, known: set[int]) -> dict[int, dict[str, Path]]:
    if path is None:
        return {}
    path = path.resolve()
    payload = _read_json(path)
    if payload.get("schema") != INPUT_SCHEMA:
        raise ValueError(f"{path}: expected schema {INPUT_SCHEMA}")
    rows = _indexed_records(payload.get("episodes"), "episode_index", str(path))
    if set(rows) - known:
        raise ValueError(f"Inputs refer to unknown episodes: {sorted(set(rows) - known)}")
    result = {}
    for index, row in rows.items():
        unknown = set(row) - {"episode_index", *INPUT_ROLES}
        if unknown:
            raise ValueError(f"episode {index}: unknown input fields {sorted(unknown)}")
        paths = {}
        for role in INPUT_ROLES:
            value = row.get(role)
            if value is not None:
                paths[role] = (path.parent / _text(value, role)).resolve()
        result[index] = paths
    return result


def _file_check(path: Path | None) -> InputFileCheck:
    if path is None:
        return InputFileCheck(None, "not_configured")
    if not path.exists():
        status = "missing"
    elif not path.is_file():
        status = "not_file"
    elif path.stat().st_size == 0:
        status = "empty"
    else:
        status = "present"
    return InputFileCheck(str(path), status)


def track1_preflight(
    dataset_root: Path,
    inputs_manifest: Path | None = None,
    episode_indices: Sequence[int] | None = None,
) -> dict[str, Any]:
    """Return a file-presence report; never launch inference or create inputs."""
    episodes = track1_episodes(dataset_root)
    known = {episode.episode_index for episode in episodes}
    prepared = _prepared_inputs(inputs_manifest, known)
    selected = known if episode_indices is None else {
        _integer(index, "selected episode index") for index in episode_indices}
    if not selected or selected - known:
        raise ValueError(f"Select known episode indexes; unknown: {sorted(selected - known)}")
    rows = []
    for episode in episodes:
        index = episode.episode_index
        if index not in selected:
            continue
        paths = prepared.get(index, {})
        checks = {"source_video": _file_check(episode.source_video)}
        checks.update({role: _file_check(paths.get(role)) for role in INPUT_ROLES})
        issues = [f"{role}: {check.status}" for role, check in checks.items()
                  if check.status != "present"]
        video = paths.get("video_path")
        if video is not None and video.name != f"episode_{index:06d}.0.color.mp4":
            issues.append("video_path: resolved filename must be "
                          f"episode_{index:06d}.0.color.mp4")
        row = asdict(episode)
        row["source_video"] = str(episode.source_video)
        row["files"] = {role: asdict(check) for role, check in checks.items()}
        row["issues"] = issues
        rows.append(row)
    return {
        "schema": REPORT_SCHEMA,
        "validation_level": "metadata_and_file_presence",
        "not_checked": ["decoded_video_frames", "mask_contents", "mesh_geometry_and_scale",
                        "weights", "gpu_runtime", "source_version_and_cache"],
        "dataset_root": str(dataset_root.resolve()),
        "inputs_manifest": str(inputs_manifest.resolve()) if inputs_manifest else None,
        "summary": {
            "dataset_episodes": len(episodes),
            "selected_episodes": len(rows),
            "expected_frames": sum(row["expected_frames"] for row in rows),
            "source_videos_present": sum(row["files"]["source_video"]["status"] == "present"
                                         for row in rows),
            "episodes_with_no_file_issues": sum(not row["issues"] for row in rows),
            "episodes_needing_preparation": sum(bool(row["issues"]) for row in rows),
        },
        "episodes": rows,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset_root", type=Path, required=True,
                        help="Track 1 directory containing meta/ and videos/")
    parser.add_argument("--inputs_manifest", type=Path,
                        help="Optional v2d.track1.inputs.v1 JSON; paths relative to this file")
    parser.add_argument("--episodes", type=int, nargs="+", help="Episode indexes; default: all")
    args = parser.parse_args(argv)
    try:
        report = track1_preflight(args.dataset_root, args.inputs_manifest, args.episodes)
    except (OSError, ValueError, KeyError, TypeError) as error:
        print(f"track1_preflight: {error}", file=sys.stderr)
        return 1
    print(json.dumps(report, indent=2, allow_nan=False))
    return 2 if report["summary"]["episodes_needing_preparation"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
