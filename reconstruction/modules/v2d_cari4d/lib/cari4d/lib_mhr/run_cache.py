"""Bind an inference directory to the code, inputs, and settings that produced it."""

from __future__ import annotations

import fcntl
from functools import wraps
import hashlib
from importlib import metadata
import inspect
import json
from pathlib import Path
import platform
import sys
from typing import Any, Callable, Mapping

from v2d.common.artifacts import artifact_record, atomic_json


SOURCE_SUFFIXES = {".py", ".yaml", ".yml", ".json", ".toml", ".txt", ".cpp", ".cu", ".h",
                   ".hpp", ".so", ".npz", ".npy", ".pkl"}
IGNORED_PARTS = {".git", "__pycache__", ".pytest_cache", ".mypy_cache", ".ruff_cache"}


def source_identity(root: Path) -> dict[str, Any]:
    root = root.resolve()
    files = {}
    for path in sorted(root.rglob("*")):
        relative = path.relative_to(root)
        if (path.is_file() and path.suffix in SOURCE_SUFFIXES
                and not IGNORED_PARTS.intersection(relative.parts)):
            files[str(relative)] = artifact_record(path)["sha256"]
    if not files:
        raise FileNotFoundError(f"Required runtime source tree is missing or empty: {root}")
    return {"root": str(root), "files": len(files), "sha256": _digest(files)}


def _digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def run_identity(inputs: Mapping[str, Path], sources: Mapping[str, Path],
                 settings: Mapping[str, Any]) -> dict[str, Any]:
    """Fingerprint bytes, including dirty source edits, instead of trusting a Git label."""
    payload = {
        "schema": "v2d.cari4d.run_identity.v1",
        "inputs": {name: artifact_record(path) for name, path in sorted(inputs.items())},
        "sources": {name: source_identity(path) for name, path in sorted(sources.items())},
        "settings": dict(settings),
        "runtime": {"python": sys.version, "executable": sys.executable,
                    "platform": platform.platform(),
                    "packages": sorted([d.metadata["Name"], d.version] for d in metadata.distributions()
                                       if d.metadata["Name"])},
    }
    return {**payload, "sha256": _digest(payload)}


def bind_run(output_root: Path, identity: dict[str, Any]) -> Path:
    """Resume only the same experiment; overwrite never bypasses this boundary."""
    marker = output_root / "run_identity.json"
    if marker.exists():
        if json.loads(marker.read_text()) != identity:
            raise ValueError("Inference source, inputs, runtime, or settings changed; use a new output_dir")
    elif output_root.exists() and any(output_root.iterdir()):
        raise ValueError("Existing inference results have no run identity; use a new output_dir")
    else:
        atomic_json(marker, identity)
    return marker


def episode_run_lock(function: Callable) -> Callable:
    """Reject concurrent writers to one episode, including separate processes."""
    signature = inspect.signature(function)

    @wraps(function)
    def locked(*args, **kwargs):
        arguments = signature.bind(*args, **kwargs).arguments
        video = Path(arguments["video_path"]).resolve()
        sequence = video.name.removesuffix(".0.color.mp4")
        parent = Path(arguments["output_dir"]).resolve()
        parent.mkdir(parents=True, exist_ok=True)
        # Leave the inode in place: unlinking a lock allows two different inodes
        # to be locked concurrently by waiting/new processes.
        with (parent / f".{sequence}.lock").open("a") as handle:
            try:
                fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as exc:
                raise RuntimeError(f"Another process is using {parent / sequence}") from exc
            try:
                return function(*args, **kwargs)
            finally:
                fcntl.flock(handle, fcntl.LOCK_UN)

    return locked
