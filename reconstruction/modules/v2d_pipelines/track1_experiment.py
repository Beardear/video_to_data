"""Source and weight identities shared by Track 1 preparation and inference jobs."""

from pathlib import Path
import subprocess
from typing import Any

from v2d.common.artifacts import artifact_record


def checkout_identity(repository: Path, require_clean: bool) -> dict[str, Any]:
    def git(*args):
        return subprocess.run(["git", "-C", str(repository), *args], check=True,
                              capture_output=True, text=True).stdout.strip()
    commit = git("rev-parse", "HEAD")
    dirty = git("status", "--porcelain", "--untracked-files=normal")
    if require_clean and dirty:
        raise ValueError("Commit the business checkout before execution; planning permits uncommitted edits")
    return {"repository": str(repository), "business_commit": commit, "dirty": bool(dirty)}


def weight_identity(root: Path) -> dict[str, dict[str, Any]]:
    if root.is_file():
        return {root.name: artifact_record(root)}
    excluded = {".git", "__pycache__", ".cache", ".locks"}
    files = {str(p.relative_to(root)): artifact_record(p) for p in sorted(root.rglob("*"))
             if p.is_file() and not excluded.intersection(p.relative_to(root).parts)
             and p.suffix not in {".pyc", ".lock"}}
    if not files:
        raise FileNotFoundError(f"Weights are missing or empty: {root}")
    return files
