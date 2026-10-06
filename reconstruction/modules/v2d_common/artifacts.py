"""Content identities and atomic JSON records for file-based pipeline artifacts."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any


@dataclass(frozen=True)
class ArtifactIdentity:
    path: str
    size: int
    sha256: str


def artifact_identity(path: str | Path) -> ArtifactIdentity:
    path = Path(path).resolve()
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        before = os.fstat(handle.fileno())
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
        after = os.fstat(handle.fileno())
    current = path.stat()
    stamp = lambda st: (st.st_dev, st.st_ino, st.st_size, st.st_mtime_ns, st.st_ctime_ns)
    if stamp(before) != stamp(after) or stamp(after) != stamp(current):
        raise ValueError(f"Artifact changed while hashing: {path}")
    return ArtifactIdentity(str(path), after.st_size, digest.hexdigest())


def atomic_json(path: str | Path, payload: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}-", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def artifact_record(path: str | Path) -> dict[str, Any]:
    return asdict(artifact_identity(path))


@contextmanager
def artifact_lock(path: str | Path):
    """Hold a nonblocking POSIX writer lock; retain its inode after release."""
    import fcntl

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(f"Another process is using {path}") from exc
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)
