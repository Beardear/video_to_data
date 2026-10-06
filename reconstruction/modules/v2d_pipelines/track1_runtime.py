"""Bounded subprocess execution of existing module CLIs, with no ML imports."""

from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
from typing import Any, Mapping
import uuid


@dataclass(frozen=True)
class ModuleInvocation:
    name: str
    command: tuple[str, ...]
    log_path: Path


@dataclass(frozen=True)
class Track1Runtime:
    modules_dir: Path
    mode: str = "docker"
    python: str = sys.executable

    def __post_init__(self) -> None:
        if self.mode not in {"docker", "container"}:
            raise ValueError("Runtime mode must be docker or container")

    def invocation(self, name: str, package: str, library_script: str, wrapper_script: str,
                   arguments: Mapping[str, Any], log_path: Path) -> ModuleInvocation:
        folder = "lib" if self.mode == "container" else "docker"
        script = self.modules_dir / package / folder / (library_script if folder == "lib" else wrapper_script)
        if not script.is_file():
            raise FileNotFoundError(script)
        command = [self.python, str(script)]
        for key, value in arguments.items():
            if key == "download_models":
                if not value:
                    command.append("--skip_weight_download")
            elif value is True:
                command.append(f"--{key}")
            elif value is not False and value is not None:
                command.extend((f"--{key}", str(value)))
        if self.mode == "docker":
            command.append("--dev")
        return ModuleInvocation(name, tuple(command), log_path)

    def verify_packages(self, packages: Mapping[str, Path]) -> None:
        """Prove imports resolve to this checkout, rather than the old baked image."""
        probe = ("import importlib.util,json,sys; "
                 "print(json.dumps({n:list(importlib.util.find_spec(n).submodule_search_locations or []) "
                 "for n in json.loads(sys.argv[1])}))")
        result = subprocess.run([self.python, "-c", probe, json.dumps(list(packages))],
                                check=True, capture_output=True, text=True, timeout=60)
        resolved = json.loads(result.stdout)
        for name, expected in packages.items():
            if str(expected.resolve()) not in {str(Path(p).resolve()) for p in resolved[name]}:
                raise ValueError(f"{name} is not installed from {expected}; install this checkout before execution")

    def execute(self, invocation: ModuleInvocation, timeout_seconds: float) -> None:
        """Terminate the process group on timeout or cancellation; retain the log."""
        invocation.log_path.parent.mkdir(parents=True, exist_ok=True)
        environment = {**os.environ, "PYTHONUNBUFFERED": "1"}
        container_name = f"v2d-track1-{uuid.uuid4().hex}" if self.mode == "docker" else None
        if container_name is not None:
            environment["V2D_DOCKER_CONTAINER_NAME"] = container_name
        with invocation.log_path.open("w") as log:
            process = subprocess.Popen(invocation.command, stdout=log, stderr=subprocess.STDOUT,
                                       start_new_session=True, env=environment)
            try:
                code = process.wait(timeout=timeout_seconds)
                if code:
                    raise subprocess.CalledProcessError(code, invocation.command)
            except BaseException:
                # Workers can outlive a failed parent, so signal the group even
                # when the group leader has already exited.
                try:
                    os.killpg(process.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    pass
                finally:
                    try:
                        os.killpg(process.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                    process.wait()
                raise
            finally:
                if container_name is not None:
                    cleanup = subprocess.run(["docker", "rm", "--force", container_name],
                                             capture_output=True, text=True, timeout=30)
                    if cleanup.returncode and "no such container" not in cleanup.stderr.lower():
                        raise RuntimeError(f"Cannot verify cleanup of {container_name}: {cleanup.stderr.strip()}")

    def verify_cari4d_image(self, image_digest: str) -> dict[str, Any]:
        if self.mode == "container":
            return {"source": "deployment_record", "digest": image_digest}
        query = "from v2d.cari4d.docker._config import IMAGE_NAME; print(IMAGE_NAME)"
        image = subprocess.run([self.python, "-c", query], check=True, capture_output=True,
                               text=True, timeout=60).stdout.strip()
        result = subprocess.run(["docker", "image", "inspect", image], check=True,
                                capture_output=True, text=True, timeout=60)
        metadata = json.loads(result.stdout)[0]
        expected = image_digest.split("@")[-1]
        if expected not in {value.split("@")[-1] for value in metadata.get("RepoDigests", [])}:
            raise ValueError(f"Docker image {image} does not match the recorded immutable digest")
        return {"source": "docker_inspect", "digest": image_digest, "image_id": metadata["Id"]}
