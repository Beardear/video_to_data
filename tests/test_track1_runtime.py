"""Real CPU subprocess checks for the bounded module runner."""

import importlib.util
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

from v2d.pipelines.track1_runtime import ModuleInvocation, Track1Runtime


def test_timeout_terminates_the_child_process_group(tmp_path):
    child_file = tmp_path / "child.txt"
    script = ("import os,subprocess,sys,time; from pathlib import Path; "
              "p=subprocess.Popen([sys.executable,'-c','import time; time.sleep(60)']); "
              "Path(sys.argv[1]).write_text(str(p.pid)); print('started',flush=True); time.sleep(60)")
    invocation = ModuleInvocation("timeout", (sys.executable, "-c", script, str(child_file)), tmp_path / "timeout.log")
    with pytest.raises(subprocess.TimeoutExpired):
        Track1Runtime(tmp_path, "container").execute(invocation, 1)
    child = int(child_file.read_text())
    # A terminated child may briefly remain as a zombie awaiting its reaper.
    result = subprocess.run(["ps", "-o", "stat=", "-p", str(child)], capture_output=True, text=True)
    assert not result.stdout.strip() or result.stdout.strip().startswith("Z")
    assert "started" in invocation.log_path.read_text()


def test_failed_process_retains_stderr_and_exit_status(tmp_path):
    invocation = ModuleInvocation("failure", (sys.executable, "-c", "import sys; print('failure',file=sys.stderr); sys.exit(7)"), tmp_path / "failed.log")
    with pytest.raises(subprocess.CalledProcessError) as caught:
        Track1Runtime(tmp_path, "container").execute(invocation, 10)
    assert caught.value.returncode == 7
    assert "failure" in invocation.log_path.read_text()


def test_package_origin_check_detects_a_different_checkout(tmp_path):
    runtime = Track1Runtime(tmp_path, "container")
    with pytest.raises(ValueError, match="not installed from"):
        runtime.verify_packages({"v2d.common": tmp_path / "wrong_checkout"})


def test_workers_are_stopped_when_the_parent_exits_with_failure(tmp_path):
    child_file = tmp_path / "orphan.txt"
    script = ("import subprocess,sys; from pathlib import Path; "
              "p=subprocess.Popen([sys.executable,'-c','import time; time.sleep(60)']); "
              "Path(sys.argv[1]).write_text(str(p.pid)); sys.exit(3)")
    invocation = ModuleInvocation("orphan", (sys.executable, "-c", script, str(child_file)), tmp_path / "orphan.log")
    with pytest.raises(subprocess.CalledProcessError):
        Track1Runtime(tmp_path, "container").execute(invocation, 10)
    result = subprocess.run(["ps", "-o", "stat=", "-p", child_file.read_text()], capture_output=True, text=True)
    assert not result.stdout.strip() or result.stdout.strip().startswith("Z")


def test_docker_runtime_cleans_only_its_uniquely_named_container(tmp_path, monkeypatch):
    removed = []
    def cleanup(command, **kwargs):
        assert command[:3] == ["docker", "rm", "--force"]
        removed.append(command[3])
        return SimpleNamespace(returncode=1, stderr="Error: No such container")
    monkeypatch.setattr(subprocess, "run", cleanup)
    invocation = ModuleInvocation("docker", (sys.executable, "-c",
        "import os; print(os.environ['V2D_DOCKER_CONTAINER_NAME'])"), tmp_path / "docker.log")
    Track1Runtime(tmp_path, "docker").execute(invocation, 10)
    assert removed[0].startswith("v2d-track1-")
    assert invocation.log_path.read_text().strip() == removed[0]


def test_shared_docker_wrapper_uses_the_orchestrators_container_name(tmp_path, monkeypatch):
    path = Path(__file__).resolve().parents[1] / "reconstruction/modules/v2d_docker/container.py"
    spec = importlib.util.spec_from_file_location("container_fixture", path)
    container = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(container)
    calls = []
    monkeypatch.setenv("V2D_DOCKER_CONTAINER_NAME", "v2d-track1-test-fixture")
    monkeypatch.setattr(container.subprocess, "run", lambda command, **kwargs: calls.append(command))
    container.run_in_container("test-image", "fixture.module", {}, {"output": str(tmp_path / "result")})
    command = calls[0]
    assert command[command.index("--name") + 1] == "v2d-track1-test-fixture"
