"""Exercise cache decisions with real subprocesses, without any neural models."""

import ast
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any, Sequence
from types import SimpleNamespace

import pytest

MODULE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(MODULE / "lib/cari4d"))
from v2d.common.artifacts import artifact_record, atomic_json
from lib_mhr.run_cache import bind_run, episode_run_lock, run_identity, source_identity


@pytest.fixture
def experiment(tmp_path):
    source = tmp_path / "code"
    source.mkdir()
    script = source / "model.py"
    script.write_text("# version A\n")
    input_file = tmp_path / "video"
    input_file.write_bytes(b"video A")
    inputs = {"video": input_file}
    sources = {"model": source}
    settings = {"steps": 300}
    output = tmp_path / "output"
    return inputs, sources, settings, output


def test_same_experiment_can_resume(experiment):
    inputs, sources, settings, output = experiment
    identity = run_identity(inputs, sources, settings)
    marker = bind_run(output, identity)
    (output / "partial.pth").write_bytes(b"partial checkpoint")
    assert bind_run(output, run_identity(inputs, sources, settings)) == marker


@pytest.mark.parametrize("change", ["source", "input", "settings", "add_source", "delete_source"])
def test_changes_never_reuse_old_directory(experiment, change):
    inputs, sources, settings, output = experiment
    bind_run(output, run_identity(inputs, sources, settings))
    if change in {"source", "input"}:
        path = sources["model"] / "model.py" if change == "source" else inputs["video"]
        stamp = path.stat()
        # Same byte count and mtime: metadata-based identities would miss this.
        path.write_bytes(path.read_bytes().replace(b"A", b"B"))
        os.utime(path, ns=(stamp.st_atime_ns, stamp.st_mtime_ns))
    elif change == "settings":
        settings["steps"] = 500
    elif change == "add_source":
        (sources["model"] / "new.py").write_text("# new runtime code\n")
    else:
        (sources["model"] / "model.py").unlink()
        (sources["model"] / "different.py").write_text("# version A\n")
    with pytest.raises(ValueError, match="new output_dir"):
        bind_run(output, run_identity(inputs, sources, settings))


def test_legacy_results_are_preserved_but_cannot_be_adopted(experiment):
    inputs, sources, settings, output = experiment
    output.mkdir()
    legacy = output / "refined.pth"
    legacy.write_bytes(b"legacy")
    with pytest.raises(ValueError, match="no run identity"):
        bind_run(output, run_identity(inputs, sources, settings))
    assert legacy.read_bytes() == b"legacy"
    assert not (output / "run_identity.json").exists()


def test_python_bytecode_does_not_change_source_identity(experiment):
    root = experiment[1]["model"]
    identity = source_identity(root)
    (root / "__pycache__").mkdir()
    (root / "__pycache__/model.pyc").write_bytes(b"compiled")
    assert source_identity(root) == identity


def test_lock_rejects_another_process_and_is_released_after_failure(tmp_path):
    video = tmp_path / "episode_000016.0.color.mp4"
    output = tmp_path / "out"
    calls = []

    @episode_run_lock
    def operation(video_path, output_dir):
        script = ("import fcntl,sys; f=open(sys.argv[1], 'a'); "
                  "fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)")
        result = subprocess.run([sys.executable, "-c", script, str(output / ".episode_000016.lock")],
                                capture_output=True)
        assert result.returncode != 0
        calls.append(True)
        raise ValueError("simulated failure")

    for _ in range(2):
        with pytest.raises(ValueError, match="simulated failure"):
            operation(str(video), str(output))
    assert len(calls) == 2


@pytest.fixture
def stage(tmp_path):
    # Extract only the real stage helpers: downloader imports need the ML image.
    namespace = dict(Path=Path, Any=Any, Sequence=Sequence, hashlib=hashlib, json=json,
                     os=os, time=time, subprocess=subprocess, SOURCE_ROOT=tmp_path,
                     artifact_record=artifact_record, atomic_json=atomic_json)
    tree = ast.parse((MODULE / "lib/run_inference.py").read_text())
    names = {"_file_identity", "_atomic_json", "_signature", "_output_identities", "_stage_run"}
    functions = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names]
    exec(compile(ast.fix_missing_locations(ast.Module(body=functions, type_ignores=[])),
                 "run_inference.py", "exec"), namespace)
    source = tmp_path / "input"
    source.write_text("input")
    output = tmp_path / "output"
    root = tmp_path / "episode"
    root.mkdir()
    atomic_json(root / "run_identity.json", {"fixture": "current source"})
    command = [sys.executable, "-c", "from pathlib import Path; import sys; Path(sys.argv[1]).write_text('result')", str(output)]
    arguments = dict(name="fixture", command=command, inputs=[source], outputs=[output],
                     marker_root=root / ".stages", env=os.environ.copy(), overwrite=False)
    return namespace["_stage_run"], arguments


def test_completed_stage_is_reused_and_modified_output_is_rejected(stage):
    run, args = stage
    assert not run(**args)["reused"]
    assert run(**args)["reused"]
    output = args["outputs"][0]
    stamp = output.stat()
    output.write_text("edited")
    os.utime(output, ns=(stamp.st_atime_ns, stamp.st_mtime_ns))
    with pytest.raises(ValueError, match="do not match"):
        run(**args)


def test_stage_includes_run_source_identity(stage):
    run, args = stage
    run(**args)
    atomic_json(args["marker_root"].parent / "run_identity.json", {"fixture": "other source"})
    with pytest.raises(ValueError, match="do not match"):
        run(**args)


def test_changed_inputs_during_stage_are_never_marked_complete(stage):
    run, args = stage
    args["command"][2] += "; Path(sys.argv[2]).write_text('changed')"
    args["command"].append(str(args["inputs"][0]))
    with pytest.raises(ValueError, match="Inputs changed"):
        run(**args)
    assert not (args["marker_root"] / "fixture.json").exists()


def test_pipeline_binds_all_stages_and_requires_a_new_experiment_on_change(tmp_path, capsys):
    source = tmp_path / "source"
    source.mkdir()
    (source / "model.py").write_text("# model A")
    calls = []

    def execute(command, **kwargs):
        calls.append(command)
        # Model-free stage fixtures exercise the actual orchestration and cache.
        for option in ("--output-depth", "--output-intrinsics", "--output-report", "--out-file",
                       "--sam3d-cache", "--output", "--input-cache", "--out",
                       "--postopt-checkpoint", "--comparison-output"):
            if option in command:
                path = Path(command[command.index(option) + 1])
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("synthetic stage output")
        if "--output-root" in command:
            path = Path(command[command.index("--output-root") + 1]) / "episode_000016"
            (path / "object_mesh").mkdir(parents=True)
            (path / "wild_export.json").write_text("synthetic export manifest")
            (path / "object_mesh/output_aligned.glb").write_text("synthetic geometry")

    namespace = dict(Path=Path, Any=Any, Sequence=Sequence, hashlib=hashlib, json=json,
                     os=os, time=time, sys=sys, subprocess=SimpleNamespace(run=execute),
                     SOURCE_ROOT=source, SAM3D_SOURCE_ROOT=source, PIPELINE_SCHEMA="v2d.cari4d.wild_inference.v1",
                     artifact_record=artifact_record, atomic_json=atomic_json, run_identity=run_identity,
                     bind_run=bind_run, episode_run_lock=episode_run_lock,
                     CARI4D_CHECKPOINT_SHA256="fixture", CARI4D_REVISION="fixture")
    tree = ast.parse((MODULE / "lib/run_inference.py").read_text())
    functions = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name not in {"_parser", "main"}]
    exec(compile(ast.fix_missing_locations(ast.Module(body=functions, type_ignores=[])),
                 "run_inference.py", "exec"), namespace)
    weight = tmp_path / "weights/model"
    weight.parent.mkdir()
    weight.write_bytes(b"synthetic weights")
    namespace["_required_weights"] = lambda _: {name: weight for name in (
        "checkpoint", "config", "manifest", "sam3d_checkpoint", "mhr_model",
        "foundationpose_score", "foundationpose_refine", "moge2_model")}
    namespace["_runtime_source_roots"] = lambda _: {"model": source}
    arguments = dict(video_path=str(tmp_path / "episode_000016.0.color.mp4"),
                     mask_h5_path=str(tmp_path / "masks"), object_mesh_path=str(tmp_path / "mesh"),
                     weights_path=str(weight.parent), output_dir=str(tmp_path / "results"),
                     download_models=False, expected_frames=360)
    for name in ("video_path", "mask_h5_path", "object_mesh_path"):
        Path(arguments[name]).write_bytes(b"input fixture")
    run = namespace["run_inference"]
    report_path = run(**arguments)
    report = json.loads(report_path.read_text())
    assert len(calls) == len(report["stages"]) == 8
    assert report["run_identity"]["sha256"]
    run(**arguments)
    assert len(calls) == 8
    assert all(s["reused"] for s in json.loads(report_path.read_text())["stages"])
    with pytest.raises(ValueError, match="new output_dir"):
        run(**arguments, postopt_num_steps=1, overwrite=True)
    (source / "model.py").write_text("# model B")
    with pytest.raises(ValueError, match="new output_dir"):
        run(**arguments)
    assert len(calls) == 8
