"""CPU adapter contracts. These do not validate neural decoding or MHR fitting."""

from __future__ import annotations

import argparse
import ast
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest
import trimesh
from v2d.common.track1_conversion import track1_added_acceleration, track1_scoring


MODULE = Path(__file__).resolve().parents[1]
SOURCE = MODULE / "lib/cari4d"
if str(SOURCE) not in sys.path:
    sys.path.insert(0, str(SOURCE))
from lib_mhr.object_pose_frame import stamp_object_pose_frame_metadata
from lib_mhr.schema import MHR_PARAM_DIMS
from lib_mhr.track1 import track1_submission_arrays, track1_validate_bundle

SPEC = importlib.util.spec_from_file_location("track1_adapter", MODULE / "lib/export_track1.py")
adapter = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(adapter)


def bundle(frames=3):
    names = [f"{i:06d}" for i in range(frames)]
    pr = {key: np.zeros((frames, dim), np.float32) for key, dim in MHR_PARAM_DIMS.items()}
    pr["mhr_global_rot6d"][:] = [1, 0, 0, 0, 1, 0]
    pr["pose_abs"] = np.tile(np.eye(4, dtype=np.float32), (frames, 1, 1))
    pr["pose_abs"][:, :3, 3] = [0.2, -0.4, 1.5]
    return {"schema": "cari4d.mhr_wild_inference.v1", "frames": names, "pr": pr,
            "frame_meta": [{"frame": name, "src_frame": i, "kid": 0} for i, name in enumerate(names)],
            "postopt": {"frame_indices": list(range(frames))},
            "metadata": stamp_object_pose_frame_metadata({"object_mesh_file": "/mesh/output_aligned.glb"})}


def fitted(frames=3):
    return {"pose": np.zeros((frames, 136), np.float32), "scales": np.zeros(68, np.float32),
            "shape": np.zeros(45, np.float32), "valid_input": np.ones(frames, bool),
            "per_frame_vertex_error_mm": np.full(frames, 0.05, np.float32),
            "report": json.dumps({"test_fixture": True, "frames": frames})}


def test_motion_preserves_nontrivial_rotation_translation_and_mesh_scale():
    value = bundle()
    rotation = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]], np.float32)
    value["pr"]["pose_abs"][:, :3, :3] = rotation
    motion = track1_validate_bundle(value, 3)
    arrays = track1_submission_arrays(fitted(), motion, max_vertex_error_mm=1)
    point = np.array([0.3, 0.5, 0.7])
    expected = np.array([-0.3, -0.1, 2.2])
    actual = arrays["object_scale"] * arrays["object_rotation"][0] @ point + arrays["object_translation"][0]
    np.testing.assert_allclose(actual, expected, atol=1e-6)
    assert set(arrays) == {"pose", "scales", "shape", "object_rotation", "object_translation", "object_scale"}


@pytest.mark.parametrize("change,match", [
    (lambda b: b["frames"].reverse(), "every video frame"),
    (lambda b: b["frame_meta"][1].update(src_frame=8), "Frame metadata"),
    (lambda b: b["postopt"].update(frame_indices=[1, 2, 3]), "entire episode"),
    (lambda b: b["pr"]["mhr_trans"].fill(np.nan), "must be finite"),
    (lambda b: b["pr"]["pose_abs"].__setitem__((0, 0, 0), -1), "proper"),
    (lambda b: b["pr"]["pose_abs"].__setitem__((0, 0, 0), 2), "orthonormal"),
    (lambda b: b["pr"]["pose_abs"].__setitem__((0, 3, 0), 1), "homogeneous"),
    (lambda b: b["metadata"].pop("object_pose_frame_revision"), "current object"),
])
def test_reject_invalid_bundle(change, match):
    value = bundle()
    change(value)
    with pytest.raises(ValueError, match=match):
        track1_validate_bundle(value, 3)


def test_non_aligned_mesh_is_not_silently_paired_with_training_frame_poses():
    value = bundle()
    transform = np.eye(4)
    transform[0, 3] = 2
    value["metadata"]["object_mesh_to_pose_transform"] = transform.tolist()
    value["metadata"]["object_mesh_to_training_transform"] = transform.tolist()
    with pytest.raises(ValueError, match="aligned object mesh"):
        track1_validate_bundle(value, 3)


@pytest.mark.parametrize("change,match", [
    (lambda f: f["valid_input"].__setitem__(1, False), "invalid input frames"),
    (lambda f: f["per_frame_vertex_error_mm"].__setitem__(1, 2), "exceeds"),
    (lambda f: f["per_frame_vertex_error_mm"].__setitem__(1, np.nan), "must be finite"),
    (lambda f: f.update(scales=np.zeros((3, 68))), "scales must"),
    (lambda f: f.update(shape=np.zeros((3, 45))), "shape must"),
    (lambda f: f.update(pose=np.zeros((2, 136))), "pose must"),
])
def test_reject_lossy_or_malformed_fit(change, match):
    fit = fitted()
    change(fit)
    with pytest.raises(ValueError, match=match):
        track1_submission_arrays(fit, track1_validate_bundle(bundle(), 3), max_vertex_error_mm=1)


@pytest.fixture
def export_inputs(tmp_path, monkeypatch):
    sam3d_source = tmp_path / "sam3d_source"
    sam3d_source.mkdir()
    (sam3d_source / "fixture.py").write_text("# CPU orchestration fixture\n")
    monkeypatch.setattr(adapter, "SAM3D_SOURCE_ROOT", sam3d_source)
    result = tmp_path / "result"
    (result / "inference").mkdir(parents=True)
    (result / "inference/refined.pth").write_bytes(b"synthetic adapter fixture; not a neural result")
    (result / "pipeline_report.json").write_text(json.dumps({
        "schema": "v2d.cari4d.wild_inference.v1", "sequence": "episode_000016", "verdict": "PASS"}))
    mesh = result / "export/episode_000016/object_mesh/output_aligned.glb"
    mesh.parent.mkdir(parents=True)
    trimesh.creation.box(extents=[0.2, 0.3, 0.4]).export(mesh)
    kit = tmp_path / "kit"
    (kit / "tools/track1").mkdir(parents=True)
    (kit / "tools/track1/mesh_to_mhr_params.py").write_text("# converter fixture\n")
    (kit / "v2dlb").mkdir()
    (kit / "v2dlb/mhr_metrics.py").write_text(f"MHR_TABLE3_BODY_JOINT_INDICES = {tuple(range(22))!r}\n")
    (kit / "data").mkdir()
    pd.DataFrame({"row_id": [f"t1_000016_{frame:06d}_{role}_000000"
                             for frame in range(3) for role in (0, 3, 4)]}).to_parquet(
        kit / "data/track_1_sample_submission.parquet", index=False)
    weights = tmp_path / "weights"
    model = weights / "sam3d_body/checkpoints/sam-3d-body-dinov3/assets/mhr_model.pt"
    model.parent.mkdir(parents=True)
    model.write_bytes(b"fixture")
    return dict(result_dir=str(result), submission_kit=str(kit), weights_path=str(weights),
                output_dir=str(tmp_path / "export"), episode_index=16, expected_frames=3,
                business_commit="a" * 40, image_build_commit="b" * 40,
                image_digest="sha256:" + "c" * 64)


def simulate_children(command, **kwargs):
    """Exercise orchestration with explicit substitutes for GPU-only subprocesses."""
    out = Path(command[command.index("--output") + 1])
    if command[1].endswith("export_track1_vertices.py"):
        np.save(out / "human_vertices.npy", np.zeros((3, 18439, 3), np.float32))
        np.save(out / "human_joints.npy", np.zeros((3, 127, 3), np.float32))
        motion = track1_validate_bundle(bundle(), 3)
        np.savez(out / "object_motion.npz", rotation=motion.rotation, translation=motion.translation)
        (out / "decoder.json").write_text('{"test_fixture": true}')
    elif command[1].endswith("mesh_to_mhr_params.py"):
        np.savez(out, **fitted())
    elif command[1].endswith("check_track1_acceleration.py"):
        argument = lambda key: command[command.index(key) + 1]
        with np.load(argument("--submission"), allow_pickle=False) as arrays:
            assert arrays["pose"].dtype == np.float32
            assert set(arrays.files) == {"pose", "shape", "scales", "object_rotation", "object_translation", "object_scale"}
        original = np.load(argument("--original-joints"), allow_pickle=False)
        scoring = track1_scoring(Path(argument("--submission-kit")), 16, 3)
        report = track1_added_acceleration(original, original, scoring,
                                          threshold_cm=float(argument("--threshold-cm")))
        out.write_text(json.dumps(report))
    else:
        pytest.fail(f"Unexpected subprocess: {command}")


def test_publish_and_refuse_reuse(export_inputs, monkeypatch):
    monkeypatch.setattr(adapter.subprocess, "run", simulate_children)
    out = adapter.export_track1(**export_inputs)
    assert sorted(p.name for p in out.iterdir()) == [
        "episode_000016.npz", "episode_000016_export.json", "episode_000016_object.glb"]
    report = json.loads((out / "episode_000016_export.json").read_text())
    assert report["kaggle_scored"] is False
    assert "official_converter" in report["input_sha256"]
    assert len(report["conversion"]["per_frame_mean_vertex_error_mm"]) == 3
    assert report["conversion"]["added_acceleration"]["added_acc_h_cm"] == 0
    assert report["acceptance"]["added_acceleration_check"] == "PASS"
    assert report["conversion"]["vertex_spikes"]["no_scored_spikes"] is True
    with pytest.raises(FileExistsError):
        adapter.export_track1(**export_inputs)


def test_failed_conversion_never_publishes_partial_episode(export_inputs, monkeypatch):
    def fail(command, **kwargs):
        raise subprocess.CalledProcessError(1, command)
    monkeypatch.setattr(adapter.subprocess, "run", fail)
    with pytest.raises(subprocess.CalledProcessError):
        adapter.export_track1(**export_inputs)
    assert not Path(export_inputs["output_dir"]).exists()


def test_excessive_fit_error_keeps_diagnostics_without_publishing(export_inputs, monkeypatch):
    def poor_fit(command, **kwargs):
        simulate_children(command, **kwargs)
        if command[1].endswith("mesh_to_mhr_params.py"):
            output = Path(command[command.index("--output") + 1])
            fit = fitted()
            fit["per_frame_vertex_error_mm"][:] = 4
            np.savez(output, **fit)
    monkeypatch.setattr(adapter.subprocess, "run", poor_fit)
    with pytest.raises(ValueError, match="exceeds"):
        adapter.export_track1(**export_inputs)
    destination = Path(export_inputs["output_dir"])
    assert not destination.exists()
    assert len(list(destination.parent.glob(".export-*/human_fit.npz"))) == 1
    spikes = list(destination.parent.glob(".export-*/vertex_spikes.json"))
    assert len(spikes) == 1
    assert json.loads(spikes[0].read_text())["policy"] == "report_only"


def test_report_policy_publishes_with_explicit_error_acceptance(export_inputs, monkeypatch):
    def lossy_fit(command, **kwargs):
        simulate_children(command, **kwargs)
        if command[1].endswith("mesh_to_mhr_params.py"):
            output = Path(command[command.index("--output") + 1])
            fit = fitted()
            fit["per_frame_vertex_error_mm"][:] = [0.4, 2.36, 0.5]
            np.savez(output, **fit)
    monkeypatch.setattr(adapter.subprocess, "run", lossy_fit)
    out = adapter.export_track1(**export_inputs, conversion_error_policy="report")
    report = json.loads((out / "episode_000016_export.json").read_text())
    acceptance = report["acceptance"]
    assert acceptance["accepted_for_packing"] is True
    assert acceptance["conversion_error_policy"] == "report"
    assert acceptance["frames_above_reference_tolerance"] == [1]
    assert acceptance["within_reference_tolerance"] is False
    assert acceptance["reference_tolerance_is_competition_rule"] is False
    assert report["conversion"]["worst_frame_mean_vertex_error_mm"] == pytest.approx(2.36)
    assert report["conversion"]["vertex_spikes"]["scored_spike_frames"] == [1]
    assert report["conversion"]["vertex_spikes"]["no_scored_spikes"] is False
    assert report["kaggle_scored"] is False


@pytest.mark.parametrize("change,match", [
    (lambda f: f["valid_input"].__setitem__(1, False), "invalid input frames"),
    (lambda f: f["per_frame_vertex_error_mm"].__setitem__(1, np.nan), "finite"),
    (lambda f: f["per_frame_vertex_error_mm"].__setitem__(1, -1), "nonnegative"),
    (lambda f: f["pose"].__setitem__((1, 0), np.inf), "finite"),
    (lambda f: f.update(shape=np.zeros((3, 45))), "shape must"),
])
def test_report_policy_does_not_relax_structural_checks(change, match):
    fit = fitted()
    change(fit)
    with pytest.raises(ValueError, match=match):
        track1_submission_arrays(fit, track1_validate_bundle(bundle(), 3),
                                 max_vertex_error_mm=1, conversion_error_policy="report")


def test_report_policy_revalidates_object_motion():
    motion = track1_validate_bundle(bundle(), 3)
    motion.rotation[1, 0, 0] = -1
    with pytest.raises(ValueError, match="proper"):
        track1_submission_arrays(fitted(), motion, max_vertex_error_mm=1,
                                 conversion_error_policy="report")


def test_unknown_policy_fails_before_running_models(export_inputs, monkeypatch):
    def unexpected(*args, **kwargs):
        pytest.fail("Must validate policy before running models")
    monkeypatch.setattr(adapter.subprocess, "run", unexpected)
    with pytest.raises(ValueError, match="conversion_error_policy"):
        adapter.export_track1(**export_inputs, conversion_error_policy="ignore_everything")


def test_changed_source_during_export_is_rejected(export_inputs, monkeypatch):
    def mutate(command, **kwargs):
        simulate_children(command, **kwargs)
        Path(export_inputs["result_dir"], "inference/refined.pth").write_bytes(b"changed")
    monkeypatch.setattr(adapter.subprocess, "run", mutate)
    with pytest.raises(ValueError, match="input changed"):
        adapter.export_track1(**export_inputs)
    assert not Path(export_inputs["output_dir"]).exists()


def test_added_acceleration_failure_is_not_bypassed_by_report_policy(export_inputs, monkeypatch):
    from v2d.common.track1_conversion import track1_acceleration_summary

    def jitter(command, **kwargs):
        simulate_children(command, **kwargs)
        if command[1].endswith("check_track1_acceleration.py"):
            scoring = track1_scoring(Path(export_inputs["submission_kit"]), 16, 3)
            output = Path(command[command.index("--output") + 1])
            output.write_text(json.dumps(track1_acceleration_summary([0.021], scoring, 0.02)))
    monkeypatch.setattr(adapter.subprocess, "run", jitter)
    with pytest.raises(ValueError, match="Added acceleration"):
        adapter.export_track1(**export_inputs, conversion_error_policy="report")
    destination = Path(export_inputs["output_dir"])
    assert not destination.exists()
    diagnostics = list(destination.parent.glob(".export-*/added_acceleration.json"))
    assert len(diagnostics) == 1
    assert json.loads(diagnostics[0].read_text())["added_acc_h_cm"] == pytest.approx(0.021)


def test_changed_official_scoring_source_is_rejected(export_inputs, monkeypatch):
    def mutate(command, **kwargs):
        simulate_children(command, **kwargs)
        if command[1].endswith("check_track1_acceleration.py"):
            path = Path(export_inputs["submission_kit"], "v2dlb/mhr_metrics.py")
            path.write_text(path.read_text() + "# changed source\n")
    monkeypatch.setattr(adapter.subprocess, "run", mutate)
    with pytest.raises(ValueError, match="input changed"):
        adapter.export_track1(**export_inputs)
    assert not Path(export_inputs["output_dir"]).exists()


def test_short_scoring_span_fails_before_model_work(export_inputs, monkeypatch):
    path = Path(export_inputs["submission_kit"], "data/track_1_sample_submission.parquet")
    table = pd.read_parquet(path)
    table = table.loc[~table.row_id.str.contains("_000002_")]
    table.to_parquet(path, index=False)
    def unexpected(*args, **kwargs):
        pytest.fail("Must validate scored triplets before starting models")
    monkeypatch.setattr(adapter.subprocess, "run", unexpected)
    with pytest.raises(ValueError, match="three-frame"):
        adapter.export_track1(**export_inputs)


def test_wrapper_signature_defaults_and_cli_match_library():
    library_tree = ast.parse((MODULE / "lib/export_track1.py").read_text())
    wrapper_tree = ast.parse((MODULE / "docker/run_export_track1.py").read_text())
    library = next(n for n in library_tree.body if isinstance(n, ast.FunctionDef) and n.name == "export_track1")
    wrapper = next(n for n in wrapper_tree.body if isinstance(n, ast.FunctionDef) and n.name == "run_export_track1")
    names = lambda fn: [a.arg for a in fn.args.args + fn.args.kwonlyargs]
    assert names(wrapper) == names(library) + ["dev"]
    assert [ast.dump(n) if n else None for n in wrapper.args.kw_defaults[:-1]] == [
        ast.dump(n) if n else None for n in library.args.kw_defaults]
    node = next(n for n in wrapper_tree.body if isinstance(n, ast.FunctionDef) and n.name == "_parser")
    namespace = {"argparse": argparse, "__doc__": "test"}
    exec(compile(ast.Module(body=[node], type_ignores=[]), "wrapper", "exec"), namespace)
    options = lambda parser: {a.dest: a.default for a in parser._actions if a.dest not in {"help", "dev"}}
    assert options(namespace["_parser"]()) == options(adapter._parser())
