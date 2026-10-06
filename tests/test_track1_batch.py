"""Batch orchestration with explicit model-free substitutes for the GPU stages."""

import json
from pathlib import Path
import shutil
import subprocess

import pytest

from test_track1_pack import COMMIT, make_inputs
from v2d.pipelines import track1_batch as batch
from v2d.pipelines.track1_runtime import Track1Runtime


@pytest.fixture
def experiment(tmp_path, monkeypatch):
    packed = make_inputs(tmp_path)
    dataset = Path(packed["dataset_root"])
    manifest = tmp_path / "inputs.json"
    entries = []
    for index in range(2):
        sequence = f"episode_{index:06d}"
        source = dataset / "videos" / f"{sequence}.mp4"
        source.parent.mkdir(exist_ok=True)
        source.write_bytes(b"synthetic video; content validator separately tested")
        directory = tmp_path / "prepared" / sequence
        directory.mkdir(parents=True)
        paths = {"video_path": directory / f"{sequence}.0.color.mp4",
                 "mask_h5_path": directory / "masks.h5", "object_mesh_path": directory / "object.glb",
                 "mesh_scale_report_path": directory / "scale.json"}
        for path in paths.values():
            path.write_bytes(b"synthetic prepared input")
        entries.append({"episode_index": index, **{k: str(p) for k, p in paths.items()}})
    manifest.write_text(json.dumps({"schema": "v2d.track1.inputs.v1", "episodes": entries}))
    config = tmp_path / "config.json"
    shutil.copyfile(batch.MODULES / "v2d_pipelines/track1_baseline.json", config)
    weights = tmp_path / "weights"
    weights.mkdir()
    (weights / "model.pt").write_bytes(b"synthetic weights")
    args = dict(dataset_root=str(dataset), inputs_manifest=str(manifest), weights_path=str(weights),
                submission_kit=packed["submission_kit"], output_dir=str(tmp_path / "batch"),
                config_path=str(config), image_build_commit="b" * 40, image_digest="sha256:" + "c" * 64,
                runtime_mode="container")
    monkeypatch.setattr(batch, "_checkout_identity", lambda execute: {
        "repository": str(batch.REPOSITORY), "business_commit": COMMIT, "dirty": False})
    monkeypatch.setattr(Track1Runtime, "verify_packages", lambda *a: None)
    monkeypatch.setattr(Track1Runtime, "verify_cari4d_image", lambda *a: {"fixture": True})
    calls = []
    failures = {}

    def argument(invocation, key):
        return invocation.command[invocation.command.index("--" + key) + 1]

    def execute(self, invocation, timeout):
        if invocation.name == "export":
            index = int(argument(invocation, "episode_index"))
        else:
            index = int(Path(argument(invocation, "video_path")).name.split(".")[0].split("_")[1])
        calls.append((index, invocation.name))
        invocation.log_path.parent.mkdir(parents=True, exist_ok=True)
        invocation.log_path.write_text("explicit synthetic stage\n")
        failure = failures.get((index, invocation.name))
        if failure:
            failures[(index, invocation.name)] -= 1
            raise subprocess.CalledProcessError(1, invocation.command)
        sequence = f"episode_{index:06d}"
        frames = int(argument(invocation, "expected_frames"))
        if invocation.name == "validate":
            path = Path(argument(invocation, "report_path"))
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps({"schema": "v2d.cari4d.input_validation.v1", "sequence": sequence,
                "verdict": "PASS", "scale_provenance_verified": True, "video": {"decoded_frames": frames}}))
        elif invocation.name == "export":
            destination = Path(argument(invocation, "output_dir"))
            shutil.copytree(Path(packed["export_root"]) / sequence, destination)
            path = destination / f"{sequence}_export.json"
            report = json.loads(path.read_text())
            settings = batch._settings(config)
            report["settings"] = settings.export
            report["inference_settings"] = {k: v for k, v in settings.inference.items() if k != "download_models"}
            report["inference_settings"]["expected_frames"] = frames
            path.write_text(json.dumps(report))
    monkeypatch.setattr(Track1Runtime, "execute", execute)
    return args, calls, failures, entries


def test_default_only_writes_a_plan(experiment):
    args, calls, _, _ = experiment
    path = batch.run_track1_batch(**args)
    plan = json.loads(path.read_text())
    assert plan["verdict"] == "PLANNED"
    assert plan["selected_episodes"] == [0, 1]
    assert not calls
    assert not (path.parent / "batch_identity.json").exists()


def test_full_execution_requires_explicit_scope(experiment):
    args, calls, _, _ = experiment
    with pytest.raises(ValueError, match="Full-roster"):
        batch.run_track1_batch(**args, execute=True)
    assert not calls


def test_one_episode_failure_does_not_block_the_next_and_can_resume(experiment):
    args, calls, failures, _ = experiment
    failures[(0, "inference")] = 2
    path = batch.run_track1_batch(**args, execute=True, all_episodes=True)
    report = json.loads(path.read_text())
    assert report["summary"] == {"pass": 1, "fail": 1}
    assert report["episodes"]["0"]["status"] == "FAIL"
    assert report["episodes"]["1"]["status"] == "PASS"
    assert calls.count((0, "inference")) == 2
    assert (1, "export") in calls
    path = batch.run_track1_batch(**args, execute=True, episodes=[0])
    report = json.loads(path.read_text())
    assert report["verdict"] == "PASS"
    assert len(report["episodes"]["0"]["attempts"]) == 3
    assert report["episodes"]["1"]["status"] == "PASS"
    assert len(list(path.parent.glob("logs/episode_000000/attempt-*/inference.log"))) == 3
    assert report["cumulative_summary"]["pass"] == 2
    assert report["cumulative_summary"]["not_passed"] == 0


def test_transient_failure_retries_and_completed_exports_are_reused(experiment):
    args, calls, failures, _ = experiment
    failures[(0, "export")] = 1
    path = batch.run_track1_batch(**args, execute=True, all_episodes=True)
    assert json.loads(path.read_text())["verdict"] == "PASS"
    assert calls.count((0, "export")) == 2
    completed_calls = list(calls)
    batch.run_track1_batch(**args, execute=True, all_episodes=True)
    assert calls == completed_calls
    report = json.loads(path.read_text())
    assert all(row["attempts"][-1]["reused"] for row in report["episodes"].values())


def test_bad_inputs_are_not_retried_and_never_reach_gpu(experiment):
    args, calls, failures, _ = experiment
    failures[(0, "validate")] = 10
    path = batch.run_track1_batch(**args, execute=True, all_episodes=True)
    assert calls.count((0, "validate")) == 1
    assert (0, "inference") not in calls
    assert (1, "export") in calls
    assert json.loads(path.read_text())["verdict"] == "FAIL"


@pytest.mark.parametrize("change", ["input", "weights", "config"])
def test_changed_experiment_cannot_reuse_old_results(experiment, change):
    args, calls, _, entries = experiment
    batch.run_track1_batch(**args, execute=True, episodes=[0])
    count = len(calls)
    if change == "input":
        Path(entries[0]["mask_h5_path"]).write_bytes(b"different mask bytes")
    elif change == "weights":
        (Path(args["weights_path"]) / "model.pt").write_bytes(b"different model")
    else:
        path = Path(args["config_path"])
        config = json.loads(path.read_text())
        config["inference"]["postopt_num_steps"] = 10
        path.write_text(json.dumps(config))
    with pytest.raises(ValueError, match="new output_dir"):
        batch.run_track1_batch(**args, execute=True, episodes=[0])
    assert len(calls) == count


def test_cancellation_records_interruption_and_does_not_start_next_episode(experiment, monkeypatch):
    args, _, _, _ = experiment
    seen = []
    def interrupt(self, invocation, timeout):
        seen.append(invocation.name)
        raise KeyboardInterrupt()
    monkeypatch.setattr(Track1Runtime, "execute", interrupt)
    with pytest.raises(KeyboardInterrupt):
        batch.run_track1_batch(**args, execute=True, all_episodes=True)
    report = json.loads((Path(args["output_dir"]) / "batch_report.json").read_text())
    assert report["verdict"] == "INTERRUPTED"
    assert report["episodes"]["0"]["status"] == "INTERRUPTED"
    assert "1" not in report["episodes"]
    assert seen == ["validate"]


def test_foreign_export_settings_are_not_silently_reused(experiment):
    args, calls, _, _ = experiment
    batch.run_track1_batch(**args, execute=True, episodes=[0])
    path = Path(args["output_dir"]) / "exports/episode_000000/episode_000000_export.json"
    record = json.loads(path.read_text())
    record["inference_settings"]["postopt_num_steps"] = 1
    path.write_text(json.dumps(record))
    before = len(calls)
    result = batch.run_track1_batch(**args, execute=True, episodes=[0])
    assert json.loads(result.read_text())["verdict"] == "FAIL"
    assert len(calls) == before


def test_unknown_config_parameter_is_rejected_before_execution(experiment):
    args, calls, _, _ = experiment
    path = Path(args["config_path"])
    record = json.loads(path.read_text())
    record["inference"]["output_dir"] = "/wrong/place"
    path.write_text(json.dumps(record))
    with pytest.raises(ValueError, match="reserved"):
        batch.run_track1_batch(**args, execute=True, episodes=[0])
    assert not calls
