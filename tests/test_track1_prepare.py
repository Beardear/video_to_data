"""Real CPU file pipeline; SAM2/MoGe/SAM3D/rendered alignment are explicit substitutes."""

import json
from pathlib import Path
import subprocess
import sys

import av
import numpy as np
from PIL import Image
import pytest
import trimesh

from v2d.common.datatypes import CameraIntrinsics, Transform3d
from v2d.common.video import FrameSource, FrameWriter
from v2d.pipelines import track1_prepare as preparation
from v2d.pipelines.track1_prompts import validated_prompts
from v2d.pipelines.track1_runtime import Track1Runtime
from v2d.pipelines.track1_stages import PreparationStages


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


@pytest.fixture
def experiment(tmp_path, monkeypatch):
    dataset = tmp_path / "dataset"
    meta = dataset / "meta"
    write_json(meta / "info.json", {
        "codebase_version": "v2.1", "total_episodes": 2, "total_frames": 8, "fps": 30,
        "chunks_size": 1000, "video_path": "videos/episode_{episode_index:06d}.mp4",
        "episodes_metadata": "meta/episodes_metadata.jsonl", "features": {"rgb": {"dtype": "video"}}})
    for name, rows in {
        "episodes": [{"episode_index": i, "length": 4, "tasks": ["Lift the box"]} for i in range(2)],
        "episodes_metadata": [{"episode_index": i, "sequence_id": f"sequence_{i}", "camera": "front",
                               "object": "box", "object_prompt": "a box", "video_key": "rgb"} for i in range(2)],
        "tasks": [{"task_index": 0, "task": "Lift the box"}],
    }.items():
        (meta / f"{name}.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
    prompts = []
    for index in range(2):
        video = dataset / "videos" / f"episode_{index:06d}.mp4"
        video.parent.mkdir(exist_ok=True)
        with av.open(str(video), "w") as container:
            stream = container.add_stream("mpeg4", rate=30)
            stream.width, stream.height, stream.pix_fmt = 32, 24, "yuv420p"
            for frame_index in range(4):
                frame = av.VideoFrame.from_ndarray(np.full((24, 32, 3), 30 + frame_index * 40, np.uint8), format="rgb24")
                for packet in stream.encode(frame):
                    container.mux(packet)
            for packet in stream.encode():
                container.mux(packet)
        path = tmp_path / f"prompts_{index}.json"
        write_json(path, {"prompts": [
            {"frame_index": 0, "object_id": 0, "role": "human", "box": {"x0": 2, "y0": 2, "x1": 16, "y1": 22}},
            {"frame_index": 0, "object_id": 1, "role": "object", "box": {"x0": 18, "y0": 4, "x1": 28, "y1": 12}}]})
        prompts.append(path)
    manifest = tmp_path / "preparation.json"
    write_json(manifest, {"schema": "v2d.track1.preparation.v1", "episodes": [
        {"episode_index": i, "prompts_path": p.name, "reference_frame": 1} for i, p in enumerate(prompts)]})
    weights = tmp_path / "weights"
    weights.mkdir()
    (weights / "model.pt").write_bytes(b"explicit synthetic weights")
    args = dict(dataset_root=str(dataset), preparation_manifest=str(manifest), output_dir=str(tmp_path / "prepared"),
                sam2_weights=str(weights), moge_weights=str(weights / "model.pt"), sam3d_weights=str(weights),
                image_build_commit="b" * 40, image_digest="sha256:" + "c" * 64,
                runtime_python=sys.executable, sam2_python=sys.executable, sam3d_python=sys.executable)
    monkeypatch.setattr(preparation, "checkout_identity", lambda *a: {
        "repository": str(preparation.REPOSITORY), "business_commit": "a" * 40, "dirty": False})
    monkeypatch.setattr(Track1Runtime, "verify_packages", lambda *a: None)
    monkeypatch.setattr(Track1Runtime, "package_versions", lambda *a: {"fixture": "CPU integration"})
    calls, failures = [], {}
    real_execute = Track1Runtime.execute

    def execute(self, invocation, timeout):
        directory = invocation.log_path.parent.parent
        index = int(directory.name.split("_")[-1])
        calls.append((index, invocation.name))
        invocation.log_path.parent.mkdir(parents=True, exist_ok=True)
        invocation.log_path.write_text("Explicit CPU test substitute for GPU model stage\n")
        if failures.get((index, invocation.name), 0):
            failures[(index, invocation.name)] -= 1
            raise subprocess.CalledProcessError(1, invocation.command)
        def argument(key):
            return Path(invocation.command[invocation.command.index("--" + key) + 1])
        camera = CameraIntrinsics(fx=30, fy=30, cx=16, cy=12, width=32, height=24)
        pose = Transform3d(rotation=[1, 0, 0, 0], translation=[0, 0, 2], scale=[1, 1, 1])
        if invocation.name == "02_sam2":
            assert invocation.environment["HF_HUB_OFFLINE"] == "1"
            assert str(argument("mask_extension")) == ".h5"
            for role in range(2):
                mask = np.zeros((24, 32), np.uint8)
                mask[2:22, 2:16] = 255 if role == 0 else 0
                if role == 1:
                    mask[4:12, 18:28] = 255
                with FrameWriter.from_path(argument("masks_dir") / f"{role}.h5") as writer:
                    for frame_index in range(4):
                        writer.write_frame(mask, stem=f"{frame_index:06d}")
        elif invocation.name == "04_depth":
            Image.fromarray(np.full((24, 32), 2000, np.uint16)).save(argument("depth_path"))
            camera.save(str(argument("intrinsics_path")))
        elif invocation.name == "05_sam3d":
            trimesh.creation.box().export(argument("mesh_path"))
            pose.save(str(argument("transform_path")))
            camera.save(str(argument("intrinsics_path")))
        elif invocation.name == "07_align_scale":
            Transform3d([1, 0, 0, 0], [0, 0, 0], [0.1] * 3).save(str(argument("output_transform")))
        else:
            # Real CPU mesh simplify/transform, CARI4D mask packing and content validation CLIs.
            real_execute(self, invocation, timeout)
    monkeypatch.setattr(Track1Runtime, "execute", execute)
    return args, calls, failures, prompts


def test_default_plan_checks_prompts_without_starting_models(experiment):
    args, calls, _, _ = experiment
    path = preparation.prepare_track1(**args)
    value = json.loads(path.read_text())
    assert value["verdict"] == "PLANNED"
    assert all(row["issue"] is None for row in value["episodes"])
    assert not calls
    assert not (path.parent / "preparation_identity.json").exists()
    with pytest.raises(ValueError, match="Full-roster"):
        preparation.prepare_track1(**args, execute=True)


def test_real_cpu_artifacts_pass_validation_and_completed_stages_resume(experiment):
    args, calls, _, _ = experiment
    path = preparation.prepare_track1(**args, execute=True, all_episodes=True)
    report = json.loads(path.read_text())
    assert report["verdict"] == "PASS", report
    assert report["summary"] == {"pass": 2, "not_prepared": 0}
    entry = json.loads((path.parent / "inputs.json").read_text())["episodes"][0]
    scale = json.loads(Path(entry["mesh_scale_report_path"]).read_text())
    assert scale["reference_frame"] == 1
    assert scale["applied_scale"] == [0.1] * 3
    assert len(scale["evidence"]) == 8
    assert scale["physical_scale_ground_truth_checked"] is False
    with FrameSource.from_path(path.parent / "episode_000000/sam2/0.h5") as human:
        assert human.stems == [f"{i:06d}" for i in range(4)]
    assert Image.open(path.parent / "episode_000000/mask_review.png").size == (1280, 788)
    assert report["episodes"]["0"]["semantic_review"] == "required"
    before = list(calls)
    preparation.prepare_track1(**args, execute=True, episodes=[0])
    assert calls == before
    row = json.loads(path.read_text())["episodes"]["0"]
    assert all(stage["reused"] for stage in row["stages"])


def test_failed_episode_does_not_block_others_and_resumes_from_failed_stage(experiment):
    args, calls, failures, _ = experiment
    failures[(0, "05_sam3d")] = 1
    path = preparation.prepare_track1(**args, execute=True, all_episodes=True)
    report = json.loads(path.read_text())
    assert report["episodes"]["0"]["status"] == "FAIL"
    assert report["episodes"]["1"]["status"] == "PASS", report
    assert [r["episode_index"] for r in json.loads((path.parent / "inputs.json").read_text())["episodes"]] == [1]
    preparation.prepare_track1(**args, execute=True, episodes=[0])
    assert json.loads(path.read_text())["summary"]["pass"] == 2
    assert calls.count((0, "02_sam2")) == 1
    assert calls.count((0, "05_sam3d")) == 2


def test_mask_prompt_identity_is_independent_of_selected_subset(experiment):
    args, _, _, prompts = experiment
    value = json.loads(prompts[1].read_text())
    value["prompts"][1].pop("box")
    value["prompts"][1]["mask_path"] = "prompt_mask.png"
    mask = np.zeros((24, 32), np.uint8)
    mask[4:12, 18:28] = 255
    Image.fromarray(mask).save(prompts[1].parent / "prompt_mask.png")
    write_json(prompts[1], value)
    preparation.prepare_track1(**args, execute=True, episodes=[0])
    path = preparation.prepare_track1(**args, execute=True, episodes=[1])
    assert json.loads(path.read_text())["summary"]["pass"] == 2
    mask[4, 18] = 0
    Image.fromarray(mask).save(prompts[1].parent / "prompt_mask.png")
    with pytest.raises(ValueError, match="new output_dir"):
        preparation.prepare_track1(**args, execute=True, episodes=[0])


def test_corrupted_unselected_result_is_removed_from_the_manifest(experiment):
    args, _, _, _ = experiment
    path = preparation.prepare_track1(**args, execute=True, all_episodes=True)
    (path.parent / "episode_000001/object_scaled.glb").write_bytes(b"corrupted")
    preparation.prepare_track1(**args, execute=True, episodes=[0])
    report = json.loads(path.read_text())
    assert report["episodes"]["1"]["status"] == "FAIL"
    assert [r["episode_index"] for r in json.loads((path.parent / "inputs.json").read_text())["episodes"]] == [0]


def test_cancellation_retains_logs_but_clears_the_consumable_manifest(experiment, monkeypatch):
    args, calls, _, _ = experiment
    def interrupt(*args):
        raise KeyboardInterrupt()
    monkeypatch.setattr(Track1Runtime, "execute", interrupt)
    with pytest.raises(KeyboardInterrupt):
        preparation.prepare_track1(**args, execute=True, all_episodes=True)
    output = Path(args["output_dir"])
    assert json.loads((output / "preparation_report.json").read_text())["verdict"] == "INTERRUPTED"
    assert json.loads((output / "inputs.json").read_text())["episodes"] == []
    assert not calls


@pytest.mark.parametrize("change", ["frame", "box", "role", "points"])
def test_bad_prompts_fail_before_model_execution(experiment, change):
    args, calls, _, prompts = experiment
    value = json.loads(prompts[0].read_text())
    prompt = value["prompts"][0]
    if change == "frame":
        prompt["frame_index"] = 4
    elif change == "box":
        prompt["box"]["x1"] = 33
    elif change == "role":
        prompt["role"] = "object"
    else:
        prompt.pop("box")
        prompt.update(points=[{"x": 10, "y": 10}], point_labels=[0])
    write_json(prompts[0], value)
    path = preparation.prepare_track1(**args, execute=True, episodes=[0])
    assert json.loads(path.read_text())["verdict"] == "FAIL"
    assert not calls


def test_prompts_changed_during_validation_are_not_used(experiment, monkeypatch):
    args, calls, _, prompts = experiment
    real = validated_prompts
    def changed(request, episode, image_size):
        result = real(request, episode, image_size)
        request.prompts_path.write_text(request.prompts_path.read_text() + "\n")
        return result
    monkeypatch.setattr(preparation, "validated_prompts", changed)
    with pytest.raises(ValueError, match="changed while planning"):
        preparation.prepare_track1(**args, execute=True, episodes=[0])
    assert not calls


def test_stages_recover_partial_outputs_and_reject_changed_successes(tmp_path):
    identity, source, output = (tmp_path / name for name in ("identity.json", "source", "output"))
    identity.write_text("{}")
    source.write_text("original")
    stages = PreparationStages(tmp_path, identity)
    def fail():
        output.write_text("partial")
        raise RuntimeError("interrupted model")
    with pytest.raises(RuntimeError):
        stages.run("stage", (source,), (output,), fail)
    def finish():
        assert not output.exists()
        output.write_text("complete")
    assert not stages.run("stage", (source,), (output,), finish)["reused"]
    assert stages.run("stage", (source,), (output,), fail)["reused"]
    output.write_text("modified")
    with pytest.raises(ValueError, match="identity changed"):
        stages.run("stage", (source,), (output,), finish)
    with pytest.raises(ValueError, match="overwrite their inputs"):
        stages.run("bad", (source,), (source,), fail)
    with pytest.raises(ValueError, match="inside their episode"):
        stages.run("outside", (source,), (tmp_path.parent / "outside",), fail)
