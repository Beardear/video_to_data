"""CPU checks using real encoded video, HDF5 masks, and a scene mesh."""

import ast
import importlib.util
import json
from pathlib import Path
import shutil
import sys

import av
import h5py
import numpy as np
import pytest
import trimesh


MODULE = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("cari4d_validate_inputs", MODULE / "lib/validate_inputs.py")
validator = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = validator
spec.loader.exec_module(validator)


@pytest.fixture
def inputs(tmp_path):
    video = tmp_path / "episode_000016.0.color.mp4"
    with av.open(str(video), "w") as container:
        stream = container.add_stream("mpeg4", rate=30)
        stream.width, stream.height, stream.pix_fmt = 32, 24, "yuv420p"
        for index in range(4):
            frame = av.VideoFrame.from_ndarray(np.full((24, 32, 3), index * 50, np.uint8), format="rgb24")
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
    source = tmp_path / "episode_000016.mp4"
    shutil.copyfile(video, source)
    masks = tmp_path / "masks.h5"
    with h5py.File(masks, "w") as handle:
        group = handle.create_group("episode_000016")
        for index in range(4):
            for ending in ("person_mask.png", "obj_rend_mask.png"):
                mask = np.zeros((24, 32), np.uint8)
                mask[2:8, 4:10] = 255
                group.create_dataset(f"{index:06d}-k0.{ending}", data=mask)
    mesh = tmp_path / "object.glb"
    scene = trimesh.Scene()
    transform = np.eye(4)
    transform[0, 3] = 1
    scene.add_geometry(trimesh.creation.box(extents=[0.2, 0.3, 0.4]), transform=transform)
    scene.export(mesh)
    scale_path = tmp_path / "scale.json"
    scale_path.write_text(json.dumps({
        "schema": "v2d.track1.mesh_scale.v1", "units": "metres", "method": "depth_alignment",
        "reference_frame": 0, "applied_scale": [0.1, 0.1, 0.1],
        "mesh_sha256": validator.artifact_record(mesh)["sha256"],
        "source_video_sha256": validator.artifact_record(video)["sha256"],
    }))
    return dict(video_path=str(video), mask_h5_path=str(masks), object_mesh_path=str(mesh),
                report_path=str(tmp_path / "validation.json"), source_video_path=str(source),
                mesh_scale_report_path=str(scale_path), require_scale_provenance=True,
                expected_frames=4, expected_fps=30)


def report(inputs):
    return json.loads(validator.validate_inputs(**inputs).read_text())


def errors(value):
    return {item["code"] for item in value["issues"] if item["severity"] == "error"}


def test_valid_contents_are_hashed_and_decoded(inputs):
    value = report(inputs)
    assert value["verdict"] == "PASS"
    assert value["video"]["decoded_frames"] == 4
    assert value["scale_provenance_verified"]
    np.testing.assert_allclose(value["mesh"]["extents_m"], [0.2, 0.3, 0.4], atol=1e-6)
    assert value["inputs"]["video"]["sha256"] == value["inputs"]["source_video"]["sha256"]
    assert "segmentation_semantic_accuracy" in value["not_checked"]


@pytest.mark.parametrize("field,value,code", [
    ("expected_frames", 5, "video_frame_count"),
    ("expected_fps", 25, "video_fps"),
])
def test_metadata_disagrees_with_decoded_video(inputs, field, value, code):
    inputs[field] = value
    assert code in errors(report(inputs))


@pytest.mark.parametrize("mutation,code", [
    ("missing", "mask_timeline"), ("extra", "mask_timeline"),
    ("dimensions", "mask_dimensions"), ("encoding", "mask_encoding"),
    ("empty_human", "empty_human_masks"), ("empty_object_zero", "object_initialization"),
    ("full", "full_frame_mask"), ("metadata", "mask_metadata"), ("sequence", "mask_sequence"),
])
def test_malformed_masks_fail_before_inference(inputs, mutation, code):
    with h5py.File(inputs["mask_h5_path"], "a") as handle:
        group = handle["episode_000016"]
        key = "000000-k0.person_mask.png"
        if mutation == "missing":
            del group[key]
        elif mutation == "extra":
            group.create_dataset("000004-k0.person_mask.png", data=np.zeros((24, 32), np.uint8))
        elif mutation == "dimensions":
            del group[key]
            group.create_dataset(key, data=np.zeros((12, 16), np.uint8))
        elif mutation == "encoding":
            group[key][2:4, 2:4] = 128
        elif mutation == "empty_human":
            group[key][:] = 0
        elif mutation == "empty_object_zero":
            group["000000-k0.obj_rend_mask.png"][:] = 0
        elif mutation == "full":
            group[key][:] = 255
        elif mutation == "metadata":
            handle.attrs["frame_count"] = 99
        elif mutation == "sequence":
            handle.move("episode_000016", "episode_000017")
    assert code in errors(report(inputs))


def test_occlusion_is_visible_without_rejecting_episode(inputs):
    with h5py.File(inputs["mask_h5_path"], "a") as handle:
        handle["episode_000016/000002-k0.obj_rend_mask.png"][:] = 0
    value = report(inputs)
    assert value["verdict"] == "PASS"
    assert value["empty_mask_frames"]["object"] == [2]
    assert "object_occlusion" in {i["code"] for i in value["issues"]}


@pytest.mark.parametrize("field,value", [
    ("mesh_sha256", "wrong-mesh"), ("source_video_sha256", "wrong-video"),
    ("applied_scale", [1, -1, 1]), ("reference_frame", 4), ("units", "centimetres"),
])
def test_scale_record_is_bound_to_inputs(inputs, field, value):
    path = Path(inputs["mesh_scale_report_path"])
    scale = json.loads(path.read_text())
    scale[field] = value
    path.write_text(json.dumps(scale))
    assert "mesh_scale_provenance" in errors(report(inputs))


def test_missing_scale_record_is_explicit(inputs):
    inputs["mesh_scale_report_path"] = None
    assert "mesh_scale_provenance" in errors(report(inputs))
    inputs["require_scale_provenance"] = False
    value = report(inputs)
    assert value["verdict"] == "PASS"
    assert not value["scale_provenance_verified"]


def test_report_cannot_overwrite_source(inputs):
    inputs["report_path"] = inputs["source_video_path"]
    with pytest.raises(ValueError, match="overwrite"):
        report(inputs)


def test_wrong_source_video_is_rejected(inputs):
    Path(inputs["source_video_path"]).write_bytes(b"other episode")
    assert "source_video_mismatch" in errors(report(inputs))


def test_inputs_changing_during_validation_are_rejected(inputs, monkeypatch):
    original = validator.artifact_record
    count = 0

    def changing(path):
        nonlocal count
        count += 1
        value = original(path)
        if count > 5:
            value["sha256"] = "changed"
        return value

    monkeypatch.setattr(validator, "artifact_record", changing)
    assert "input_changed" in errors(report(inputs))


def test_wrapper_exposes_every_parameter():
    lib = ast.parse((MODULE / "lib/validate_inputs.py").read_text())
    wrapper = ast.parse((MODULE / "docker/run_validate_inputs.py").read_text())

    def arguments(tree, name):
        function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name)
        return {a.arg for a in function.args.args + function.args.kwonlyargs}

    assert arguments(wrapper, "run_validate_inputs") == arguments(lib, "validate_inputs") | {"dev"}
