"""CPU integration: real official packer/roster, explicitly synthetic reconstructions."""

import json
from pathlib import Path
import shutil
import subprocess
import zipfile

import numpy as np
import pandas as pd
import pytest
import trimesh

from v2d.common.artifacts import artifact_record
from v2d.common.track1_conversion import (
    Track1Scoring, track1_body_joint_indices, track1_acceleration_summary, track1_vertex_spikes,
)
from v2d.pipelines import track1_pack as packer


ROOT = Path(__file__).resolve().parents[1]
COMMIT = "a" * 40


def make_inputs(tmp_path, count=2):
    with zipfile.ZipFile(ROOT / "docs/v2d_challenge/assets/v2d_submission_kit.zip") as archive:
        archive.extractall(tmp_path / "official")
    kit = tmp_path / "official/v2d_submission_kit"
    sample_path = kit / "data/track_1_sample_submission.parquet"
    sample = pd.read_parquet(sample_path, columns=["row_id"])
    keys = sample.row_id.str.extract(r"^t1_(\d{6})_(\d{6})_\d_\d{6}$").astype(int)
    if count is not None:
        sample = sample.loc[keys[0] < count].reset_index(drop=True)
        sample.to_parquet(sample_path, index=False)
        keys = keys.loc[keys[0] < count]
    frames = keys.loc[keys[1] != 999999].groupby(0)[1].max() + 1
    count = len(frames)
    meta = tmp_path / "dataset/meta"
    meta.mkdir(parents=True)
    (meta / "info.json").write_text(json.dumps({
        "codebase_version": "v2.1", "total_episodes": count, "total_frames": int(frames.sum()),
        "fps": 30, "chunks_size": 1000, "video_path": "videos/episode_{episode_index:06d}.mp4",
        "episodes_metadata": "meta/episodes_metadata.jsonl", "features": {"rgb": {"dtype": "video"}},
    }))
    def rows(name, values):
        (meta / name).write_text("".join(json.dumps(v) + "\n" for v in values))
    rows("episodes.jsonl", [{"episode_index": i, "length": int(n), "tasks": ["synthetic test"]}
                            for i, n in frames.items()])
    rows("episodes_metadata.jsonl", [{"episode_index": i, "sequence_id": f"fixture-{i}",
        "camera": "front", "object": "box", "object_prompt": "box", "video_key": "rgb"} for i in range(count)])
    rows("tasks.jsonl", [{"task_index": 0, "task": "synthetic test"}])
    exports = tmp_path / "exports"
    joint_indices = track1_body_joint_indices(kit / "v2dlb/mhr_metrics.py")
    for i, n in frames.items():
        sequence = f"episode_{i:06d}"
        directory = exports / sequence
        directory.mkdir(parents=True)
        npz = directory / f"{sequence}.npz"
        mesh = directory / f"{sequence}_object.glb"
        np.savez(npz, pose=np.zeros((n, 136), np.float32), scales=np.zeros(68, np.float32),
                 shape=np.zeros(45, np.float32), object_rotation=np.tile(np.eye(3), (n, 1, 1)),
                 object_translation=np.tile([0.2, -0.4, 1.5], (n, 1)), object_scale=np.array(1.0))
        trimesh.creation.box(extents=[0.2, 0.3, 0.4]).export(mesh)
        scoring = Track1Scoring(joint_indices, tuple(sorted(set(keys.loc[(keys[0] == i) & (keys[1] != 999999), 1]))))
        record = {
            "schema": "v2d.cari4d.track1_export.v1", "sequence": sequence, "frames": int(n),
            "business_commit": COMMIT, "image_build_commit": "b" * 40, "image_digest": "sha256:" + "c" * 64,
            "export_source_sha256": {"fixture": "synthetic export"}, "decoder_identity": {"fixture": True},
            "inference_settings": {"postopt_num_steps": 300, "expected_frames": int(n)},
            "settings": {"conversion_error_policy": "report", "max_vertex_error_mm": 1.0,
                         "added_acc_h_reference_cm": 0.02},
            "input_sha256": {"official_converter": artifact_record(kit / "tools/track1/mesh_to_mhr_params.py")["sha256"],
                             "mhr_model": "d" * 64,
                             "official_metrics": artifact_record(kit / "v2dlb/mhr_metrics.py")["sha256"],
                             "official_sample": artifact_record(sample_path)["sha256"],
                             "acceleration_contract": artifact_record(ROOT / "reconstruction/modules/v2d_common/track1_conversion.py")["sha256"]},
            "acceptance": {"structural_checks": "PASS", "accepted_for_packing": True,
                           "conversion_error_policy": "report", "conversion_diagnostics": "RECORDED"},
            "conversion": {"per_frame_mean_vertex_error_mm": [0.4] * int(n),
                           "vertex_spikes": track1_vertex_spikes([0.4] * int(n), scoring),
                           "added_acceleration": track1_acceleration_summary([0] * len(scoring.centers()), scoring, 0.02)},
            "output_sha256": {p.name: artifact_record(p)["sha256"] for p in (npz, mesh)},
        }
        (directory / f"{sequence}_export.json").write_text(json.dumps(record))
    return dict(dataset_root=str(meta.parent), export_root=str(exports), submission_kit=str(kit),
                output_dir=str(tmp_path / "packed"),
                code_commit_url="https://github.com/example/repo/commit/" + COMMIT)


@pytest.fixture
def inputs(tmp_path):
    return make_inputs(tmp_path)


def test_full_official_roster_packs_all_30_synthetic_episodes(tmp_path):
    inputs = make_inputs(tmp_path, count=None)
    result = packer.pack_track1(**inputs)
    table = pd.read_parquet(result)
    report = json.loads((result.parent / "packing_report.json").read_text())
    assert report["episodes"] == list(range(30))
    assert report["scope"] == "full_official_roster"
    assert len(table) == report["rows"] == 740780
    assert report["kaggle_submitted"] is False
    assert report["kaggle_scored"] is False
    assert (result.parent / "pack.log").stat().st_size > 0
    row = table.loc[table.row_id == "t1_000016_000050_4_000000", ["x", "y", "z"]].to_numpy()[0]
    np.testing.assert_allclose(row, [0.2, -0.4, 1.5])


@pytest.mark.parametrize("mutation", ["missing", "wrong_frames", "changed_reference"])
def test_acceleration_report_integrity_is_rechecked_before_packing(inputs, mutation):
    path = Path(inputs["export_root"]) / "episode_000000/episode_000000_export.json"
    report = json.loads(path.read_text())
    metric = report["conversion"]["added_acceleration"]
    if mutation == "missing":
        del report["conversion"]["added_acceleration"]
    elif mutation == "wrong_frames":
        metric["scored_frames"][0] -= 1
    else:
        report["settings"]["added_acc_h_reference_cm"] = 1
    path.write_text(json.dumps(report))
    with pytest.raises(ValueError, match="acceleration"):
        packer.pack_track1(**inputs)
    assert not Path(inputs["output_dir"]).exists()


def test_packing_keeps_high_acceleration_and_scored_spikes_as_diagnostics(inputs):
    path = Path(inputs["export_root"]) / "episode_000000/episode_000000_export.json"
    report = json.loads(path.read_text())
    conversion = report["conversion"]
    metric = conversion["added_acceleration"]
    scoring = Track1Scoring(tuple(metric["joint_indices"]), tuple(metric["scored_frames"]))
    conversion["added_acceleration"] = track1_acceleration_summary([0.1] * len(scoring.centers()), scoring, 0.02)
    errors = conversion["per_frame_mean_vertex_error_mm"]
    errors[scoring.frames[0]] = 3.0
    conversion["vertex_spikes"] = track1_vertex_spikes(errors, scoring)
    assert conversion["vertex_spikes"]["scored_spike_count"] == 1
    path.write_text(json.dumps(report))
    result = packer.pack_track1(**inputs, episodes=[0])
    saved = json.loads((result.parent / "episodes/episode_000000_export.json").read_text())
    assert saved["conversion"] == conversion
    assert saved["acceptance"]["accepted_for_packing"] is True


@pytest.mark.parametrize("mutation", ["missing", "tampered"])
def test_missing_or_inconsistent_spike_report_cannot_be_packed(inputs, mutation):
    path = Path(inputs["export_root"]) / "episode_000000/episode_000000_export.json"
    report = json.loads(path.read_text())
    if mutation == "missing":
        del report["conversion"]["vertex_spikes"]
    else:
        report["conversion"]["vertex_spikes"]["scored_spike_count"] = 1
    path.write_text(json.dumps(report))
    with pytest.raises(ValueError, match="vertex-spike"):
        packer.pack_track1(**inputs)


def test_changed_official_metric_source_cannot_reuse_old_acceleration(inputs):
    metrics = Path(inputs["submission_kit"]) / "v2dlb/mhr_metrics.py"
    metrics.write_text(metrics.read_text() + "\n# changed official contract\n")
    with pytest.raises(ValueError, match="different official scoring contracts"):
        packer.pack_track1(**inputs)


@pytest.mark.parametrize("mutation,match", [
    ("missing", "episode_000001"), ("tampered", "changed after export"),
    ("unaccepted", "explicit accepted"), ("mixed_settings", "mix"),
    ("wrong_commit", "requested business commit"), ("nonfinite", "finite"),
    ("wrong_converter", "different converter"),
])
def test_invalid_collections_are_rejected_before_packer(inputs, monkeypatch, mutation, match):
    path = Path(inputs["export_root"]) / "episode_000001"
    report_path = path / "episode_000001_export.json"
    report = json.loads(report_path.read_text())
    if mutation == "missing":
        shutil.rmtree(path)
    elif mutation == "tampered":
        (path / "episode_000001_object.glb").write_bytes(b"changed")
    elif mutation == "unaccepted":
        report.pop("acceptance")
    elif mutation == "mixed_settings":
        report["inference_settings"]["postopt_num_steps"] = 1
    elif mutation == "wrong_commit":
        report["business_commit"] = "f" * 40
    elif mutation == "nonfinite":
        file = path / "episode_000001.npz"
        with np.load(file) as arrays:
            data = dict(arrays)
        data["pose"][0, 0] = np.nan  # Outside the scored interval; still must fail.
        np.savez(file, **data)
        report["output_sha256"][file.name] = artifact_record(file)["sha256"]
    elif mutation == "wrong_converter":
        (Path(inputs["submission_kit"]) / "tools/track1/mesh_to_mhr_params.py").write_text("# different fitter")
    if path.exists():
        report_path.write_text(json.dumps(report))
    def forbidden(*args, **kwargs):
        pytest.fail("Invalid collection must be rejected before invoking the packer")
    monkeypatch.setattr(packer.subprocess, "run", forbidden)
    with pytest.raises((ValueError, FileNotFoundError), match=match):
        packer.pack_track1(**inputs)
    assert not Path(inputs["output_dir"]).exists()


def test_explicit_subset_is_labelled_as_format_check(inputs):
    shutil.rmtree(Path(inputs["export_root"]) / "episode_000001")
    result = packer.pack_track1(**inputs, episodes=[0])
    report = json.loads((result.parent / "packing_report.json").read_text())
    assert report["scope"] == "subset_format_check"
    assert report["episodes"] == [0]


def test_failed_packer_does_not_publish_partial_output(inputs, monkeypatch):
    def fail(command, **kwargs):
        Path(command[command.index("--out") + 1]).write_bytes(b"partial parquet")
        raise subprocess.CalledProcessError(1, command)
    monkeypatch.setattr(packer.subprocess, "run", fail)
    with pytest.raises(subprocess.CalledProcessError):
        packer.pack_track1(**inputs)
    output = Path(inputs["output_dir"])
    assert not output.exists()
    assert list(output.parent.glob(".packed-*/submission.parquet"))
