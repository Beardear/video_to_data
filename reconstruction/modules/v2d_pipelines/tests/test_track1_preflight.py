# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""CPU-only contract tests; runnable with unittest without package installation."""

import contextlib
import importlib.util
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest


SOURCE = Path(__file__).resolve().parents[1] / "track1_preflight.py"
SPEC = importlib.util.spec_from_file_location("track1_preflight", SOURCE)
preflight = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = preflight
SPEC.loader.exec_module(preflight)


class Track1PreflightTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name).resolve()
        self.dataset = self.root / "track_1"
        self.meta = self.dataset / "meta"
        self.meta.mkdir(parents=True)
        self.info = {
            "codebase_version": "v2.1", "total_episodes": 2, "total_frames": 7,
            "fps": 30, "chunks_size": 1,
            "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
            "episodes_metadata": "meta/episodes_metadata.jsonl",
            "features": {"rgb": {"dtype": "video"}},
        }
        self.write_info()
        self.write_jsonl("episodes.jsonl", [
            {"episode_index": 0, "length": 3, "tasks": ["Lift the box"]},
            {"episode_index": 1, "length": 4, "tasks": ["Lift the box"]},
        ])
        self.metadata = [{"episode_index": i, "sequence_id": f"sequence_{i}",
                          "camera": "front", "object": "box", "object_prompt": "a box",
                          "video_key": "rgb"} for i in range(2)]
        self.write_jsonl("episodes_metadata.jsonl", self.metadata)
        self.write_jsonl("tasks.jsonl", [{"task_index": 0, "task": "Lift the box"}])
        for i in range(2):
            path = self.dataset / self.info["video_path"].format(
                episode_chunk=i, video_key="rgb", episode_index=i)
            path.parent.mkdir(parents=True)
            # Deliberately not valid MP4: presence must never imply content validation.
            path.write_bytes(b"unvalidated video")

    def write_info(self):
        (self.meta / "info.json").write_text(json.dumps(self.info))

    def write_jsonl(self, name, rows):
        (self.meta / name).write_text("".join(json.dumps(row) + "\n" for row in rows))

    def inputs(self, index=1):
        folder = self.root / "prepared"
        folder.mkdir(exist_ok=True)
        paths = {"video_path": f"prepared/episode_{index:06d}.0.color.mp4",
                 "mask_h5_path": "prepared/masks.h5", "object_mesh_path": "prepared/object.glb"}
        for path in paths.values():
            (self.root / path).write_bytes(b"not content-validated")
        manifest = self.root / "inputs.json"
        self.write_inputs(manifest, [{"episode_index": index, **paths}])
        return manifest, paths

    def write_inputs(self, path, rows):
        path.write_text(json.dumps({"schema": preflight.INPUT_SCHEMA, "episodes": rows}))

    def test_roster_uses_metadata_paths_and_reports_missing_preparation(self):
        report = preflight.track1_preflight(self.dataset)
        self.assertEqual(report["summary"]["expected_frames"], 7)
        self.assertEqual(report["summary"]["source_videos_present"], 2)
        self.assertEqual(report["summary"]["episodes_needing_preparation"], 2)
        self.assertIn("chunk-001/rgb/episode_000001.mp4", report["episodes"][1]["source_video"])
        self.assertEqual(report["episodes"][0]["files"]["mask_h5_path"]["status"], "not_configured")

    def test_explicit_paths_resolve_from_manifest_and_do_not_claim_content_validity(self):
        manifest, paths = self.inputs()
        report = preflight.track1_preflight(self.dataset, manifest, [1])
        self.assertEqual(report["summary"]["selected_episodes"], 1)
        self.assertEqual(report["summary"]["episodes_with_no_file_issues"], 1)
        self.assertEqual(report["episodes"][0]["files"]["object_mesh_path"]["path"],
                         str(self.root / paths["object_mesh_path"]))
        self.assertEqual(report["validation_level"], "metadata_and_file_presence")
        self.assertIn("mask_contents", report["not_checked"])

    def test_missing_empty_and_directory_inputs_are_distinct(self):
        manifest, paths = self.inputs()
        (self.root / paths["video_path"]).unlink()
        (self.root / paths["mask_h5_path"]).write_bytes(b"")
        mesh = self.root / paths["object_mesh_path"]
        mesh.unlink()
        mesh.mkdir()
        row = preflight.track1_preflight(self.dataset, manifest, [1])["episodes"][0]
        self.assertEqual([row["files"][role]["status"] for role in preflight.INPUT_ROLES],
                         ["missing", "empty", "not_file"])

    def test_symlink_to_original_video_does_not_satisfy_cari4d_name_contract(self):
        manifest, paths = self.inputs()
        video = self.root / paths["video_path"]
        video.unlink()
        video.symlink_to(preflight.track1_episodes(self.dataset)[1].source_video)
        report = preflight.track1_preflight(self.dataset, manifest, [1])
        self.assertTrue(any("resolved filename" in issue for issue in report["episodes"][0]["issues"]))

    def test_duplicate_metadata_rejected(self):
        self.write_jsonl("episodes_metadata.jsonl", [self.metadata[0], self.metadata[0]])
        with self.assertRaisesRegex(ValueError, "duplicate"):
            preflight.track1_preflight(self.dataset)

    def test_inconsistent_roster_rejected_even_for_selected_episode(self):
        self.write_jsonl("episodes_metadata.jsonl", [self.metadata[1]])
        with self.assertRaisesRegex(ValueError, "cover exactly"):
            preflight.track1_preflight(self.dataset, episode_indices=[1])

    def test_frame_total_and_unknown_action_rejected(self):
        self.info["total_frames"] = 8
        self.write_info()
        with self.assertRaisesRegex(ValueError, "total_frames"):
            preflight.track1_preflight(self.dataset)
        self.info["total_frames"] = 7
        self.write_info()
        self.write_jsonl("tasks.jsonl", [])
        with self.assertRaisesRegex(ValueError, "action missing"):
            preflight.track1_preflight(self.dataset)

    def test_unknown_episode_and_input_typo_rejected(self):
        with self.assertRaisesRegex(ValueError, "unknown"):
            preflight.track1_preflight(self.dataset, episode_indices=[16])
        manifest, paths = self.inputs()
        self.write_inputs(manifest, [{"episode_index": 16, **paths}])
        with self.assertRaisesRegex(ValueError, "unknown episodes"):
            preflight.track1_preflight(self.dataset, manifest)
        self.write_inputs(manifest, [{"episode_index": 1, "mesh_path": "typo.glb"}])
        with self.assertRaisesRegex(ValueError, "unknown input fields"):
            preflight.track1_preflight(self.dataset, manifest)

    def test_invalid_video_feature_rejected_with_metadata_error(self):
        self.info["features"]["rgb"] = None
        self.write_info()
        with self.assertRaisesRegex(ValueError, "unknown video feature"):
            preflight.track1_preflight(self.dataset)

    def test_escaping_and_colliding_video_paths_rejected(self):
        self.info["video_path"] = "../outside.mp4"
        self.write_info()
        with self.assertRaisesRegex(ValueError, "stay within"):
            preflight.track1_preflight(self.dataset)
        self.info["video_path"] = "videos/same.mp4"
        self.write_info()
        with self.assertRaisesRegex(ValueError, "multiple episodes"):
            preflight.track1_preflight(self.dataset)

    def test_cli_emits_json_and_distinguishes_missing_inputs_from_invalid_metadata(self):
        stdout, stderr = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            code = preflight.main(["--dataset_root", str(self.dataset)])
        self.assertEqual(code, 2)
        self.assertEqual(json.loads(stdout.getvalue())["summary"]["selected_episodes"], 2)
        manifest, _ = self.inputs()
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(preflight.main(["--dataset_root", str(self.dataset),
                                            "--inputs_manifest", str(manifest), "--episodes", "1"]), 0)
        self.info["fps"] = 0
        self.write_info()
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(stderr):
            self.assertEqual(preflight.main(["--dataset_root", str(self.dataset)]), 1)
        self.assertIn("fps", stderr.getvalue())


if __name__ == "__main__":
    unittest.main()
