# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Host wrapper for CPU-only prepared-input content checks."""

from __future__ import annotations

import argparse
from v2d.cari4d.docker._config import IMAGE_NAME, MODULES_DIR
from v2d.docker.container import run_in_container


def run_validate_inputs(
    video_path: str, mask_h5_path: str, object_mesh_path: str, report_path: str, *,
    expected_frames: int, expected_fps: float, source_video_path: str | None = None,
    mesh_scale_report_path: str | None = None, require_scale_provenance: bool = False,
    dev: bool = False,
) -> None:
    run_in_container(image=IMAGE_NAME, module="v2d.cari4d.lib.validate_inputs",
                     inputs={"video_path": video_path, "mask_h5_path": mask_h5_path,
                             "object_mesh_path": object_mesh_path, "source_video_path": source_video_path,
                             "mesh_scale_report_path": mesh_scale_report_path},
                     outputs={"report_path": report_path},
                     extra_args={"expected_frames": expected_frames, "expected_fps": expected_fps,
                                 "require_scale_provenance": require_scale_provenance},
                     dev=dev, modules_dir=MODULES_DIR, gpus=False)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("video_path", "mask_h5_path", "object_mesh_path", "report_path"):
        parser.add_argument(f"--{name}", required=True)
    parser.add_argument("--expected_frames", type=int, required=True)
    parser.add_argument("--expected_fps", type=float, required=True)
    parser.add_argument("--source_video_path")
    parser.add_argument("--mesh_scale_report_path")
    parser.add_argument("--require_scale_provenance", action="store_true")
    parser.add_argument("--dev", action="store_true")
    return parser


if __name__ == "__main__":
    run_validate_inputs(**vars(_parser().parse_args()))
