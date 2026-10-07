# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Host wrapper for the CARI4D Track 1 result adapter."""

from __future__ import annotations

import argparse

from v2d.cari4d.docker._config import IMAGE_NAME, MODULES_DIR
from v2d.docker.container import run_in_container


def run_export_track1(
    result_dir: str, submission_kit: str, weights_path: str, output_dir: str, *,
    episode_index: int, expected_frames: int, business_commit: str,
    image_build_commit: str, image_digest: str, device: str = "cuda",
    decode_batch_size: int = 16, fit_model_batch_size: int = 128,
    fit_precision: str = "float64", max_vertex_error_mm: float = 1.0,
    conversion_error_policy: str = "reject", max_added_acc_h_cm: float = 0.02, dev: bool = False,
) -> None:
    run_in_container(
        image=IMAGE_NAME, module="v2d.cari4d.lib.export_track1",
        inputs={"result_dir": result_dir, "submission_kit": submission_kit, "weights_path": weights_path},
        outputs={"output_dir": output_dir},
        extra_args={"episode_index": episode_index, "expected_frames": expected_frames,
                    "business_commit": business_commit, "image_build_commit": image_build_commit,
                    "image_digest": image_digest, "device": device,
                    "decode_batch_size": decode_batch_size, "fit_model_batch_size": fit_model_batch_size,
                    "fit_precision": fit_precision, "max_vertex_error_mm": max_vertex_error_mm,
                    "conversion_error_policy": conversion_error_policy,
                    "max_added_acc_h_cm": max_added_acc_h_cm},
        dev=dev, modules_dir=MODULES_DIR, gpus=device.startswith("cuda"),
        env={"PYTHONUNBUFFERED": "1"},
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("result_dir", "submission_kit", "weights_path", "output_dir",
                 "business_commit", "image_build_commit", "image_digest"):
        parser.add_argument(f"--{name}", required=True)
    parser.add_argument("--episode_index", type=int, required=True)
    parser.add_argument("--expected_frames", type=int, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--decode_batch_size", type=int, default=16)
    parser.add_argument("--fit_model_batch_size", type=int, default=128)
    parser.add_argument("--fit_precision", choices=("float32", "float64"), default="float64")
    parser.add_argument("--max_vertex_error_mm", type=float, default=1.0)
    parser.add_argument("--conversion_error_policy", choices=("reject", "report"), default="reject")
    parser.add_argument("--max_added_acc_h_cm", type=float, default=0.02)
    parser.add_argument("--dev", action="store_true")
    return parser


if __name__ == "__main__":
    run_export_track1(**vars(_parser().parse_args()))
