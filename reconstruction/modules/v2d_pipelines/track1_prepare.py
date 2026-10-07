"""Prepare Track 1 masks and static object meshes inside the existing GPU image.

Planning is the default. Execution composes existing model CLIs in their own
Python environments; it does not run the temporal CARI4D reconstruction.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import json
import math
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time

from v2d.common.artifacts import artifact_lock, artifact_record, atomic_json
from v2d.common.datatypes import Sam2Prompts, Transform3d
from v2d.pipelines.track1_experiment import checkout_identity, weight_identity
from v2d.pipelines.track1_preflight import track1_episodes, track1_metadata_identity, Track1Episode
from v2d.pipelines.track1_prompts import PreparationRequest, preparation_requests, validated_prompts
from v2d.pipelines.track1_runtime import Track1Runtime
from v2d.pipelines.track1_stages import PreparationStages


REPOSITORY = Path(__file__).resolve().parents[3]
MODULES = REPOSITORY / "reconstruction/modules"


def _input_identity(metadata: dict[int, Track1Episode], requests: dict[int, PreparationRequest]) -> dict:
    """Bind the full roster, including mask prompt files outside the selected subset."""
    paths = {f"video_{i}": e.source_video for i, e in metadata.items()}
    records = {}
    for index, request in requests.items():
        path = request.prompts_path
        record = artifact_record(path) if path.is_file() else {"path": str(path), "missing": True}
        records[f"prompts_{index}"] = record
        if not path.is_file():
            continue
        try:
            payload = json.loads(path.read_text())
            rows = payload.get("prompts", []) if isinstance(payload, dict) else []
            for number, row in enumerate(rows if isinstance(rows, list) else []):
                mask = row.get("mask_path") if isinstance(row, dict) else None
                if isinstance(mask, str) and mask:
                    paths[f"prompt_mask_{index}_{number}"] = (path.parent / mask).resolve()
        except json.JSONDecodeError:
            pass  # The per-episode prompt validator reports malformed JSON.
        if artifact_record(path) != record:
            raise ValueError("Prompt file changed while being read")
    records.update({name: artifact_record(path) if path.is_file() else {"path": str(path), "missing": True}
                    for name, path in paths.items()})
    return records


def _verify_prepared_rows(rows: dict, metadata: dict[int, Track1Episode], root: Path) -> None:
    """Remove stale successes from manifests without deleting their diagnostic artifacts."""
    roles = {"video_path", "mask_h5_path", "object_mesh_path", "mesh_scale_report_path"}
    for key, row in rows.items():
        if row.get("status") != "PASS":
            continue
        try:
            index = row["episode_index"]
            if type(index) is not int or index not in metadata or key != str(index):
                raise ValueError("Unknown prepared episode")
            if set(row["inputs"]) != roles or set(row["artifacts"]) != roles | {"validation", "review"}:
                raise ValueError("Incomplete preparation record")
            directory = root / f"episode_{index:06d}"
            for name, record in row["artifacts"].items():
                path = Path(record["path"]).resolve()
                if not path.is_relative_to(directory) or artifact_record(path) != record:
                    raise ValueError(f"Prepared artifact changed: {name}")
                if name in roles and str(path) != row["inputs"][name]:
                    raise ValueError(f"Prepared input path disagrees with its artifact: {name}")
        except (OSError, ValueError, KeyError, TypeError) as error:
            row.update(status="FAIL", error_type=type(error).__name__, error=str(error))


def _reference(video: Path, masks: Path, index: int, image_path: Path, mask_path: Path) -> None:
    import numpy as np
    from PIL import Image
    from v2d.common.video import FrameSource

    with FrameSource.from_path(video, frames_slice=slice(index, index + 1)) as frames:
        image = next(frames.iter_frames())
    with FrameSource.from_path(masks) as frames:
        if frames.stems[index] != f"{index:06d}":
            raise ValueError("Reference mask does not match its source-video frame")
        mask = frames[index]
    if mask.shape != image.shape[:2] or not np.any(mask):
        raise ValueError("Reference object mask must be nonempty and match the image")
    Image.fromarray(image).save(image_path)
    Image.fromarray(mask).save(mask_path)


def _scale_record(source_video: Path, mesh: Path, transform: Path, reference_frame: int,
                  evidence: tuple[Path, ...], output: Path) -> None:
    import numpy as np

    value = Transform3d.load(str(transform))
    if (len(value.scale) != 3 or not np.isfinite(value.scale).all() or min(value.scale) <= 0
            or not np.allclose(value.rotation, [1, 0, 0, 0], rtol=0, atol=1e-8)
            or not np.allclose(value.translation, [0, 0, 0], rtol=0, atol=1e-8)):
        raise ValueError("Depth alignment must produce a finite positive scale-only transform")
    atomic_json(output, {"schema": "v2d.track1.mesh_scale.v1", "units": "metres",
        "method": "depth_alignment", "reference_frame": reference_frame, "applied_scale": value.scale,
        "mesh_sha256": artifact_record(mesh)["sha256"],
        "source_video_sha256": artifact_record(source_video)["sha256"],
        "evidence": [artifact_record(path) for path in (transform, *evidence)],
        "physical_scale_ground_truth_checked": False})


def _mask_review(video: Path, human: Path, obj: Path, frames: int, reference: int, output: Path) -> None:
    import numpy as np
    from PIL import Image, ImageDraw
    from v2d.common.video import FrameSource

    selected = sorted({0, reference, frames // 4, frames // 2, 3 * frames // 4, frames - 1})
    thumbnails = []
    with FrameSource.from_path(video) as rgb, FrameSource.from_path(human) as humans, FrameSource.from_path(obj) as objects:
        for index, image in enumerate(rgb.iter_frames()):
            if index not in selected:
                continue
            pixels = image.astype(np.float32)
            for mask, color in ((humans[index] > 0, [255, 160, 30]), (objects[index] > 0, [40, 180, 255])):
                pixels[mask] = 0.6 * pixels[mask] + 0.4 * np.asarray(color)
            tile = Image.fromarray(pixels.astype(np.uint8))
            tile.thumbnail((640, 360))
            panel = Image.new("RGB", (640, 394), "#202428")
            panel.paste(tile, ((640 - tile.width) // 2, 34))
            ImageDraw.Draw(panel).text((10, 10), f"frame {index} | human: orange | object: blue", fill="white")
            thumbnails.append(panel)
    if len(thumbnails) != len(selected):
        raise ValueError("Cannot render all requested mask review frames")
    canvas = Image.new("RGB", (1280, 394 * ((len(thumbnails) + 1) // 2)), "#202428")
    for index, tile in enumerate(thumbnails):
        canvas.paste(tile, ((index % 2) * 640, (index // 2) * 394))
    canvas.save(output)


def _prepare_episode(episode: Track1Episode, request: PreparationRequest, prompts: Sam2Prompts,
                     root: Path, identity_path: Path, runtimes: dict[str, Track1Runtime],
                     weights: dict[str, Path], face_count: int, seed: int, timeout: float) -> dict:
    sequence = f"episode_{episode.episode_index:06d}"
    directory = root / sequence
    stages = PreparationStages(directory, identity_path)
    reports = []
    video = directory / f"{sequence}.0.color.mp4"
    normalized_prompts = directory / "prompts.json"
    human = directory / "sam2" / f"{request.human_id}.h5"
    obj = directory / "sam2" / f"{request.object_id}.h5"
    rgb, mask = directory / "reference.png", directory / "reference_object.png"
    depth, intrinsics = directory / "depth.png", directory / "intrinsics.json"
    mesh, pose, sam_intrinsics = directory / "object_raw.glb", directory / "sam3d_pose.json", directory / "sam3d_intrinsics.json"
    simplified, scale_transform = directory / "object_simplified.glb", directory / "scale_transform.json"
    scaled, scale_record = directory / "object_scaled.glb", directory / "mesh_scale.json"
    packed = directory / f"{sequence}_masks_k0.h5"
    validation, review = directory / "validation.json", directory / "mask_review.png"

    def local(name, inputs, outputs, operation):
        reports.append(stages.run(name, tuple(inputs), tuple(outputs), operation))

    def module(name, package, lib, wrapper, arguments, inputs, outputs):
        runtime = runtimes.get(package, runtimes["base"])
        attempt = len(list((directory / "logs").glob(f"{name}-*.log"))) + 1
        invocation = runtime.invocation(name, package, lib, wrapper, arguments,
                                        directory / "logs" / f"{name}-{attempt:03d}.log")
        environment = {"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "PYOPENGL_PLATFORM": "egl"}
        if package == "v2d_sam2":
            environment.update(CONFIG_FILE="configs/sam2.1/sam2.1_hiera_l.yaml", CHECKPOINT_FILE="sam2.1_hiera_large.pt")
        if package == "v2d_sam3d":
            environment.update(HF_HOME=str(weights["sam3d"] / "hf_home"),
                               TORCH_HOME=str(weights["sam3d"] / "torch_home"))
        invocation = replace(invocation, environment=environment)
        local(name, inputs, outputs, lambda: runtime.execute(invocation, timeout))

    def copy_inputs():
        shutil.copyfile(episode.source_video, video)
        atomic_json(normalized_prompts, prompts.to_dict())
    mask_prompts = [Path(p.mask_path) for p in prompts.prompts if p.mask_path]
    local("01_inputs", [episode.source_video, request.prompts_path, *mask_prompts],
          [video, normalized_prompts], copy_inputs)
    module("02_sam2", "v2d_sam2", "video_to_masks.py", "run_video_to_masks.py",
           {"video_path": video, "prompts_path": normalized_prompts, "masks_dir": directory / "sam2",
            "weights_dir": weights["sam2"], "mask_extension": ".h5"},
           [video, normalized_prompts, *mask_prompts], [human, obj])
    local("03_reference", [video, obj], [rgb, mask], lambda: _reference(video, obj, request.reference_frame, rgb, mask))
    module("04_depth", "v2d_moge", "image_to_depth.py", "run_image_to_depth.py",
           {"image_path": rgb, "depth_path": depth, "intrinsics_path": intrinsics, "weights_path": weights["moge"]},
           [rgb], [depth, intrinsics])
    module("05_sam3d", "v2d_sam3d", "image_to_mesh.py", "run_image_to_mesh.py",
           {"image_path": rgb, "mask_path": mask, "mesh_path": mesh, "transform_path": pose,
            "intrinsics_path": sam_intrinsics, "weights_dir": weights["sam3d"], "seed": seed,
            "with_layout_postprocess": True}, [rgb, mask], [mesh, pose, sam_intrinsics])
    module("06_simplify", "v2d_mesh", "run_mesh_simplify.py", "run_mesh_simplify.py",
           {"input_mesh": mesh, "output_mesh": simplified, "face_count": face_count}, [mesh], [simplified])
    module("07_align_scale", "v2d_mesh", "run_mesh_align_depth.py", "run_mesh_align_depth.py",
           {"mesh": simplified, "depth": depth, "intrinsics": intrinsics,
            "transform": pose, "output_transform": scale_transform}, [simplified, depth, intrinsics, pose], [scale_transform])
    module("08_scale_mesh", "v2d_mesh", "run_mesh_transform.py", "run_mesh_transform.py",
           {"input_mesh": simplified, "transform": scale_transform, "output_mesh": scaled},
           [simplified, scale_transform], [scaled])
    evidence = (mesh, simplified, rgb, mask, depth, intrinsics, pose)
    local("09_scale_record", [video, scaled, scale_transform, *evidence], [scale_record],
          lambda: _scale_record(video, scaled, scale_transform, request.reference_frame, evidence, scale_record))
    module("10_pack_masks", "v2d_cari4d", "pack_masks.py", "run_pack_masks.py",
           {"video_path": video, "human_masks_path": human, "object_masks_path": obj, "output_path": packed},
           [video, human, obj], [packed])
    module("11_validate", "v2d_cari4d", "validate_inputs.py", "run_validate_inputs.py",
           {"video_path": video, "mask_h5_path": packed, "object_mesh_path": scaled, "report_path": validation,
            "expected_frames": episode.expected_frames, "expected_fps": episode.fps,
            "source_video_path": episode.source_video, "mesh_scale_report_path": scale_record,
            "require_scale_provenance": True}, [video, packed, scaled, scale_record, episode.source_video], [validation])
    checked = json.loads(validation.read_text())
    if (checked.get("schema") != "v2d.cari4d.input_validation.v1" or checked.get("sequence") != sequence
            or checked.get("verdict") != "PASS" or checked.get("scale_provenance_verified") is not True
            or checked.get("video", {}).get("decoded_frames") != episode.expected_frames):
        raise ValueError("Prepared inputs did not pass their content checks")
    local("12_review", [video, human, obj], [review],
          lambda: _mask_review(video, human, obj, episode.expected_frames, request.reference_frame, review))
    paths = {"video_path": video, "mask_h5_path": packed, "object_mesh_path": scaled, "mesh_scale_report_path": scale_record}
    result = {"episode_index": episode.episode_index, "status": "PASS", "stages": reports,
              "inputs": {name: str(path) for name, path in paths.items()},
              "artifacts": {name: artifact_record(path) for name, path in {**paths, "validation": validation, "review": review}.items()},
              "semantic_review": "required", "validation_warnings": checked.get("issues", [])}
    atomic_json(directory / "preparation.json", result)
    return result


def prepare_track1(dataset_root: str, preparation_manifest: str, output_dir: str,
                   sam2_weights: str, moge_weights: str, sam3d_weights: str, *,
                   image_build_commit: str, image_digest: str, episodes: list[int] | None = None,
                   execute: bool = False, all_episodes: bool = False, face_count: int = 50000, seed: int = 0,
                   runtime_python: str = sys.executable, sam2_python: str = "/opt/venvs/sam2/bin/python",
                   sam3d_python: str = "/opt/venvs/sam3d/bin/python", stage_timeout_seconds: float = 3600) -> Path:
    """Plan or prepare episodes in a provisioned image; publish only validated inputs."""
    from v2d.common.video import FrameSource

    if type(face_count) is not int or face_count <= 0 or type(seed) is not int:
        raise ValueError("face_count must be positive and seed must be an integer")
    if not math.isfinite(stage_timeout_seconds) or stage_timeout_seconds <= 0:
        raise ValueError("stage_timeout_seconds must be finite and positive")
    if (not re.fullmatch(r"[0-9a-f]{40}", image_build_commit)
            or not re.fullmatch(r"(?:[^\s]+@)?sha256:[0-9a-f]{64}", image_digest)):
        raise ValueError("Record a full image build commit and immutable image digest")
    dataset, manifest, output = map(lambda p: Path(p).resolve(), (dataset_root, preparation_manifest, output_dir))
    if output.is_relative_to(REPOSITORY):
        raise ValueError("Keep preparation output_dir outside the business checkout")
    initial = {"manifest": artifact_record(manifest), "metadata": track1_metadata_identity(dataset)}
    metadata = {e.episode_index: e for e in track1_episodes(dataset)}
    requests = preparation_requests(manifest, metadata)
    selected = sorted(metadata) if episodes is None else episodes
    if (not selected or len(set(selected)) != len(selected)
            or any(type(i) is not int or i not in metadata for i in selected)):
        raise ValueError("Select unique known episodes")
    if all_episodes and episodes is not None:
        raise ValueError("Choose all_episodes or a subset")
    if execute and set(selected) == set(metadata) and not all_episodes:
        raise ValueError("Full-roster preparation requires explicit all_episodes=True")
    versions = {**checkout_identity(REPOSITORY, execute), "image_build_commit": image_build_commit, "image_digest": image_digest}
    initial_inputs = _input_identity(metadata, requests) if execute else None
    weights = {key: Path(path).resolve() for key, path in (("sam2", sam2_weights), ("moge", moge_weights), ("sam3d", sam3d_weights))}
    runtimes = {"base": Track1Runtime(MODULES, "container", runtime_python),
                "v2d_sam2": Track1Runtime(MODULES, "container", sam2_python),
                "v2d_sam3d": Track1Runtime(MODULES, "container", sam3d_python)}
    # Check prompts before provisioning any model state; missing requests remain
    # visible in plans and fail only their own episode during execution.
    prompts, issues = {}, {}
    for index in selected:
        try:
            if index not in requests:
                raise ValueError("No preparation request/prompt has been configured")
            with FrameSource.from_path(metadata[index].source_video) as video:
                prompts[index] = validated_prompts(requests[index], metadata[index], video.image_size)
        except (OSError, ValueError, KeyError, TypeError) as error:
            issues[index] = str(error)
    plan = {"schema": "v2d.track1.preparation_plan.v1", "verdict": "PLANNED", "execute": execute,
            "versions": versions, "selected_episodes": selected, "dataset_episodes": len(metadata),
            "face_count": face_count, "seed": seed, "weights": {k: str(v) for k, v in weights.items()},
            "python": {k: v.python for k, v in runtimes.items()},
            "episodes": [{"episode_index": i, "expected_frames": metadata[i].expected_frames,
                          "request": {**asdict(requests[i]), "prompts_path": str(requests[i].prompts_path)} if i in requests else None,
                          "issue": issues.get(i)} for i in selected]}
    with artifact_lock(output.parent / f".{output.name}.lock"):
        if not execute:
            atomic_json(output / "preparation_plan.json", plan)
            return output / "preparation_plan.json"
        base_packages = {"v2d.common": MODULES / "v2d_common", **{f"v2d.{name}.lib": MODULES / f"v2d_{name}/lib"
                         for name in ("cari4d", "moge", "mesh")}}
        runtimes["base"].verify_packages(base_packages)
        for name in ("sam2", "sam3d"):
            runtimes[f"v2d_{name}"].verify_packages({"v2d.common": MODULES / "v2d_common",
                                                   f"v2d.{name}.lib": MODULES / f"v2d_{name}/lib"})

        def identity():
            if {"manifest": artifact_record(manifest), "metadata": track1_metadata_identity(dataset)} != initial:
                raise ValueError("Preparation manifest or dataset metadata changed")
            inputs = _input_identity(metadata, requests)
            if inputs != initial_inputs:
                raise ValueError("Preparation inputs changed while planning or running; use a new output_dir")
            return {"schema": "v2d.track1.preparation_identity.v1", "manifest": initial,
                    "versions": {**versions, **checkout_identity(REPOSITORY, True)}, "face_count": face_count, "seed": seed,
                    "python": plan["python"], "runtime_packages": {k: v.package_versions() for k, v in runtimes.items()},
                    "weights": {k: weight_identity(path) for k, path in weights.items()},
                    "inputs": inputs}
        current = identity()
        identity_path = output / "preparation_identity.json"
        if identity_path.exists():
            if json.loads(identity_path.read_text()) != current:
                raise ValueError("Preparation inputs/source/weights/settings changed; use a new output_dir")
        elif output.exists() and any(p.name != "preparation_plan.json" for p in output.iterdir()):
            raise ValueError("Existing unidentified preparation outputs require a new output_dir")
        else:
            atomic_json(identity_path, current)
        atomic_json(output / "preparation_plan.json", plan)
        report_path = output / "preparation_report.json"
        old = json.loads(report_path.read_text()) if report_path.exists() else {}
        rows = old.get("episodes", {})
        _verify_prepared_rows(rows, metadata, output)
        report = {"schema": "v2d.track1.preparation.v1", "verdict": "RUNNING", "versions": versions,
                  "selected_episodes": selected, "dataset_episodes": len(metadata), "episodes": rows}

        def publish():
            passed = [row for row in rows.values() if row.get("status") == "PASS"]
            report["summary"] = {"pass": len(passed), "not_prepared": len(metadata) - len(passed)}
            atomic_json(report_path, report)
            atomic_json(output / "inputs.json", {"schema": "v2d.track1.inputs.v1", "episodes": [
                {"episode_index": row["episode_index"], **row["inputs"]} for row in sorted(passed, key=lambda r: r["episode_index"])]})

        publish()
        try:
            for index in selected:
                started = time.monotonic()
                try:
                    rows[str(index)] = {"episode_index": index, "status": "RUNNING"}
                    publish()
                    if index in issues:
                        raise ValueError(issues[index])
                    rows[str(index)] = _prepare_episode(metadata[index], requests[index], prompts[index], output,
                        identity_path, runtimes, weights, face_count, seed, stage_timeout_seconds)
                except (OSError, ValueError, KeyError, TypeError, IndexError, StopIteration, subprocess.SubprocessError) as error:
                    rows[str(index)] = {"episode_index": index, "status": "FAIL", "error_type": type(error).__name__, "error": str(error)}
                finally:
                    rows[str(index)]["elapsed_seconds"] = time.monotonic() - started
                    publish()
                print(f"TRACK1_PREPARED {index:06d} {rows[str(index)]['status']}", flush=True)
            if identity() != current:
                raise ValueError("Source, runtime, or inputs changed during preparation")
            _verify_prepared_rows(rows, metadata, output)
            report["verdict"] = "PASS" if all(rows[str(i)]["status"] == "PASS" for i in selected) else "FAIL"
        except BaseException as error:
            report.update(verdict="INTERRUPTED" if isinstance(error, KeyboardInterrupt) else "FAIL", error=str(error))
            for row in rows.values():
                if row.get("status") == "RUNNING":
                    row["status"] = "INTERRUPTED"
            publish()
            # On a global identity failure/cancellation keep artifacts as evidence,
            # but leave no automatically consumable stale manifest.
            atomic_json(output / "inputs.json", {"schema": "v2d.track1.inputs.v1", "episodes": []})
            raise
        publish()
        return report_path


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("dataset_root", "preparation_manifest", "output_dir", "sam2_weights", "moge_weights",
                 "sam3d_weights", "image_build_commit", "image_digest"):
        parser.add_argument(f"--{name}", required=True)
    parser.add_argument("--episodes", nargs="+", type=int)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--all_episodes", action="store_true")
    parser.add_argument("--face_count", type=int, default=50000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--runtime_python", default=sys.executable)
    parser.add_argument("--sam2_python", default="/opt/venvs/sam2/bin/python")
    parser.add_argument("--sam3d_python", default="/opt/venvs/sam3d/bin/python")
    parser.add_argument("--stage_timeout_seconds", type=float, default=3600)
    return parser


if __name__ == "__main__":
    path = prepare_track1(**vars(_parser().parse_args()))
    print(path)
    raise SystemExit(0 if json.loads(path.read_text())["verdict"] in {"PASS", "PLANNED"} else 2)
