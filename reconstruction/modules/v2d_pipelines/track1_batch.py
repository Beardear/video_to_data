"""Plan or execute Track 1 episodes with isolated failures and verified resume.

Planning is the default. Full-roster execution additionally requires the
explicit --all_episodes switch. No command here creates GPUs or submits results.
"""

from __future__ import annotations

import argparse
import ast
from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
import re
import subprocess
import sys
import time
from typing import Any

from v2d.common.artifacts import artifact_lock, artifact_record, atomic_json
from v2d.common.track1_conversion import track1_scoring
from v2d.pipelines.track1_pack import _accepted_export
from v2d.pipelines.track1_preflight import (
    INPUT_ROLES, Track1Episode, track1_episodes, track1_metadata_identity, track1_prepared_inputs,
)
from v2d.pipelines.track1_runtime import ModuleInvocation, Track1Runtime
from v2d.pipelines.track1_experiment import checkout_identity, weight_identity


REPOSITORY = Path(__file__).resolve().parents[3]
MODULES = REPOSITORY / "reconstruction/modules"


@dataclass(frozen=True)
class BaselineSettings:
    inference: dict[str, Any]
    export: dict[str, Any]


def _settings(path: Path) -> BaselineSettings:
    payload = json.loads(path.read_text())
    if (not isinstance(payload, dict) or payload.get("schema") != "v2d.track1.baseline.v1"
            or set(payload) != {"schema", "inference", "export"}):
        raise ValueError("Expected a v2d.track1.baseline.v1 configuration")
    result = {}
    # Read literal keyword defaults without importing the ML implementation.
    # The existing lib function/CLI remains the parameter contract.
    for name, file, function, reserved in (
        ("inference", "run_inference.py", "run_inference", {"expected_frames", "overwrite"}),
        ("export", "export_track1.py", "export_track1", {
            "episode_index", "expected_frames", "business_commit", "image_build_commit", "image_digest"}),
    ):
        tree = ast.parse((MODULES / "v2d_cari4d/lib" / file).read_text())
        node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == function)
        defaults = {a.arg: ast.literal_eval(value) for a, value in zip(node.args.kwonlyargs, node.args.kw_defaults)
                    if value is not None and a.arg not in reserved}
        supplied = payload[name]
        if not isinstance(supplied, dict) or set(supplied) - set(defaults):
            raise ValueError(f"Unknown or reserved {name} configuration parameters")
        for key, value in supplied.items():
            default = defaults[key]
            if value is None and default is None:
                continue
            valid = (type(value) in (str, int, float, bool) and
                     (default is None or type(value) is type(default)
                      or (type(default) is float and type(value) is int)))
            if not valid or (type(value) is float and not math.isfinite(value)):
                raise ValueError(f"Invalid type or non-finite {name}.{key}")
        result[name] = {**defaults, **supplied}
    if result["inference"]["download_models"]:
        raise ValueError("Prepare weights before scheduling; batch download_models must be false")
    if result["export"]["conversion_error_policy"] not in {"reject", "report"}:
        raise ValueError("Choose an explicit supported conversion_error_policy")
    return BaselineSettings(**result)


def _checkout_identity(execute: bool) -> dict[str, Any]:
    return checkout_identity(REPOSITORY, execute)


def _weight_identity(root: Path) -> dict[str, dict[str, Any]]:
    return weight_identity(root)


def _commands(episode: Track1Episode, paths: dict[str, Path], output: Path, weights: Path, kit: Path,
              runtime: Track1Runtime, settings: BaselineSettings, versions: dict[str, str],
              attempt: int) -> tuple[ModuleInvocation, ...]:
    sequence = f"episode_{episode.episode_index:06d}"
    log_root = output / "logs" / sequence / f"attempt-{attempt:03d}"
    arguments = [
        ("validate", "validate_inputs.py", "run_validate_inputs.py", {
            **paths, "source_video_path": episode.source_video,
            "report_path": output / "validation" / f"{sequence}.json",
            "expected_frames": episode.expected_frames, "expected_fps": episode.fps,
            "require_scale_provenance": True}),
        ("inference", "run_inference.py", "run_inference.py", {
            **{role: paths.get(role) for role in INPUT_ROLES},
            **settings.inference, "weights_path": weights, "output_dir": output / "inference",
            "expected_frames": episode.expected_frames}),
        ("export", "export_track1.py", "run_export_track1.py", {
            **settings.export, **versions, "result_dir": output / "inference" / sequence,
            "submission_kit": kit, "weights_path": weights, "output_dir": output / "exports" / sequence,
            "episode_index": episode.episode_index, "expected_frames": episode.expected_frames}),
    ]
    return tuple(runtime.invocation(name, "v2d_cari4d", lib, wrapper, args, log_root / f"{name}.log")
                 for name, lib, wrapper, args in arguments)


def _validate_report(output: Path, episode: Track1Episode) -> None:
    sequence = f"episode_{episode.episode_index:06d}"
    report = json.loads((output / "validation" / f"{sequence}.json").read_text())
    if (report.get("schema") != "v2d.cari4d.input_validation.v1" or report.get("verdict") != "PASS"
            or report.get("sequence") != sequence or report.get("scale_provenance_verified") is not True
            or report.get("video", {}).get("decoded_frames") != episode.expected_frames):
        raise ValueError(f"{sequence}: input content validation did not pass")


def _accept_export(exports: Path, episode: Track1Episode, settings: BaselineSettings,
                   versions: dict[str, str], kit: Path) -> None:
    accepted = _accepted_export(exports, episode, versions["business_commit"],
                               track1_scoring(kit, episode.episode_index, episode.expected_frames))
    expected_inference = {key: value for key, value in settings.inference.items() if key != "download_models"}
    if (any(accepted.provenance[name] != value for name, value in versions.items())
            or accepted.provenance["settings"] != settings.export
            or accepted.provenance["inference_settings"] != expected_inference):
        raise ValueError("Export provenance does not match the current batch configuration")


def _save_report(path: Path, report: dict[str, Any]) -> None:
    rows = report["episodes"]
    report["summary"] = {status.lower(): sum(rows.get(str(i), {}).get("status") == status
                                             for i in report["selected_episodes"])
                         for status in ("PASS", "FAIL")}
    passed = [row for row in rows.values() if row.get("status") == "PASS"]
    report["cumulative_summary"] = {
        "pass": len(passed), "fail": sum(row.get("status") == "FAIL" for row in rows.values()),
        "not_passed": report["dataset_episodes"] - len(passed),
        "frames_passed": sum(row.get("expected_frames", 0) for row in passed),
    }
    atomic_json(path, report)


def run_track1_batch(
    dataset_root: str, inputs_manifest: str, weights_path: str, submission_kit: str,
    output_dir: str, config_path: str, *, image_build_commit: str, image_digest: str,
    episodes: list[int] | None = None, execute: bool = False, all_episodes: bool = False,
    runtime_mode: str = "docker", runtime_python: str = sys.executable,
    max_attempts: int = 2, stage_timeout_seconds: float = 3600,
) -> Path:
    """Write an execution plan or run selected episodes, retaining every attempt's log."""
    if type(max_attempts) is not int or not 1 <= max_attempts <= 5:
        raise ValueError("max_attempts must be between 1 and 5")
    if not math.isfinite(stage_timeout_seconds) or stage_timeout_seconds <= 0:
        raise ValueError("stage_timeout_seconds must be finite and positive")
    if not re.fullmatch(r"[0-9a-f]{40}", image_build_commit):
        raise ValueError("image_build_commit must be a full commit SHA")
    if not re.fullmatch(r"(?:[^\s]+@)?sha256:[0-9a-f]{64}", image_digest):
        raise ValueError("image_digest must be immutable")
    dataset, manifest, weights, kit, output, config = map(lambda p: Path(p).resolve(),
        (dataset_root, inputs_manifest, weights_path, submission_kit, output_dir, config_path))
    if output.is_relative_to(REPOSITORY):
        raise ValueError("Keep batch output_dir outside the business checkout")
    initial = {"config": artifact_record(config), "manifest": artifact_record(manifest),
               "metadata": track1_metadata_identity(dataset)}
    metadata = {e.episode_index: e for e in track1_episodes(dataset)}
    selected = sorted(metadata) if episodes is None else episodes
    if (not selected or len(set(selected)) != len(selected)
            or any(type(i) is not int or i not in metadata for i in selected)):
        raise ValueError("Select unique known episode indexes")
    if execute and set(selected) == set(metadata) and not all_episodes:
        raise ValueError("Full-roster execution requires explicit all_episodes=True")
    if all_episodes and episodes is not None:
        raise ValueError("Choose all_episodes or an episode subset, not both")
    prepared = track1_prepared_inputs(manifest, set(metadata))
    settings = _settings(config)
    checkout = _checkout_identity(execute)
    versions = {"business_commit": checkout["business_commit"],
                "image_build_commit": image_build_commit, "image_digest": image_digest}
    runtime = Track1Runtime(MODULES, runtime_mode, runtime_python)
    plan = {"schema": "v2d.track1.batch_plan.v1", "verdict": "PLANNED", "execute": execute,
            "checkout": checkout, "versions": versions, "settings": asdict(settings),
            "selected_episodes": selected, "dataset_episodes": len(metadata),
            "max_attempts": max_attempts, "stage_timeout_seconds": stage_timeout_seconds,
            "episodes": [{"episode_index": i, "expected_frames": metadata[i].expected_frames,
                "missing_inputs": [role for role in (*INPUT_ROLES, "mesh_scale_report_path")
                                   if role not in prepared.get(i, {}) or not prepared[i][role].is_file()],
                "commands": [dict(name=c.name, command=c.command, log_path=str(c.log_path)) for c in
                    _commands(metadata[i], prepared.get(i, {}), output, weights, kit, runtime, settings, versions, 1)]}
                for i in selected]}
    with artifact_lock(output.parent / f".{output.name}.lock"):
        if not execute:
            atomic_json(output / "batch_plan.json", plan)
            return output / "batch_plan.json"
        package = "lib" if runtime_mode == "container" else "docker"
        packages = {"v2d.common": MODULES / "v2d_common",
                    f"v2d.cari4d.{package}": MODULES / f"v2d_cari4d/{package}"}
        if runtime_mode == "docker":
            packages["v2d.docker"] = MODULES / "v2d_docker"
        runtime.verify_packages(packages)
        image_identity = runtime.verify_cari4d_image(image_digest)

        def identity():
            if {"config": artifact_record(config), "manifest": artifact_record(manifest),
                "metadata": track1_metadata_identity(dataset)} != initial:
                raise ValueError("Batch configuration or metadata changed while reading it")
            paths = {"inputs_manifest": manifest, "config": config,
                     "official_converter": kit / "tools/track1/mesh_to_mhr_params.py",
                     "official_metrics": kit / "v2dlb/mhr_metrics.py",
                     "official_sample": kit / "data/track_1_sample_submission.parquet"}
            for index, entry in prepared.items():
                paths.update({f"episode_{index}_{role}": path for role, path in entry.items()})
            paths.update({f"source_video_{i}": e.source_video for i, e in metadata.items()})
            return {"schema": "v2d.track1.batch_identity.v1", "checkout": _checkout_identity(True),
                    "versions": versions, "settings": asdict(settings),
                    "runtime": {"mode": runtime_mode, "python": runtime_python, "image": image_identity},
                    "inputs": {name: artifact_record(p) if p.is_file() else {"path": str(p), "missing": True}
                               for name, p in paths.items()},
                    "dataset_metadata": track1_metadata_identity(dataset), "weights": _weight_identity(weights)}

        current = identity()
        identity_path = output / "batch_identity.json"
        if identity_path.exists():
            if json.loads(identity_path.read_text()) != current:
                raise ValueError("Batch source, inputs, weights, or configuration changed; use a new output_dir")
        elif output.exists() and any(p.name != "batch_plan.json" for p in output.iterdir()):
            raise ValueError("Unidentified batch outputs cannot be adopted; use a new output_dir")
        else:
            atomic_json(identity_path, current)
        atomic_json(output / "batch_plan.json", plan)
        report_path = output / "batch_report.json"
        previous = json.loads(report_path.read_text()) if report_path.exists() else {}
        rows = previous.get("episodes", {})
        report = {"schema": "v2d.track1.batch.v1", "verdict": "RUNNING", "versions": versions,
                  "selected_episodes": selected, "dataset_episodes": len(metadata), "episodes": rows,
                  "scope": "full_roster" if set(selected) == set(metadata) else "subset",
                  "kaggle_submitted": False}
        _save_report(report_path, report)
        try:
            for index in selected:
                episode = metadata[index]
                key = str(index)
                prior = rows.get(key, {})
                attempts = list(prior.get("attempts", []))
                row = {"episode_index": index, "expected_frames": episode.expected_frames,
                       "status": "RUNNING", "attempts": attempts}
                rows[key] = row
                for _ in range(max_attempts):
                    attempt = {"number": len(attempts) + 1, "status": "RUNNING", "stage": "inputs"}
                    attempts.append(attempt)
                    started = time.monotonic()
                    try:
                        paths = prepared.get(index, {})
                        missing = [role for role in (*INPUT_ROLES, "mesh_scale_report_path") if role not in paths]
                        if missing:
                            raise ValueError(f"Missing prepared input roles: {missing}")
                        for path in (*paths.values(), episode.source_video):
                            if not path.is_file() or path.stat().st_size == 0:
                                raise FileNotFoundError(path)
                        exports = output / "exports"
                        if (exports / f"episode_{index:06d}").exists():
                            _accept_export(exports, episode, settings, versions, kit)
                            attempt.update(status="PASS", stage="reuse", reused=True)
                        else:
                            for invocation in _commands(episode, paths, output, weights, kit, runtime, settings,
                                                        versions, attempt["number"]):
                                attempt.update(stage=invocation.name, log_path=str(invocation.log_path))
                                _save_report(report_path, report)
                                runtime.execute(invocation, stage_timeout_seconds)
                                if invocation.name == "validate":
                                    _validate_report(output, episode)
                            _accept_export(exports, episode, settings, versions, kit)
                            attempt.update(status="PASS", reused=False)
                        row["status"] = "PASS"
                    except (OSError, ValueError, KeyError, subprocess.SubprocessError) as error:
                        attempt.update(status="FAIL", error_type=type(error).__name__, error=str(error))
                        row["status"] = "FAIL"
                    finally:
                        attempt["elapsed_seconds"] = time.monotonic() - started
                        _save_report(report_path, report)
                    retryable = (attempt["status"] == "FAIL" and attempt["stage"] in {"inference", "export"}
                                 and attempt["error_type"] in {"CalledProcessError", "TimeoutExpired"})
                    if not retryable:
                        break
                print(f"TRACK1_EPISODE {index:06d} {row['status']}", flush=True)
            if identity() != current:
                raise ValueError("Source or inputs changed during batch execution; use a new output_dir")
            report["verdict"] = "PASS" if all(rows[str(i)]["status"] == "PASS" for i in selected) else "FAIL"
        except BaseException as error:
            for row in rows.values():
                if row.get("status") == "RUNNING":
                    row["status"] = "INTERRUPTED"
                    row["attempts"][-1]["status"] = "INTERRUPTED"
            report.update(verdict="INTERRUPTED" if isinstance(error, KeyboardInterrupt) else "FAIL",
                          error_type=type(error).__name__, error=str(error))
            _save_report(report_path, report)
            raise
        _save_report(report_path, report)
        return report_path


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("dataset_root", "inputs_manifest", "weights_path", "submission_kit", "output_dir",
                 "config_path", "image_build_commit", "image_digest"):
        parser.add_argument(f"--{name}", required=True)
    parser.add_argument("--episodes", nargs="+", type=int)
    parser.add_argument("--all_episodes", action="store_true")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--runtime_mode", choices=("docker", "container"), default="docker")
    parser.add_argument("--runtime_python", default=sys.executable)
    parser.add_argument("--max_attempts", type=int, default=2)
    parser.add_argument("--stage_timeout_seconds", type=float, default=3600)
    return parser


if __name__ == "__main__":
    result = run_track1_batch(**vars(_parser().parse_args()))
    print(result)
    raise SystemExit(0 if json.loads(result.read_text())["verdict"] in {"PASS", "PLANNED"} else 2)
