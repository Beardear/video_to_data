# Plan and run a Track 1 batch

The runner composes existing CARI4D entry points:

```text
dataset metadata + prepared-input manifest
  → content validation (CPU)
  → eight-stage CARI4D inference
  → official MHR conversion and accepted episode export
  → per-episode logs and aggregate batch report
```

Input preparation uses [the separate preparation runner](track1_prepare.md).
Every selected episode needs the three input
paths described in [the inventory](track1_preflight.md), plus a
`mesh_scale_report_path`. The scale record is checked before inference. The
runner never invents masks, substitutes meshes, downloads weights, creates a GPU,
or submits to Kaggle. Planning is the default; execution is opt-in.

## Runtime and source setup

Run from a Git checkout with the host packages installed:

```bash
pip install -e reconstruction/modules/v2d_common \
  -e 'reconstruction/modules/v2d_pipelines[track1]'
```

The default `--runtime_mode docker` calls the existing Docker wrappers with
`--dev`. Install the current checkout's lightweight wrapper packages:

```bash
pip install --no-deps -e reconstruction/modules/v2d_docker \
  -e reconstruction/modules/v2d_cari4d/docker
```

The image tagged by the CARI4D wrapper must match the supplied immutable digest;
the runner checks Docker's actual `RepoDigests`. For a RunPod already running
the CARI4D image, choose `--runtime_mode container` and install the current
business libraries into that container's Python environment:

```bash
python -m pip install --no-deps --no-build-isolation \
  -e reconstruction/modules/v2d_common -e reconstruction/modules/v2d_cari4d/lib
```

Use `--runtime_python /path/to/python` if needed. Imports are checked against the
current source directory before execution. Cloning does not update the image's
installed packages. Container mode records the image digest supplied from the
deployment record; verify it against the Pod when allocating the resource. It
cannot independently inspect its host's image store.

Commit and push the business code before executing. Dirty checkouts are permitted
for planning only. Keep experiment output outside the business checkout.

## Plan a bounded run

`track1_baseline.json` preserves the existing baseline: 300 refinement steps,
16-frame refinement batches, official float64 fitting, and explicit `report`
conversion acceptance. These are experiment parameters, not a new algorithm.
The config accepts keyword settings from the existing inference/export APIs;
episode paths, frame counts, provenance and overwrite controls are reserved.

```bash
python -m v2d.pipelines.track1_batch \
  --dataset_root /vol/data/track_1 \
  --inputs_manifest /vol/inputs/track1/inputs.json \
  --weights_path /vol/weights/cari4d \
  --submission_kit /vol/v2d_submission_kit \
  --output_dir /vol/outputs/track1/run-001 \
  --config_path reconstruction/modules/v2d_pipelines/track1_baseline.json \
  --image_build_commit "$IMAGE_BUILD_COMMIT" --image_digest "$IMAGE_DIGEST" \
  --runtime_mode container --episodes 16
```

This writes `batch_plan.json`, including exact child commands, selected frame
counts and missing prepared inputs. It starts no model or Docker job. Set the
version variables from the actual deployment record. The business commit comes
from this checkout's Git HEAD.

After reviewing the plan, add `--execute` to the same command. The default is
at most two attempts per episode, with a one-hour limit per validation,
inference, or export invocation. Override with `--max_attempts` (1–5) and
`--stage_timeout_seconds`. These limits do not impose a global GPU billing cap;
resource lifetime must also be bounded by the deployment operator.

To execute the full roster later, omit `--episodes` and pass both `--execute`
and `--all_episodes`. Listing every episode explicitly does not bypass this
scope guard. A normal plan may cover the full roster without either flag.

## Failure handling and resume

- A content failure never reaches inference and is not retried. The next
  episode still runs.
- An inference/export process failure or timeout is retried up to the configured
  limit. Every attempt has a separate log. Existing CARI4D stage identities
  decide which inference stages can be reused.
- Rerun the same command to resume, or select failed episode indexes. Accepted
  exports are revalidated before reuse. A changed source commit, input content,
  weight content, image identity, or configuration requires a new output directory.
- Cancellation and timeouts terminate the command's process group. Docker jobs
  receive unique names so their exact containers can also be removed. Unverified
  Docker cleanup is an error, not successful completion.

Outputs:

```text
batch_plan.json
batch_identity.json
batch_report.json
validation/episode_XXXXXX.json
inference/episode_XXXXXX/...
exports/episode_XXXXXX/...
logs/episode_XXXXXX/attempt-001/{validate,inference,export}.log
```

The report retains previous attempts, tracks the current selection, and shows
cumulative completed episodes/frames. A subset `PASS` does not mean that the full
dataset was reconstructed. CLI exit codes are `0` for a plan or successful
selected batch, `2` for episode failures, and nonzero for configuration errors or
interruptions. Use the [separate official packing command](track1_pack.md) once
real exports are ready.

## CPU verification

```bash
PYTHONDONTWRITEBYTECODE=1 python -m pytest \
  tests/test_track1_batch.py tests/test_track1_runtime.py -q
```

Batch tests use explicit synthetic stage substitutes to exercise failure
isolation, bounded retries, verified reuse, changed-input rejection, cancellation
and report continuity. Runtime tests launch real CPU processes to verify timeout
and orphan-worker cleanup. They do not establish real GPU reconstruction success.
