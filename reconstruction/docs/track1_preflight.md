# Track 1 input inventory

Before preparing or running a batch, inventory the released Track 1 dataset:

```bash
python -m v2d.pipelines.track1_preflight --dataset_root /data/track_1
```

Install the host packages as described in `reconstruction/CONTRIBUTING.md`, or
run `python reconstruction/modules/v2d_pipelines/track1_preflight.py` from the
repository root with the same arguments. This entry point uses only the Python
standard library. It does not download data, launch Docker or GPU jobs, rename
videos, or create masks and meshes.

The JSON report lists each episode's action, object prompt, source video,
expected frame count, and missing prepared inputs. Paths and chunk numbers come
from the LeRobot v2.1 metadata, not a hard-coded list of episodes.

To inspect prepared inputs, pass `--inputs_manifest /data/inputs.json`. Paths
are absolute or relative to that manifest's directory. Partial manifests are
supported; omitted paths are reported as `not_configured`. For example:

```json
{
  "schema": "v2d.track1.inputs.v1",
  "episodes": [
    {
      "episode_index": 16,
      "video_path": "outputs/episode_000016/baseline-inputs/episode_000016.0.color.mp4",
      "mask_h5_path": "outputs/episode_000016/baseline-inputs/episode_000016_masks_k0.h5",
      "object_mesh_path": "outputs/episode_000016/baseline-inputs/sam3d_mesh/object_scaled_50k.glb",
      "mesh_scale_report_path": "outputs/episode_000016/baseline-inputs/mesh_scale.json"
    }
  ]
}
```

Use `--episodes 16` to inspect one episode, or `--episodes 0 16 29` for a subset.
The full metadata roster is validated even when inspecting a subset. Copy a
prepared video to the CARI4D name `episode_XXXXXX.0.color.mp4`; a symlink to the
original dataset filename does not satisfy its resolved-name contract. Masks
must correspond to that episode, and the selected mesh must have its scale
established during input preparation. This inventory does not establish either.

Exit codes:

- `0`: all selected source and prepared files are nonempty regular files with
  the expected prepared-video names.
- `2`: some selected files need preparation; the JSON report describes them.
- `1`: invalid/inconsistent metadata, invalid input configuration, or read error.

`validation_level` is **metadata_and_file_presence**. Frame counts in the report
are expected counts from metadata; videos have not been decoded. Mask contents,
mesh geometry/scale, weights, GPU readiness, and source/cache identity remain
unchecked. A zero exit code is not permission to reuse inference outputs or
evidence of successful reconstruction. Scan the mounted cloud volume separately
to inspect its inputs; a local inventory describes only local files.

The scale-report path is optional for this file inventory; when provided, its
presence is checked. The [batch runner](track1_batch.md) requires it and performs
the content validation below before inference.

## Prepared-input content validation

After preparing an episode, use the CARI4D container's CPU checker. It loads no
model and requires no GPU:

```bash
python -m v2d.cari4d.docker.run_validate_inputs \
  --video_path /data/inputs/episode_000016.0.color.mp4 \
  --source_video_path /data/track_1/videos/chunk-000/observation.images.exo_camera/episode_000016.mp4 \
  --mask_h5_path /data/inputs/episode_000016_masks_k0.h5 \
  --object_mesh_path /data/inputs/object_scaled.glb \
  --mesh_scale_report_path /data/inputs/mesh_scale.json \
  --require_scale_provenance --expected_frames 360 --expected_fps 30 \
  --report_path /data/inputs/validation.json --dev
```

Inside an existing container, invoke `v2d.cari4d.lib.validate_inputs` with the
same arguments except `--dev`. Counts and FPS must come from dataset metadata.
The report hashes all inputs, fully decodes the video, checks timestamps and
dimensions, checks exact HDF5 frame/role keys and binary masks, and validates
mesh vertices, faces and surface area after applying scene-node transforms.
Human masks must be nonempty. The object mask must be nonempty on frame zero;
later empty object masks are explicit occlusion warnings. Non-watertight meshes
also produce warnings. Inspect warnings before scheduling an expensive run.

The optional scale record has this contract (hashes are full SHA-256 values):

```json
{
  "schema": "v2d.track1.mesh_scale.v1",
  "units": "metres",
  "method": "depth_alignment",
  "reference_frame": 0,
  "applied_scale": [0.12, 0.12, 0.12],
  "mesh_sha256": "<hash of the scaled mesh>",
  "source_video_sha256": "<hash of the dataset video>"
}
```

Record actual preparation measurements, never guessed scale factors. Supported
methods are `depth_alignment`, `sam3d_pointmap`, and `measured`. The record binds
the method, positive scale factors, and reference frame to these exact files;
it does not prove physical scale against ground truth. Additional preparation
evidence may be stored in the same record. Missing scale records warn by default
and fail when `--require_scale_provenance` is set. Supplying an inconsistent
record always fails.

The CLI exits `0` for `PASS`, `2` for content `FAIL`, and nonzero on an unreadable
or corrupt input. A caller must require a successful exit and a `PASS` report
from the current invocation; an old report never overrides a read failure.
Segmentation semantics, reconstruction accuracy, and GPU execution remain
unverified. Content validation is a separate step from the lightweight inventory.

CPU-only contract tests, from the repository root:

```bash
python -m unittest discover -s reconstruction/modules/v2d_pipelines/tests -p 'test_track1_preflight.py' -v
python -m pytest reconstruction/modules/v2d_cari4d/tests/test_validate_inputs.py -q
```

The content tests need CPU packages `av`, `h5py`, `numpy`, `scipy`, `trimesh`, and
`pytest`, plus the current `v2d-common` checkout installed with
`pip install -e reconstruction/modules/v2d_common`; no PyTorch or weights are
required. Update this shared package in a Pod too when running new business code
against an older image.
