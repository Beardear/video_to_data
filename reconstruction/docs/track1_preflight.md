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
      "object_mesh_path": "outputs/episode_000016/baseline-inputs/sam3d_mesh/object_scaled_50k.glb"
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

CPU-only contract tests, from the repository root:

```bash
python -m unittest discover -s reconstruction/modules/v2d_pipelines/tests -p 'test_track1_preflight.py' -v
```
