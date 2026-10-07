# Prepare Track 1 inputs

This entry point composes the existing algorithms inside the provisioned GPU
image. It prepares inputs for [the batch runner](track1_batch.md):

```text
dataset videos + reviewed human/object prompts
  → SAM2 human/object mask streams
  → selected reference RGB + object mask
  → MoGe depth + SAM3D static mesh and pose
  → mesh simplify → depth alignment → scale-only mesh transform
  → CARI4D mask packing + input content validation
  → inputs.json + scale evidence + mask review images
```

Preparation does not run the temporal CARI4D reconstruction, fit the official MHR
submission parameters, submit to Kaggle, or allocate a GPU. It defaults to a CPU
plan. Preparing all inputs and running the full reconstruction are separate
scope decisions.

## Prompts and reference frames

Create a manifest with schema `v2d.track1.preparation.v1`:

```json
{
  "schema": "v2d.track1.preparation.v1",
  "episodes": [
    {
      "episode_index": 16,
      "prompts_path": "prompts/episode_000016.json",
      "reference_frame": 0,
      "human_id": 0,
      "object_id": 1
    }
  ]
}
```

Prompt paths are relative to this manifest, or absolute. Use the existing
`Sam2Prompts` JSON structure; every prompt supplies `frame_index`, `object_id`,
and exactly one of `box`, `points` with `point_labels`, or `mask_path`. Boxes use
pixel coordinates `{ "x0": ..., "y0": ..., "x1": ..., "y1": ... }`; points use
`{ "x": ..., "y": ... }`. Optional `role` labels must match the configured IDs.
Exactly the two configured IDs must be covered. Mask paths are resolved relative
to their prompt file; masks must be binary and match the video dimensions.

Choose prompts by inspecting the actual video. Geometry validation cannot tell
whether the selected pixels depict the person or the correct object. The
reference frame should show enough of the object for reconstruction. SAM2 uses
its existing forward/backward propagation; prompt frames may differ from the
reference frame. CARI4D still needs a nonempty object mask in frame zero and a
nonempty human mask in every frame.

## Environment

Use the existing image with its already downloaded weights. Do not install the
model dependencies on the host. In the image, install the current checkout's
packages into the appropriate environments, keeping their existing ML versions:

```bash
python -m pip install --no-deps --no-build-isolation \
  -e reconstruction/modules/v2d_common -e reconstruction/modules/v2d_pipelines \
  -e reconstruction/modules/v2d_cari4d/lib -e reconstruction/modules/v2d_moge/lib \
  -e reconstruction/modules/v2d_mesh/lib
/opt/venvs/sam2/bin/python -m pip install --no-deps --no-build-isolation \
  -e reconstruction/modules/v2d_common -e reconstruction/modules/v2d_sam2/lib
/opt/venvs/sam3d/bin/python -m pip install --no-deps --no-build-isolation \
  -e reconstruction/modules/v2d_common -e reconstruction/modules/v2d_sam3d/lib
```

The base environment needs the CPU I/O dependencies declared by
`v2d-pipelines[track1]`. Its MoGe environment and the separate SAM3D environment
retain their respective dependencies. Execution checks the imported package
locations in each interpreter before starting models. `--runtime_python`,
`--sam2_python`, and `--sam3d_python` can override the interpreter paths.

The SAM2 weights directory must contain `sam2.1_hiera_large.pt`; configuration is
`configs/sam2.1/sam2.1_hiera_l.yaml`. SAM3D uses its existing weights layout,
including `hf_home` and `torch_home` underneath its weight root. Pass MoGe's
existing local checkpoint path to `--moge_weights`. Model commands run with
Hugging Face/Transformers offline mode enabled. The tool does not provision
missing model resources.

## Plan, execute, and resume

Run from a committed business checkout; output must be outside that checkout.
Fill the image version variables from the actual Pod deployment record. The
process records them but cannot independently inspect its host's image store.

```bash
python -m v2d.pipelines.track1_prepare \
  --dataset_root /vol/data/track_1 \
  --preparation_manifest /vol/inputs/track1/preparation.json \
  --output_dir /vol/outputs/track1/prepared-001 \
  --sam2_weights /vol/weights/sam2 \
  --moge_weights /vol/weights/moge/model.pt \
  --sam3d_weights /vol/weights/sam3d \
  --image_build_commit "$IMAGE_BUILD_COMMIT" --image_digest "$IMAGE_DIGEST" \
  --episodes 16
```

Without `--execute`, this checks metadata/prompts and writes only
`preparation_plan.json`. It neither loads models nor checks weight contents or
the GPU. Add `--execute` to prepare that subset. Full-roster preparation requires
omitting `--episodes` and explicitly passing `--execute --all_episodes`.

Each command has a default one-hour timeout, configurable with
`--stage_timeout_seconds`. The runner terminates the subprocess group on timeout
or cancellation. This is not a GPU billing limit; the deployment operator must
bound the resource lifetime and release the Pod afterwards.

Failures are isolated by episode. Rerun the same command, or select the failed
episodes, to retry. Completed stages are reused only when the identity, input
hashes and output hashes still match. Partial files from an unsuccessful stage
are replaced on retry. A changed business commit, runtime packages, weights,
source video, prompt/mask file or configuration requires a new output directory.
There is no automatic retry loop in this preparation command.

The report retains results across subset runs. Before publishing the input
manifest, old successes are checked, including episodes outside the current
selection; changed/missing artifacts become failures. A global identity error or
cancellation clears the consumable manifest while retaining diagnostics.

## Outputs and limits

```text
preparation_plan.json
preparation_identity.json
preparation_report.json
inputs.json
episode_XXXXXX/
  episode_XXXXXX.0.color.mp4
  prompts.json
  sam2/{human_id,object_id}.h5
  reference.png, reference_object.png
  depth.png, intrinsics.json
  object_raw.glb, sam3d_pose.json, sam3d_intrinsics.json
  object_simplified.glb, scale_transform.json, object_scaled.glb
  mesh_scale.json
  episode_XXXXXX_masks_k0.h5
  validation.json, mask_review.png, preparation.json
  .stages/..., logs/...
```

`inputs.json` contains only episodes that passed file/content checks. Review the
overlay image and static mesh before the full baseline. `semantic_review` stays
`required`: automated checks do not establish segmentation or reconstruction
accuracy. `mesh_scale.json` records the applied scale, its depth/mesh/pose
evidence and content hashes. This verifies provenance, not ground-truth physical
scale. Non-watertight mesh and occlusion warnings remain visible.

The report's verdict refers to the current selection; its summary describes the
whole dataset. Exit codes are `0` for a plan or passing selection, `2` for episode
failures, and nonzero for configuration errors or interruption.

## CPU verification

`tests/test_track1_prepare.py` uses real encoded videos, HDF5 streams, CPU mesh
simplification/transforms, the actual mask packer and the actual content
validator. Only SAM2, MoGe, SAM3D and rendered depth alignment are substitutes.
The tests cover stage recovery, subset resume, stale manifest removal, prompt
validation, input races and cancellation. They require the CPU-only
mesh functions from this checkout (`v2d-mesh-lib`, installed without ML
dependencies, plus its lightweight `pyglet` import) and do not establish GPU
model quality.
