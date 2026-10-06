# Initial prompts for the Track 1 baseline

`preparation.json` covers the 30 released Track 1 episodes. Each episode uses
frame zero as the reference, with human ID 0 and object ID 1. The boxes use
pixel coordinates in the original 1536 × 1152 image. They were selected by
inspecting the real reference images and magnified object crops, including the
main interacting person where background people are present.

`source_review.json` binds these annotations to the reviewed source video and
prompt contents by SHA-256. The source media are not included here. Verify the
dataset against this record before using the prompts with a different copy.
These are initial segmentation prompts, not ground-truth masks or accepted
reconstructions. Inspect the generated overlay images and static meshes before
the full baseline, especially thin hoops, pan/roller handles and occluded cart
legs. A geometrically valid box does not prove that SAM2 followed the right
object throughout the video.

Pass this manifest to the [preparation runner](../../docs/track1_prepare.md):

```bash
python -m v2d.pipelines.track1_prepare \
  --preparation_manifest reconstruction/configs/track1_inputs/preparation.json \
  --dataset_root /vol/data/track_1 \
  --output_dir /vol/outputs/track1/prepared-001 \
  --sam2_weights /vol/weights/sam2 \
  --moge_weights "$MOGE_CHECKPOINT" \
  --sam3d_weights /vol/weights/sam3d \
  --image_build_commit "$IMAGE_BUILD_COMMIT" --image_digest "$IMAGE_DIGEST"
```

This command only creates a plan. Resolve weight paths and image provenance from
the actual deployment. Adding `--execute --episodes 16` prepares one episode;
full-roster input preparation requires `--execute --all_episodes`. Neither
command starts temporal CARI4D reconstruction or submits results to Kaggle.
