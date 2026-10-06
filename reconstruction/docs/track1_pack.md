# Collect and pack Track 1 exports

`v2d.pipelines.track1_pack` collects accepted episode exports, invokes the
**unchanged official** `eval_reconstruction.py`, and checks the resulting Parquet
against the full official sample roster. It needs no GPU and never submits to
Kaggle. Install the host dependencies from `reconstruction/`:

```bash
pip install -e modules/v2d_common -e 'modules/v2d_pipelines[track1]'
```

Keep the official submission kit unpacked from the tracked
`docs/v2d_challenge/assets/v2d_submission_kit.zip`. Run after episode exports are
available under a single experiment root:

The [batch runner](track1_batch.md) writes this layout under its `exports/`
directory and records failures separately.

```text
exports/run-001/
  episode_000000/
    episode_000000.npz
    episode_000000_object.glb
    episode_000000_export.json
  episode_000001/...
  ...
```

```bash
python -m v2d.pipelines.track1_pack \
  --dataset_root /data/track_1 \
  --export_root /data/exports/run-001 \
  --submission_kit /data/v2d_submission_kit \
  --output_dir /data/packages/run-001 \
  --code_commit_url "https://github.com/Beardear/video_to_data/commit/$BUSINESS_COMMIT"
```

`BUSINESS_COMMIT` must be the actual inference business commit recorded in all
episode reports, not a guessed branch head. Every official episode is required
by default. The official roster and dataset metadata must agree, and each export
must cover all its original frames. Reports without explicit acceptance, changed
files, invalid arrays, or mixtures of inference settings, image versions,
converter/decoder identities and export settings are rejected before packing.
The current baseline uses the explicit `report` conversion policy described in
[the export adapter](track1_export.md); diagnostic files do not become accepted
exports simply by moving them into this directory.

The output directory is published only after successful packing and validation:

```text
episodes/             # flat copies of NPZ, GLB, and export reports
sample.parquet        # exact row IDs used by this invocation
submission.parquet
packing_report.json   # hashes, provenance, row/episode counts, and scope
pack.log
```

The official packer performs mesh budgeting. The collector does not implement
its own mesh decimation or rewrite the official row format. It verifies every
row ID in order, finite coordinates and the commit URL, and rejects input,
metadata, or official-kit changes during packing. Existing output directories
are refused. Failed runs retain an unpublished sibling directory with the log
and any partial output for diagnosis.

For a limited **format check**, append `--episodes 16`. Such packages are marked
`subset_format_check`, even when all explicitly selected episodes succeeded.
They must not be presented as a full submission. No mode generates a Kaggle
score, and a successful format check does not establish reconstruction accuracy.

## CPU verification

From the repository root, with the host packages and `pytest` installed:

```bash
python -m pytest tests/test_track1_pack.py -q
```

The integration test uses the actual complete 30-episode, 740,780-row official
roster with **synthetic reconstructions**, plus failure cases for missing,
modified and mixed-version exports. This validates packaging without running a
full video baseline. Real-data GPU acceptance is a separate experiment.
