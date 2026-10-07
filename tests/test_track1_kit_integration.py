"""Host integration with the repository's actual kit; no GPU or MHR assets.

Requires numpy, scipy, trimesh, pytest, pandas, pyarrow and fast_simplification.
Kept outside the CARI4D-only container tests, which do not mount docs/assets.
"""

import importlib.util
from pathlib import Path
import subprocess
import sys
import zipfile

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
CONTRACT_TEST = ROOT / "reconstruction/modules/v2d_cari4d/tests/test_track1_export.py"
SPEC = importlib.util.spec_from_file_location("track1_export_contracts", CONTRACT_TEST)
contracts = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(contracts)
export_inputs = contracts.export_inputs


def test_official_packer_accepts_exported_contract(export_inputs, monkeypatch, tmp_path):
    # Real packer, synthetic fitted parameters: schema/coordinate test only.
    with monkeypatch.context() as patch:
        patch.setattr(contracts.adapter.subprocess, "run", contracts.simulate_children)
        out = contracts.adapter.export_track1(**export_inputs)
    with zipfile.ZipFile(ROOT / "docs/v2d_challenge/assets/v2d_submission_kit.zip") as archive:
        archive.extractall(tmp_path / "official")
    kit = tmp_path / "official/v2d_submission_kit"
    # A synthetic roster with gaps checks indexing by source-video frame.
    rows = []
    for array, count, frames in [(0, 46, [0, 2]), (1, 23, [999999]), (2, 15, [999999]),
                                  (3, 3, [0, 2]), (4, 1, [0, 2]), (5, 1, [999999]),
                                  (6, 8, [999999]), (7, 12, [999999])]:
        rows.extend(f"t1_000016_{frame:06d}_{array}_{point:06d}"
                    for frame in frames for point in range(count))
    sample = tmp_path / "sample.parquet"
    pd.DataFrame({"row_id": rows}).to_parquet(sample)
    packed = tmp_path / "submission.parquet"
    subprocess.run([sys.executable, str(kit / "eval_reconstruction.py"), "--episodes", str(out),
                    "--sample", str(sample), "--commit", "https://github.com/example/repo/commit/" + "a" * 40,
                    "--out", str(packed)], check=True, capture_output=True, text=True)
    table = pd.read_parquet(packed)
    assert table["row_id"].tolist() == rows
    assert np.isfinite(table[["x", "y", "z"]]).all().all()
    translation = table[table.row_id == "t1_000016_000002_4_000000"][["x", "y", "z"]].to_numpy()[0]
    np.testing.assert_allclose(translation, [0.2, -0.4, 1.5])
