"""CPU measurements of conversion residuals; no GPU decoding or reconstruction."""

from pathlib import Path
import zipfile

import numpy as np
import pytest

from v2d.common.track1_conversion import (
    Track1Scoring, track1_added_acceleration, track1_acceleration_summary,
    track1_validate_acceleration, track1_scoring, track1_vertex_spikes,
)


INDICES = tuple(range(22))  # Explicit synthetic joint ordering for arithmetic tests.


def test_constant_and_linear_residuals_do_not_add_acceleration():
    rng = np.random.default_rng(5)
    original = rng.normal(size=(8, 127, 3))
    converted = original + 3 + np.arange(8)[:, None, None] * 0.5
    report = track1_added_acceleration(original, converted, Track1Scoring(INDICES, tuple(range(8))),
                                      reference_cm=0.02)
    assert report["added_acc_h_cm"] < 1e-10
    assert report["within_reference"]


def test_quadratic_residual_uses_centimetres_per_frame_squared():
    original = np.zeros((7, 127, 3))
    converted = original.copy()
    # x = a*t^2/2 metres gives a metres/frame^2. No FPS^2 multiplier.
    converted[:, :, 0] = 0.0003 * np.arange(7)[:, None] ** 2 / 2
    report = track1_added_acceleration(original, converted, Track1Scoring(INDICES, tuple(range(7))),
                                      reference_cm=0.02)
    assert report["added_acc_h_cm"] == pytest.approx(0.03)
    assert report["center_frames"] == [1, 2, 3, 4, 5]
    assert not report["within_reference"]
    assert report["policy"] == "report_only"
    track1_validate_acceleration(report, Track1Scoring(INDICES, tuple(range(7))), 0.02)


def test_gaps_short_stretches_and_unscored_joints_do_not_create_jitter():
    original = np.zeros((15, 127, 3))
    converted = original.copy()
    converted[5:] = 100  # Huge jump across the gap must not enter the metric.
    converted[:, 126] = np.arange(15)[:, None] ** 2  # Not one of the 22 scored joints.
    scoring = Track1Scoring(INDICES, (0, 1, 2, 5, 6, 7, 8, 11, 13, 14))
    report = track1_added_acceleration(original, converted, scoring, reference_cm=0.02)
    assert report["added_acc_h_cm"] == 0
    assert report["center_frames"] == [1, 6, 7]
    assert [s["evaluated_triplets"] for s in report["stretches"]] == [1, 2, 0, 0]
    assert report["stretches"][-1]["added_acc_h_cm"] is None


def test_norm_is_taken_before_averaging_opposite_joint_accelerations():
    original = np.zeros((3, 127, 3))
    converted = original.copy()
    converted[2, :11, 0] = 0.001
    converted[2, 11:22, 0] = -0.001
    report = track1_added_acceleration(original, converted, Track1Scoring(INDICES, (0, 1, 2)),
                                      reference_cm=1)
    assert report["added_acc_h_cm"] == pytest.approx(0.1)


def test_global_mean_weights_triplets_not_stretches():
    scoring = Track1Scoring(INDICES, (0, 1, 2, 5, 6, 7, 8, 9))
    report = track1_acceleration_summary([0, 0.01, 0.01, 0.01], scoring, 0.02)
    assert report["added_acc_h_cm"] == pytest.approx(0.0075)
    assert report["stretches"][1]["added_acc_h_cm"] == pytest.approx(0.01)


def test_reference_comparison_is_strict_but_nonblocking_and_tampering_is_rejected():
    scoring = Track1Scoring(INDICES, (0, 1, 2))
    report = track1_acceleration_summary([0.02], scoring, 0.02)
    assert not report["within_reference"]
    track1_validate_acceleration(report, scoring, 0.02)
    report["within_reference"] = True
    with pytest.raises(ValueError, match="inconsistent"):
        track1_validate_acceleration(report, scoring, 0.02)


@pytest.mark.parametrize("frames", [(0, 1), (0, 2, 4)])
def test_no_complete_scored_triplet_is_not_a_zero_error_pass(frames):
    with pytest.raises(ValueError, match="at least three"):
        track1_added_acceleration(np.zeros((5, 127, 3)), np.zeros((5, 127, 3)),
                                 Track1Scoring(INDICES, frames), reference_cm=0.02)


@pytest.mark.parametrize("frames", [(1, 0, 2), (0, 1, 1), (-1, 0, 1), (True, 1, 2)])
def test_invalid_frame_indices_are_rejected(frames):
    with pytest.raises(ValueError, match="Scored frames"):
        Track1Scoring(INDICES, frames)


@pytest.mark.parametrize("bad", [np.nan, np.inf])
def test_nonfinite_joints_are_rejected_even_outside_scored_frames(bad):
    original = np.zeros((4, 127, 3))
    original[3, 126, 0] = bad
    with pytest.raises(ValueError, match="finite"):
        track1_added_acceleration(original, np.zeros_like(original), Track1Scoring(INDICES, (0, 1, 2)),
                                 reference_cm=0.02)


@pytest.mark.parametrize("threshold", [0, -1, float("nan"), float("inf"), None, True])
def test_invalid_reference(threshold):
    with pytest.raises(ValueError, match="reference"):
        track1_acceleration_summary([0], Track1Scoring(INDICES, (0, 1, 2)), threshold)


def test_real_kit_episode16_selects_official_22_joints_and_scored_span(tmp_path):
    root = Path(__file__).resolve().parents[1]
    with zipfile.ZipFile(root / "docs/v2d_challenge/assets/v2d_submission_kit.zip") as archive:
        archive.extractall(tmp_path)
    kit = tmp_path / "v2d_submission_kit"
    scoring = track1_scoring(kit, 16, 360)
    assert scoring.joint_indices == (1, 2, 18, 35, 3, 19, 36, 4, 20, 37, 8, 24, 110, 74, 38, 113, 75, 39, 76, 40, 78, 42)
    assert scoring.frames == tuple(range(50, 290))
    assert scoring.centers() == list(range(51, 289))
    with pytest.raises(ValueError, match="exceed"):
        track1_scoring(kit, 16, 289)


def test_spikes_distinguish_scored_and_unscored_frames():
    errors = np.full(21, 0.2)
    errors[[0, 10, 20]] = 2
    report = track1_vertex_spikes(errors, Track1Scoring(INDICES, tuple(range(5, 16))))
    assert report["spike_frames"] == [0, 10, 20]
    assert report["scored_spike_frames"] == [10]
    assert report["unscored_spike_frames"] == [0, 20]
    assert report["scored_spike_count"] == 1
    assert report["no_scored_spikes"] is False
    assert report["policy"] == "report_only"
    np.testing.assert_allclose(np.asarray(report["local_median_mm"])[[0, 10, 20]], 0.2)


@pytest.mark.parametrize("background,peak,detected", [(0.5, 1.5, False), (0.5, 1.51, True),
                                                     (0.2, 1.0, False), (0.2, 1.01, True),
                                                     (0.0, 1.01, True)])
def test_spike_requires_both_strict_thresholds(background, peak, detected):
    errors = np.full(11, background)
    errors[5] = peak
    report = track1_vertex_spikes(errors, Track1Scoring(INDICES, tuple(range(11))))
    assert report["spike_frames"] == ([5] if detected else [])


def test_spike_window_excludes_center_and_includes_unscored_neighbors():
    # The scored frame can use its unscored neighbor; including itself would
    # incorrectly raise the median to 1.1 mm and hide this spike.
    report = track1_vertex_spikes([2, 0.2], Track1Scoring(INDICES, (0,)))
    assert report["local_median_mm"] == [0.2, 2.0]
    assert report["scored_spike_frames"] == [0]


def test_sustained_error_is_not_a_local_spike():
    report = track1_vertex_spikes(np.full(15, 2.0), Track1Scoring(INDICES, tuple(range(15))))
    assert report["spike_frames"] == []
    assert report["no_scored_spikes"] is True


@pytest.mark.parametrize("errors", [[0], [0, np.nan, 0], [0, np.inf, 0], [0, -1, 0], [[0, 0, 0]]])
def test_invalid_spike_errors_cannot_be_reported_as_clear(errors):
    with pytest.raises(ValueError, match="Spikes require"):
        track1_vertex_spikes(errors, Track1Scoring(INDICES, (0, 1, 2)))
