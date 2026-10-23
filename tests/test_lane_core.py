"""Small CPU tests for lane_core's deterministic helpers.

Run from the repository root with: python -m pytest -q
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src" / "lane_detection"))
import lane_core as lc  # noqa: E402


def test_make_labels_with_no_lanes_returns_no_lane_class():
    labels = lc.make_labels([])
    assert labels.shape == (lc.NUM_ROWS, lc.NUM_LANES)
    assert labels.dtype == np.int64
    assert np.all(labels == lc.GRID)


def test_f1_counts_empty_prediction_and_ground_truth():
    assert lc.f1_counts([], []) == (0, 0, 0)


def test_f1_counts_missing_predictions_counts_false_negatives():
    gt = [np.array([[100.0, 150.0], [110.0, 200.0], [120.0, 250.0]])]
    assert lc.f1_counts([], gt) == (0, 0, 1)


def test_f1_score_handles_zero_denominator():
    assert lc.f1_score(0, 0, 0) == 0.0


def test_f1_score_computes_micro_f1_from_counts():
    assert np.isclose(lc.f1_score(3, 1, 2), 0.5)


def test_decode_all_no_lane_logits_returns_no_lanes():
    logits = np.zeros((lc.GRID + 1, lc.NUM_ROWS, lc.NUM_LANES), dtype=np.float32)
    logits[lc.GRID] = 10.0
    assert lc.decode(logits) == []
