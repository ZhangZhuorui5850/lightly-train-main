from __future__ import annotations

from tool_lib.det_optimize import (
    _adaptive_min_confusion_count,
    _hungarian_match,
    analyze_class_quality,
    find_merge_candidates,
)


# ---------------------------------------------------------------------------
# Original tests (preserved)
# ---------------------------------------------------------------------------

def test_find_merge_candidates_includes_strong_one_way_confusion() -> None:
    confusion_matrix = {
        "class_ids": [0, 1],
        "class_names": {"0": "电镐", "1": "冲击钻"},
        "matrix": [
            [0, 8],
            [0, 10],
        ],
        "total_gt_by_class": {0: 8, 1: 10},
    }

    candidates = find_merge_candidates(confusion_matrix, threshold=0.15)

    assert len(candidates) == 1
    candidate = candidates[0]
    assert candidate["confusion_type"] == "单向"
    assert candidate["class_id_a"] == 0
    assert candidate["class_id_b"] == 1
    assert candidate["rate_a_to_b"] == 1.0
    assert candidate["rate_b_to_a"] == 0.0


def test_find_merge_candidates_keeps_bidirectional_confusion() -> None:
    confusion_matrix = {
        "class_ids": [0, 1],
        "class_names": {"0": "电镐", "1": "冲击钻"},
        "matrix": [
            [10, 3],
            [4, 10],
        ],
        "total_gt_by_class": {0: 13, 1: 14},
    }

    candidates = find_merge_candidates(confusion_matrix, threshold=0.15)

    assert len(candidates) == 1
    assert candidates[0]["confusion_type"] == "双向"


# ---------------------------------------------------------------------------
# P0-1: Risk scoring with small samples
# ---------------------------------------------------------------------------

class TestAnalyzeClassQuality:
    def _make_ap_entry(self, name="cls", ap=0.5, gt=10, pred=10, tp=5, precision=0.5, recall=0.5):
        return {
            "name": name, "ap": ap, "gt": gt, "pred": pred,
            "tp": tp, "precision": precision, "recall": recall,
        }

    def test_gt_zero_extreme_risk(self):
        per_class_ap = {"0": self._make_ap_entry(gt=0, pred=0, tp=0, precision=0, recall=0)}
        result = analyze_class_quality(per_class_ap)
        assert result[0]["risk"] == 1.0
        assert result[0]["risk_level"] == "极高"
        assert result[0]["confidence"] == "none"

    def test_gt_1_high_risk_no_ap_used(self):
        """GT=1 with perfect AP -> still high risk (unreliable)."""
        per_class_ap = {"0": self._make_ap_entry(ap=0.99, gt=1, pred=1, tp=1, precision=1.0, recall=1.0)}
        result = analyze_class_quality(per_class_ap)
        assert result[0]["confidence"] == "low"
        assert result[0]["risk"] >= 0.6

    def test_gt_4_no_pred_extra_penalty(self):
        per_class_ap = {"0": self._make_ap_entry(gt=4, pred=0, tp=0, precision=0, recall=0)}
        result = analyze_class_quality(per_class_ap)
        assert result[0]["confidence"] == "low"
        base_risk = 0.6 + 0.4 * (1.0 - 4 / 5)
        assert result[0]["risk"] >= base_risk

    def test_gt_5_normal_formula(self):
        per_class_ap = {"0": self._make_ap_entry(ap=0.8, gt=5, pred=5, tp=4, precision=0.8, recall=0.8)}
        result = analyze_class_quality(per_class_ap)
        assert result[0]["confidence"] == "normal"
        assert result[0]["risk"] < 0.3

    def test_small_sample_wins_over_perfect_metrics(self):
        per_class_ap = {
            "0": self._make_ap_entry(name="small", ap=1.0, gt=1, pred=1, tp=1, precision=1.0, recall=1.0),
            "1": self._make_ap_entry(name="large", ap=0.8, gt=100, pred=100, tp=80, precision=0.8, recall=0.8),
        }
        result = analyze_class_quality(per_class_ap)
        small = [r for r in result if r["name"] == "small"][0]
        large = [r for r in result if r["name"] == "large"][0]
        assert small["risk"] > large["risk"]


# ---------------------------------------------------------------------------
# P0-2: Hungarian matching
# ---------------------------------------------------------------------------

class TestHungarianMatch:
    def test_empty_matrix(self):
        assert _hungarian_match([]) == []

    def test_single_pair(self):
        assert _hungarian_match([[-0.8]]) == [(0, 0)]

    def test_two_by_two_optimal(self):
        cost = [[-0.9, -0.1], [-0.2, -0.8]]
        matches = _hungarian_match(cost)
        assert set(matches) == {(0, 0), (1, 1)}

    def test_two_by_two_swap(self):
        cost = [[-0.1, -0.9], [-0.8, -0.2]]
        matches = _hungarian_match(cost)
        assert set(matches) == {(0, 1), (1, 0)}

    def test_inf_forbidden_pair(self):
        INF = float('inf')
        cost = [[-0.9, INF], [INF, -0.8]]
        matches = _hungarian_match(cost)
        assert set(matches) == {(0, 0), (1, 1)}

    def test_rectangular_more_rows(self):
        cost = [[-0.9, -0.1], [-0.2, -0.8], [-0.5, -0.3]]
        matches = _hungarian_match(cost)
        assert len(matches) == 2
        assert {m[1] for m in matches} == {0, 1}

    def test_all_inf_no_match(self):
        INF = float('inf')
        matches = _hungarian_match([[INF, INF], [INF, INF]])
        assert matches == []

    def test_three_by_three(self):
        cost = [[-0.5, -0.1, -0.2], [-0.1, -0.9, -0.3], [-0.2, -0.3, -0.8]]
        matches = _hungarian_match(cost)
        assert set(matches) == {(0, 0), (1, 1), (2, 2)}


# ---------------------------------------------------------------------------
# P0-3: Adaptive minimum confusion count
# ---------------------------------------------------------------------------

class TestAdaptiveMinConfusionCount:
    def test_small_dataset_floor(self):
        assert _adaptive_min_confusion_count(10) == 3

    def test_medium_dataset(self):
        assert _adaptive_min_confusion_count(200) == 4

    def test_large_dataset(self):
        assert _adaptive_min_confusion_count(500) == 10

    def test_very_small_dataset(self):
        assert _adaptive_min_confusion_count(1) == 3


class TestFindMergeCandidatesP03:
    def _make_cm(self, class_ids, matrix, total_gt, class_names):
        return {
            "class_ids": class_ids, "matrix": matrix,
            "total_gt_by_class": total_gt,
            "class_names": {str(k): v for k, v in class_names.items()},
            "processed_images": 100,
        }

    def test_small_sample_filtered_out(self):
        cm = self._make_cm([0, 1], [[0, 1], [0, 0]], {0: 2, 1: 5}, {0: "A", 1: "B"})
        assert len(find_merge_candidates(cm, threshold=0.15)) == 0

    def test_sufficient_confusion_passes(self):
        cm = self._make_cm([0, 1], [[0, 20], [0, 0]], {0: 100, 1: 100}, {0: "A", 1: "B"})
        candidates = find_merge_candidates(cm, threshold=0.15)
        assert len(candidates) == 1
        assert candidates[0]["count_a_to_b"] == 20

    def test_high_rate_low_count_filtered(self):
        cm = self._make_cm([0, 1], [[0, 1], [0, 0]], {0: 3, 1: 3}, {0: "A", 1: "B"})
        assert len(find_merge_candidates(cm, threshold=0.15)) == 0

    def test_large_dataset_high_threshold(self):
        cm = self._make_cm([0, 1], [[0, 15], [0, 0]], {0: 1000, 1: 1000}, {0: "A", 1: "B"})
        # min_count for GT=1000 is 20, confusion=15 < 20
        assert len(find_merge_candidates(cm, threshold=0.01)) == 0

    def test_large_dataset_enough_confusion(self):
        cm = self._make_cm([0, 1], [[0, 25], [0, 0]], {0: 1000, 1: 1000}, {0: "A", 1: "B"})
        candidates = find_merge_candidates(cm, threshold=0.01)
        assert len(candidates) == 1
