"""尺寸感知选图：纯函数打分 + 分批贪心 + split 尺寸目标。"""
from __future__ import annotations

from pathlib import Path

from tool_lib.det_analysis import (
    ratio_bucket_props,
    size_deficit_score,
    density_steer_term,
    select_balanced_train_candidates,
)
from tool_lib.det_shared import ExportImageCandidate


def test_ratio_bucket_props_excludes_tiny():
    # tiny 不进分母：small=2 medium=1 large=1 tiny=10 → small 0.5
    props = ratio_bucket_props({"tiny": 10, "small": 2, "medium": 1, "large": 1})
    assert props["small"] == 0.5
    assert props["medium"] == 0.25
    assert props["large"] == 0.25


def test_size_deficit_score_rewards_deficit_bucket():
    target = {"small": 0.3, "medium": 0.4, "large": 0.3}
    # 当前全是 large → small/medium 有赤字
    current = {"tiny": 0, "small": 0, "medium": 0, "large": 10}
    cand_small = {"tiny": 0, "small": 3, "medium": 0, "large": 0}
    cand_large = {"tiny": 0, "small": 0, "medium": 0, "large": 3}
    s_small = size_deficit_score(cand_small, current, target, over_penalty=1.0)
    s_large = size_deficit_score(cand_large, current, target, over_penalty=1.0)
    assert s_small > s_large  # 缺小目标时小目标图得分更高


def test_density_steer_term_direction():
    # 平均偏低(2)→鼓励多框图(+)，偏高(20)→鼓励少框图
    low = density_steer_term(total_boxes=8, current_avg=2.0, lo=5.0, hi=15.0)
    high = density_steer_term(total_boxes=8, current_avg=20.0, lo=5.0, hi=15.0)
    inband = density_steer_term(total_boxes=8, current_avg=10.0, lo=5.0, hi=15.0)
    assert low > 0.0
    assert high < 0.0
    assert inband == 0.0


def _cand(stem, class_box_counts):
    return ExportImageCandidate(
        split_name="train",
        rel_split_image_dir=Path("images/train"),
        rel_split_label_dir=Path("labels/train"),
        rel_path=Path(f"{stem}.jpg"),
        src_image_path=Path(f"/src/{stem}.jpg"),
        src_label_path=Path(f"/src/{stem}.txt"),
        filtered_lines=tuple(),
        class_box_counts=class_box_counts,
    )


def test_size_aware_selection_moves_ratio_toward_target():
    # 10 张大目标图 + 10 张小目标图，目标小占比高 → 应多选小目标图
    cands, buckets = [], {}
    for i in range(10):
        c = _cand(f"big{i}", {0: 3})
        cands.append(c)
        buckets[c.rel_path.as_posix() + "|train"] = {"tiny": 0, "small": 0, "medium": 0, "large": 3}
    for i in range(10):
        c = _cand(f"small{i}", {0: 3})
        cands.append(c)
        buckets[c.rel_path.as_posix() + "|train"] = {"tiny": 0, "small": 3, "medium": 0, "large": 0}

    selected, summary = select_balanced_train_candidates(
        candidates=cands,
        kept_class_ids=[0],
        target_total_images=10,
        target_boxes_per_class=0,
        balance_ratio=0.0,
        target_images_per_class=0,
        box_density_penalty=0.0,
        size_buckets_by_candidate=buckets,
        target_size_ratio={"small": 0.7, "medium": 0.0, "large": 0.3},
        size_balance_weight=1.0,
    )
    n_small = sum(1 for c in selected if c.rel_path.as_posix().startswith("small"))
    assert n_small >= 6  # 偏向小目标
    assert summary["selection_algorithm"] == "size_aware_greedy"


def test_size_ratio_off_uses_legacy_celf():
    c = _cand("a", {0: 2})
    selected, summary = select_balanced_train_candidates(
        candidates=[c],
        kept_class_ids=[0],
        target_total_images=0,
        target_boxes_per_class=0,
        balance_ratio=0.0,
        target_images_per_class=0,
        box_density_penalty=0.0,
        size_buckets_by_candidate=None,
        target_size_ratio=None,
        size_balance_weight=0.0,
    )
    assert summary["selection_algorithm"] == "celf_lazy_greedy"
