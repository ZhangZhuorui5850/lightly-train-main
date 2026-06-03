"""尺寸感知选图：纯函数打分 + 分批贪心 + split 尺寸目标。"""
from __future__ import annotations

from tool_lib.det_analysis import (
    ratio_bucket_props,
    size_deficit_score,
    density_steer_term,
)


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
