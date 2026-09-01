"""尺寸感知选图：纯函数打分 + 分批贪心 + split 尺寸目标。"""
from __future__ import annotations

from pathlib import Path

from tool_lib.det_analysis import (
    ratio_bucket_props,
    size_deficit_score,
    density_steer_term,
    select_balanced_train_candidates,
    allocate_size_bucket_targets_per_split,
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


def test_balanced_selection_always_emits_terminal_progress_event():
    candidate = _cand("only", {0: 1})
    events: list[tuple[int, int, str]] = []

    select_balanced_train_candidates(
        candidates=[candidate],
        kept_class_ids=[0],
        target_total_images=1,
        target_boxes_per_class=1,
        balance_ratio=1.0,
        target_images_per_class=1,
        box_density_penalty=0.0,
        progress_callback=lambda current, total, detail: events.append(
            (current, total, detail)
        ),
    )

    assert events
    assert events[-1][0] == events[-1][1]
    assert "phase=complete" in events[-1][2]


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


def test_split_size_targets_proportional_and_exclude_tiny():
    cands = [_cand(f"c{i}", {0: 1}) for i in range(10)]
    buckets = {}
    for i, c in enumerate(cands):
        # 每图 1 个 small 框；外加前 5 张各 1 个 tiny（tiny 不计目标）
        b = {"tiny": 1 if i < 5 else 0, "small": 1, "medium": 0, "large": 0}
        buckets[c.rel_path.as_posix() + "|train"] = b
    targets = allocate_size_bucket_targets_per_split(
        selected_candidates=cands,
        size_buckets_by_candidate=buckets,
        split_image_targets={"train": 8, "val": 1, "test": 1},
    )
    # small 总数 10，按 8:1:1 → train 8 / val 1 / test 1
    assert targets["train"]["small"] == 8
    assert targets["val"]["small"] == 1
    assert targets["test"]["small"] == 1
    # tiny 不进目标
    assert "tiny" not in targets["train"]


def test_split_assignment_balances_size_across_splits():
    # 复现并防回归：全局池小目标密集图 + 大目标稀疏图各半，划分后
    # 每个 split 的大目标占比都该接近全局（≈0.14），而不是大目标全漏进 val/test。
    from tool_lib.det_analysis import (
        assign_candidates_to_new_splits,
        allocate_box_targets_per_split,
    )

    cands, buckets = [], {}
    for i in range(20):
        c = _cand(f"sd{i:02d}", {0: 6})  # small-dense：每图 6 个小目标
        cands.append(c)
        buckets[c.rel_path.as_posix() + "|train"] = {"tiny": 0, "small": 6, "medium": 0, "large": 0}
    for i in range(20):
        c = _cand(f"ls{i:02d}", {0: 1})  # large-sparse：每图 1 个大目标
        cands.append(c)
        buckets[c.rel_path.as_posix() + "|train"] = {"tiny": 0, "small": 0, "medium": 0, "large": 1}

    split_image_targets = {"train": 32, "val": 4, "test": 4}
    desired_box = allocate_box_targets_per_split(
        selected_candidates=cands,
        split_image_targets=split_image_targets,
        kept_class_ids=[0],
    )
    assigned, _summary = assign_candidates_to_new_splits(
        selected_candidates=cands,
        split_image_targets=split_image_targets,
        desired_box_targets_by_split=desired_box,
        kept_class_ids=[0],
        size_buckets_by_candidate=buckets,
    )

    def large_frac(split: str) -> float:
        large = sum(buckets[c.rel_path.as_posix() + "|train"]["large"] for c in assigned[split])
        small = sum(buckets[c.rel_path.as_posix() + "|train"]["small"] for c in assigned[split])
        denom = large + small
        return large / denom if denom else 0.0

    fracs = {s: large_frac(s) for s in ("train", "val", "test")}
    # 全局大目标占比≈0.14。旧实现把大目标稀疏图偏流给 val/test（≈0.33）。
    # 修复后每个 split 都应接近全局，val/test 大目标占比不超过 0.25。
    assert fracs["val"] <= 0.25, fracs
    assert fracs["test"] <= 0.25, fracs


def test_scan_candidate_size_buckets(tmp_path):
    from pathlib import Path as _P
    from tool_lib import common as rt
    from tool_lib.det_export import scan_candidate_size_buckets
    from tool_lib.det_shared import ExportImageCandidate
    rt.import_runtime_dependencies()
    img = tmp_path / "a.jpg"
    rt.Image.new("RGB", (100, 100), (127, 127, 127)).save(img)
    # 一个 small 框(20x20=400px∈[256,1024)) + 一个 large 框(100x100=10000px≥9216)
    cand = ExportImageCandidate(
        split_name="train",
        rel_split_image_dir=_P("images/train"),
        rel_split_label_dir=_P("labels/train"),
        rel_path=_P("a.jpg"),
        src_image_path=img,
        src_label_path=tmp_path / "a.txt",
        filtered_lines=("0 0.5 0.5 0.2 0.2", "0 0.5 0.5 1.0 1.0"),
        class_box_counts={0: 2},
    )
    out = scan_candidate_size_buckets([cand], {0: 0}, cache_dir=tmp_path)
    key = "a.jpg|train"
    assert out[key]["small"] == 1
    assert out[key]["large"] == 1
    assert out[key]["tiny"] == 0
