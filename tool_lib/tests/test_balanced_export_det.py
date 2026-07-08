"""det 类别均衡导出回归保护。

验证收紧后的 auto_balance 行为：
- 尾部（少数）类别不再被 min_class_* 阈值删掉；
- 多数类被降采样向尾部靠拢，导出后各类框数的 max/min 落在均衡窗口内。
"""
from __future__ import annotations

import json
from pathlib import Path

from tool_lib import common as rt
from tool_lib.det_analysis import trim_selected_candidates_to_quota
from tool_lib.det_export import export_filtered_dataset
from tool_lib.det_shared import ExportImageCandidate


def _cand(name: str, class_box_counts: dict[int, int]) -> ExportImageCandidate:
    lines = tuple(
        f"{cid} 0.5 0.5 0.4 0.4"
        for cid, cnt in class_box_counts.items()
        for _ in range(cnt)
    )
    return ExportImageCandidate(
        split_name="train", rel_split_image_dir=Path("images/train"),
        rel_split_label_dir=Path("labels/train"), rel_path=Path(f"{name}.jpg"),
        src_image_path=Path(f"{name}.jpg"), src_label_path=Path(f"{name}.txt"),
        filtered_lines=lines, class_box_counts=dict(class_box_counts),
    )


def test_trim_brings_dominant_class_into_window():
    # 主导类 0 作乘客分布在很多图里；稀有类 1 只有少量框。
    cands = [_cand(f"d{i}", {0: 5, 1: 1}) for i in range(20)]
    # caps：类0 上限 20，类1 不限（保留全部）。
    trimmed, summary = trim_selected_candidates_to_quota(cands, [0, 1], {0: 20, 1: 999})
    realized = {0: 0, 1: 0}
    for c in trimmed:
        for cid, n in c.class_box_counts.items():
            realized[cid] += n
    assert realized[0] == 20, realized          # 主导类被裁到窗口上限
    assert realized[1] == 20, realized          # 稀有类一框不删
    assert summary["trimmed_total"] == 80       # 100 - 20
    # 裁剪是“削最高”：各图类0框数被摊平到 0~1，不会出现某图仍有 5 个。
    max_c0_per_image = max(c.class_box_counts.get(0, 0) for c in trimmed)
    assert max_c0_per_image <= 1, max_c0_per_image


def _make_imbalanced_dataset(root: Path, per_class_images: dict[int, int]):
    """每张图 1 个框、单类别，按 per_class_images 生成一个高度不均衡的数据集。"""
    rt.import_runtime_dependencies()
    for split in ("train", "val", "test"):
        (root / "images" / split).mkdir(parents=True, exist_ok=True)
        (root / "labels" / split).mkdir(parents=True, exist_ok=True)
    for class_id, count in per_class_images.items():
        for i in range(count):
            stem = f"c{class_id}_{i}"
            rt.Image.new("RGB", (100, 100), (127, 127, 127)).save(
                root / "images" / "train" / f"{stem}.jpg"
            )
            (root / "labels" / "train" / f"{stem}.txt").write_text(
                f"{class_id} 0.5 0.5 0.4 0.4\n", encoding="utf-8"
            )
    names = [f"obj{cid}" for cid in sorted(per_class_images)]
    data_yaml = root / "data.yaml"
    rt.dump_yaml(data_yaml, {
        "path": str(root.resolve()), "train": "images/train",
        "val": "images/val", "test": "images/test",
        "task": "detect", "nc": len(names), "names": names,
    })
    return data_yaml


def test_auto_balance_keeps_tail_classes_and_bounds_imbalance(tmp_path):
    # 200:150:80:30:12 —— 最多类是最少类的 ~16.7 倍。
    per_class_images = {0: 200, 1: 150, 2: 80, 3: 30, 4: 12}
    data_yaml = _make_imbalanced_dataset(tmp_path / "ds", per_class_images)
    neutral = {
        "config": {"data_root": str((tmp_path / "ds").resolve())},
        "summary": {}, "per_class_ap": {},
    }
    out = export_filtered_dataset(
        data_yaml, neutral, 0.0, "_BAL",
        auto_balance=True, auto_relax_class_threshold=True, balance_ratio=0.0,
        min_class_images=0, min_class_boxes=0, target_images_per_class=0,
        target_total_images=0, split_ratio="8:1:1", target_boxes_per_class=0,
        max_boxes_per_image=0, max_boxes_per_class_per_image=0, box_density_penalty=0.0,
        size_ratio="",
    )
    summary = json.loads((out / "export_summary.json").read_text(encoding="utf-8"))

    # 1) 没有类别被删（5 个全保留）。
    assert len(summary["selected_class_ids"]) == 5, summary["selected_class_ids"]
    assert not summary["dropped_classes"], summary["dropped_classes"]

    # 2) 尾部类别（原 id=4）真实出现在导出里。
    boxes_per_class = {
        int(cid): info["boxes"]
        for cid, info in summary["exported_class_summary"].items()
    }
    assert boxes_per_class[4] > 0
    assert all(v > 0 for v in boxes_per_class.values()), boxes_per_class

    # 3) 导出后各类框数的 max/min 落在均衡窗口内（auto 上限 4×，留少量松弛）。
    hi, lo = max(boxes_per_class.values()), min(boxes_per_class.values())
    assert lo > 0 and hi / lo <= 4.5, boxes_per_class

    # 4) 多数类确实被降采样（不是把全部 200 框照搬）。
    assert boxes_per_class[0] < per_class_images[0], boxes_per_class


def _make_cooccurring_dataset(root: Path):
    """主导类 0 在多数图里作乘客共现；类 1..4 各自带类 0。"""
    rt.import_runtime_dependencies()
    for split in ("train", "val", "test"):
        (root / "images" / split).mkdir(parents=True, exist_ok=True)
        (root / "labels" / split).mkdir(parents=True, exist_ok=True)
    idx = 0

    def write(lines: list[str]):
        nonlocal idx
        stem = f"img{idx}"
        rt.Image.new("RGB", (100, 100), (127, 127, 127)).save(
            root / "images" / "train" / f"{stem}.jpg"
        )
        (root / "labels" / "train" / f"{stem}.txt").write_text(
            "\n".join(lines) + "\n", encoding="utf-8"
        )
        idx += 1

    # 120 张主导类密集图（class 0，每图 4 个框）
    for _ in range(120):
        write(["0 0.5 0.5 0.4 0.4"] * 4)
    # 类 1..4：每类 40 张，每张含 2 个该类框 + 5 个主导类乘客（重度共现）
    for cls in (1, 2, 3, 4):
        for _ in range(40):
            write(["0 0.3 0.3 0.3 0.3"] * 5 + [f"{cls} 0.6 0.6 0.3 0.3"] * 2)
    names = ["dom", "a", "b", "c", "d"]
    data_yaml = root / "data.yaml"
    rt.dump_yaml(data_yaml, {
        "path": str(root.resolve()), "train": "images/train",
        "val": "images/val", "test": "images/test",
        "task": "detect", "nc": len(names), "names": names,
    })
    return data_yaml


def test_trim_boxes_balances_cooccurring_dominant(tmp_path):
    data_yaml = _make_cooccurring_dataset(tmp_path / "ds")
    neutral = {
        "config": {"data_root": str((tmp_path / "ds").resolve())},
        "summary": {}, "per_class_ap": {},
    }
    kwargs = dict(
        auto_balance=True, auto_relax_class_threshold=True, balance_ratio=4.0,
        min_class_images=0, min_class_boxes=0, target_images_per_class=0,
        target_total_images=120, split_ratio="8:1:1", target_boxes_per_class=0,
        max_boxes_per_image=0, max_boxes_per_class_per_image=0, box_density_penalty=0.3,
        size_ratio="",
    )
    out_notrim = export_filtered_dataset(data_yaml, neutral, 0.0, "_NOTRIM", trim_boxes=False, **kwargs)
    out_trim = export_filtered_dataset(data_yaml, neutral, 0.0, "_TRIM", trim_boxes=True, **kwargs)

    def boxes(out):
        s = json.loads((out / "export_summary.json").read_text(encoding="utf-8"))
        return {int(k): v["boxes"] for k, v in s["exported_class_summary"].items()}, s

    b_no, s_no = boxes(out_notrim)
    b_tr, s_tr = boxes(out_trim)

    # 不裁剪：主导类作乘客仍严重超量（共现物理底）。
    ratio_no = max(b_no.values()) / max(min(b_no.values()), 1)
    # 裁剪后：各类框数落入窗口（4× + 松弛），且确实删了框。
    ratio_tr = max(b_tr.values()) / max(min(b_tr.values()), 1)
    assert s_tr["trim_summary"]["enabled"] is True
    assert s_tr["trim_summary"]["trimmed_total"] > 0
    assert ratio_tr <= 4.5, (b_tr, ratio_tr)
    assert ratio_tr < ratio_no, (b_no, b_tr)
    # 裁剪不改变保留的类别数（不删类，只删框）。
    assert set(b_tr) == set(b_no)
