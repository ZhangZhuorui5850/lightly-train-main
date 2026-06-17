"""语义分割 EDA + curate 测试。

构造微型语义数据集（数张已知像素值的 PNG mask + 最小 data.yaml），
验证 EDA / 删除 / 压缩 / 共现保护 / 范围 / 导出 / 边界。
"""

from __future__ import annotations

import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import yaml

from tool_lib import common as rt

# ---------------------------------------------------------------------------
# 测试数据集构造
# ---------------------------------------------------------------------------

def _create_test_dataset(root: Path) -> Path:
    """创建微型语义分割数据集。

    类别:
      0: background (像素值 0)
      1: cat (像素值 1)  — 稀有类，仅 1 张 train
      2: dog (像素值 2)  — 中等类，3 张 train
      3: car (像素值 3)  — 大类，4 张 train，超阈值候选

    ignore_classes: [255]

    train: 5 张图 (img_001 不含 car)
    val: 2 张图
    test: 2 张图
    """
    from PIL import Image

    for split in ("train", "val", "test"):
        (root / split / "images").mkdir(parents=True, exist_ok=True)
        (root / split / "masks").mkdir(parents=True, exist_ok=True)

    classes = {0: "background", 1: "cat", 2: "dog", 3: "car"}
    MASK_SIZE = 100  # 100x100 = 10000 像素

    # train 数据 (5 张)
    train_masks = [
        {0: 5000, 1: 3000, 2: 2000},   # img_001: bg + cat + dog (无 car)
        {0: 3000, 2: 3000, 3: 4000},   # img_002: bg + dog + car
        {0: 4000, 3: 6000},            # img_003: bg + car
        {0: 2000, 2: 3000, 3: 5000},   # img_004: bg + dog + car
        {0: 3000, 3: 7000},            # img_005: bg + car
    ]

    for i, mask_data in enumerate(train_masks):
        img_name = f"img_{i + 1:03d}.jpg"
        mask_name = f"img_{i + 1:03d}.png"
        (root / "train" / "images" / img_name).write_bytes(b"fake")
        mask_arr = np.zeros((MASK_SIZE, MASK_SIZE), dtype=np.uint8)
        idx = 0
        for class_id, count in mask_data.items():
            count = min(count, MASK_SIZE * MASK_SIZE - idx)
            mask_arr.flat[idx:idx + count] = class_id
            idx += count
        Image.fromarray(mask_arr).save(root / "train" / "masks" / mask_name)

    # val 数据 (2 张)
    val_masks = [
        {0: 5000, 1: 2000, 3: 3000},  # val_001: bg + cat + car
        {0: 4000, 2: 3000, 3: 3000},  # val_002: bg + dog + car
    ]
    for i, mask_data in enumerate(val_masks):
        img_name = f"val_{i + 1:03d}.jpg"
        mask_name = f"val_{i + 1:03d}.png"
        (root / "val" / "images" / img_name).write_bytes(b"fake")
        mask_arr = np.zeros((MASK_SIZE, MASK_SIZE), dtype=np.uint8)
        idx = 0
        for class_id, count in mask_data.items():
            count = min(count, MASK_SIZE * MASK_SIZE - idx)
            mask_arr.flat[idx:idx + count] = class_id
            idx += count
        Image.fromarray(mask_arr).save(root / "val" / "masks" / mask_name)

    # test 数据 (2 张)
    test_masks = [
        {0: 5000, 2: 2000, 3: 3000},  # test_001: bg + dog + car
        {0: 3000, 3: 7000},            # test_002: bg + car
    ]
    for i, mask_data in enumerate(test_masks):
        img_name = f"test_{i + 1:03d}.jpg"
        mask_name = f"test_{i + 1:03d}.png"
        (root / "test" / "images" / img_name).write_bytes(b"fake")
        mask_arr = np.zeros((MASK_SIZE, MASK_SIZE), dtype=np.uint8)
        idx = 0
        for class_id, count in mask_data.items():
            count = min(count, MASK_SIZE * MASK_SIZE - idx)
            mask_arr.flat[idx:idx + count] = class_id
            idx += count
        Image.fromarray(mask_arr).save(root / "test" / "masks" / mask_name)

    data_yaml = {
        "path": str(root),
        "train": {"images": str(root / "train" / "images"), "masks": str(root / "train" / "masks")},
        "val": {"images": str(root / "val" / "images"), "masks": str(root / "val" / "masks")},
        "test": {"images": str(root / "test" / "images"), "masks": str(root / "test" / "masks")},
        "classes": classes,
        "ignore_classes": [255],
    }
    yaml_path = root / "data.yaml"
    yaml_path.write_text(yaml.dump(data_yaml, allow_unicode=True, default_flow_style=False), encoding="utf-8")

    return yaml_path


def _run_eda(tmp_path: Path, data_yaml: Path, **kwargs) -> Path:
    """运行 EDA 并返回输出目录。"""
    rt.import_runtime_dependencies()
    from tool_lib.seg_semantic_eda import generate_semantic_eda_report
    return generate_semantic_eda_report(
        source_data_path=data_yaml,
        output_dir=tmp_path / "eda_output",
        overwrite=True,
        **kwargs,
    )


def _run_curate(data_yaml: Path, eda_dir: Path, *, drop_classes=None, image_threshold=0) -> Path:
    """运行 curate 并返回新数据集路径。"""
    rt.import_runtime_dependencies()
    from tool_lib.seg_semantic_curate import run_semantic_curate
    import argparse
    args = argparse.Namespace(
        data=data_yaml,
        eda_dir=eda_dir,
        drop_classes=drop_classes,
        image_threshold=image_threshold,
        export_suffix="__curated",
    )
    run_semantic_curate(args)
    # 新数据集在 data_yaml.parent (即 dataset 目录) 同级，后缀 __curated
    return data_yaml.parent.parent / (data_yaml.parent.name + "__curated")


# ---------------------------------------------------------------------------
# Test 1: EDA — 每类图片数/像素数计数正确
# ---------------------------------------------------------------------------

class TestEDA:
    def test_class_image_counts(self, tmp_path: Path):
        """每类图片数计数正确。"""
        data_yaml = _create_test_dataset(tmp_path / "dataset")
        eda_dir = _run_eda(tmp_path, data_yaml, min_class_images=2, threshold_percentile=0.9)

        report = json.loads((eda_dir / "semantic_eda_semantic.json").read_text(encoding="utf-8"))
        classes = report["classes"]

        # cat (id=1): train 1 张 (img_001), val 1 张 (val_001), all 2 张
        assert classes["1"]["image_count_all"] == 2
        assert classes["1"]["image_count_by_split"].get("train", 0) == 1
        assert classes["1"]["image_count_by_split"].get("val", 0) == 1

        # dog (id=2): train 3 张 (img_001, img_002, img_004), val 1 张, test 1 张, all 5 张
        assert classes["2"]["image_count_all"] == 5
        assert classes["2"]["image_count_by_split"].get("train", 0) == 3

        # car (id=3): train 4 张 (img_002-005), val 2 张, test 2 张, all 8 张
        assert classes["3"]["image_count_all"] == 8
        assert classes["3"]["image_count_by_split"].get("train", 0) == 4

    def test_class_pixel_counts(self, tmp_path: Path):
        """每类像素数计数正确。"""
        data_yaml = _create_test_dataset(tmp_path / "dataset")
        eda_dir = _run_eda(tmp_path, data_yaml, min_class_images=2)

        report = json.loads((eda_dir / "semantic_eda_semantic.json").read_text(encoding="utf-8"))
        classes = report["classes"]

        # cat (id=1): img_001=3000, val_001=2000 → 5000
        assert classes["1"]["pixel_count_all"] == 5000
        # car (id=3): img_002=4000, img_003=6000, img_004=5000, img_005=7000, val_001=3000, val_002=3000, test_001=3000, test_002=7000 = 38000
        assert classes["3"]["pixel_count_all"] == 38000

    def test_recommendations(self, tmp_path: Path):
        """推荐删除/阈值符合预期。"""
        data_yaml = _create_test_dataset(tmp_path / "dataset")
        eda_dir = _run_eda(tmp_path, data_yaml, min_class_images=2, threshold_percentile=0.9)

        report = json.loads((eda_dir / "semantic_eda_semantic.json").read_text(encoding="utf-8"))
        rec = report["recommendations"]

        # min_class_images=2, cat 全局 2 张 → 不删除
        assert 1 not in rec["recommend_delete"]
        # 推荐阈值 > 0
        assert rec["recommend_threshold"] > 0

    def test_inventory_csv(self, tmp_path: Path):
        """inventory CSV 字段正确。"""
        import csv
        data_yaml = _create_test_dataset(tmp_path / "dataset")
        eda_dir = _run_eda(tmp_path, data_yaml, min_class_images=2)

        inventory_path = eda_dir / "image_class_inventory_semantic.csv"
        assert inventory_path.exists()

        with open(inventory_path, "r", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))

        assert len(rows) == 9  # 5 train + 2 val + 2 test
        for row in rows:
            assert "split" in row
            assert "class_ids" in row
            assert "per_class_pixels" in row

    def test_markdown_exists(self, tmp_path: Path):
        """markdown 报告存在且包含类别表。"""
        data_yaml = _create_test_dataset(tmp_path / "dataset")
        eda_dir = _run_eda(tmp_path, data_yaml, min_class_images=2)

        md_path = eda_dir / "semantic_eda_semantic.md"
        assert md_path.exists()
        content = md_path.read_text(encoding="utf-8")
        assert "语义分割 EDA 报告" in content
        assert "cat" in content


# ---------------------------------------------------------------------------
# Test 2: 删除 — yaml 不含被删类
# ---------------------------------------------------------------------------

class TestDeletion:
    def test_yaml_excludes_deleted_classes(self, tmp_path: Path):
        """导出 yaml 的 classes 不含被删类。"""
        data_yaml = _create_test_dataset(tmp_path / "dataset")
        eda_dir = _run_eda(tmp_path, data_yaml, min_class_images=2)
        new_root = _run_curate(data_yaml, eda_dir, drop_classes="1", image_threshold=0)

        new_yaml_path = new_root / "data.yaml"
        assert new_yaml_path.exists()

        new_cfg = yaml.safe_load(new_yaml_path.read_text(encoding="utf-8"))
        assert 1 not in new_cfg["classes"]  # cat 删除
        assert 2 in new_cfg["classes"]
        assert 3 in new_cfg["classes"]


# ---------------------------------------------------------------------------
# Test 3: 压缩 — 超阈值类的 train 图片数 ≤ 阈值
# ---------------------------------------------------------------------------

class TestCompression:
    def test_compression_respects_threshold(self, tmp_path: Path):
        """超阈值类的 train 图片数 ≤ 阈值。"""
        data_yaml = _create_test_dataset(tmp_path / "dataset")
        eda_dir = _run_eda(tmp_path, data_yaml, min_class_images=2)
        # car (id=3) 有 4 张 train，设阈值=3
        new_root = _run_curate(data_yaml, eda_dir, drop_classes="", image_threshold=3)

        manifest = json.loads((new_root / "curate_manifest.json").read_text(encoding="utf-8"))
        after_counts = manifest["downsample"]["class_image_count_after"]
        assert after_counts.get(3, 0) <= 3

    def test_threshold_zero_no_compression(self, capsys, tmp_path: Path):
        """无删除无阈值 → 跳过导出。"""
        data_yaml = _create_test_dataset(tmp_path / "dataset")
        eda_dir = _run_eda(tmp_path, data_yaml, min_class_images=2)
        _run_curate(data_yaml, eda_dir, drop_classes="", image_threshold=0)

        captured = capsys.readouterr()
        assert "跳过导出" in captured.out


# ---------------------------------------------------------------------------
# Test 4: 共现保护 — 含稀有类的 train 图被保留
# ---------------------------------------------------------------------------

class TestCooccurrenceProtection:
    def test_rare_class_images_preserved(self, tmp_path: Path):
        """含稀有类的 train 图被保留。"""
        data_yaml = _create_test_dataset(tmp_path / "dataset")
        eda_dir = _run_eda(tmp_path, data_yaml, min_class_images=2)
        # cat (id=1) 仅在 img_001 中出现（train），设阈值=2 压缩 dog/car
        new_root = _run_curate(data_yaml, eda_dir, drop_classes="", image_threshold=2)

        train_images = list((new_root / "train" / "images").iterdir())
        train_names = {p.name for p in train_images}
        # img_001.jpg 含 cat（稀有类），必须被保留
        assert "img_001.jpg" in train_names


# ---------------------------------------------------------------------------
# Test 5: 范围 — val/test 图片/掩码数量不变
# ---------------------------------------------------------------------------

class TestScope:
    def test_val_test_unchanged(self, tmp_path: Path):
        """val/test 图片/掩码数量不变。"""
        data_yaml = _create_test_dataset(tmp_path / "dataset")
        eda_dir = _run_eda(tmp_path, data_yaml, min_class_images=2)
        new_root = _run_curate(data_yaml, eda_dir, drop_classes="1", image_threshold=3)

        val_images = list((new_root / "val" / "images").iterdir())
        val_masks = list((new_root / "val" / "masks").iterdir())
        assert len(val_images) == 2
        assert len(val_masks) == 2

        test_images = list((new_root / "test" / "images").iterdir())
        test_masks = list((new_root / "test" / "masks").iterdir())
        assert len(test_images) == 2
        assert len(test_masks) == 2


# ---------------------------------------------------------------------------
# Test 6: 导出 — 硬链接 + yaml 可加载
# ---------------------------------------------------------------------------

class TestExport:
    def test_hard_links_created(self, tmp_path: Path):
        """硬链接生成（同设备）。"""
        data_yaml = _create_test_dataset(tmp_path / "dataset")
        eda_dir = _run_eda(tmp_path, data_yaml, min_class_images=2)
        new_root = _run_curate(data_yaml, eda_dir, drop_classes="", image_threshold=3)

        assert (new_root / "data.yaml").exists()
        assert (new_root / "curate_manifest.json").exists()

        # 检查硬链接（同设备 inode 相同）
        src_img = tmp_path / "dataset" / "train" / "images" / "img_001.jpg"
        dst_img = new_root / "train" / "images" / "img_001.jpg"
        if dst_img.exists() and src_img.exists():
            assert os.stat(src_img).st_ino == os.stat(dst_img).st_ino

    def test_yaml_loadable(self, tmp_path: Path):
        """产出 yaml 能被 load_semantic_segmentation_data_config 加载。"""
        from tool_lib import train_tools

        data_yaml = _create_test_dataset(tmp_path / "dataset")
        eda_dir = _run_eda(tmp_path, data_yaml, min_class_images=2)
        new_root = _run_curate(data_yaml, eda_dir, drop_classes="1", image_threshold=0)

        new_yaml = new_root / "data.yaml"
        cfg = train_tools.load_semantic_segmentation_data_config(new_yaml)
        assert "classes" in cfg
        assert "train" in cfg
        assert "val" in cfg


# ---------------------------------------------------------------------------
# Test 7: 边界 — 无删除无阈值 → 警告/跳过
# ---------------------------------------------------------------------------

class TestEdgeCases:
    def test_no_delete_no_threshold_skip(self, capsys, tmp_path: Path):
        """无删除无阈值 → 警告并跳过导出。"""
        data_yaml = _create_test_dataset(tmp_path / "dataset")
        eda_dir = _run_eda(tmp_path, data_yaml, min_class_images=2)

        # 使用 CLI 模式（drop_classes="" 表示不删除，image_threshold=0 表示不压缩）
        new_root = _run_curate(data_yaml, eda_dir, drop_classes="", image_threshold=0)

        captured = capsys.readouterr()
        assert "跳过导出" in captured.out
