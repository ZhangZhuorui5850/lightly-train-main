from __future__ import annotations

import csv
import json
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

from tool_lib import common as rt
from tool_lib import seg_tools


def test_csv_exports_selected_columns_preserving_auxiliary_data(tmp_path):
    rows = [{"image": "中文,图片.png", "prediction_path": "cache.png", "future_metadata": 7}]
    original = [dict(row) for row in rows]
    path = tmp_path / "samples.csv"
    rt.save_records_csv(path, rows, ["image", "score"])
    with path.open(encoding="utf-8", newline="") as stream:
        assert list(csv.DictReader(stream)) == [{"image": "中文,图片.png", "score": ""}]
    assert rows == original
    rt.save_records_csv(path, [], ["image", "score"])
    assert path.read_text(encoding="utf-8") == "image,score\n"


@pytest.mark.parametrize("splits", [["val"], ["val", "test"]])
@pytest.mark.parametrize("overwrite", [False, True])
@pytest.mark.parametrize("save_visualization", [False, True])
def test_semantic_shards_merge_csv_and_cached_visualizations(
    tmp_path, monkeypatch, splits, overwrite, save_visualization
):
    rt.import_data_dependencies()
    output = tmp_path / "eval"
    shard_dirs = [output / "_shards" / f"shard_{i:02d}" for i in range(2)]
    image_path = tmp_path / "image.png"
    mask_path = tmp_path / "mask.png"
    Image.new("RGB", (8, 8)).save(image_path)
    Image.new("L", (8, 8)).save(mask_path)
    monkeypatch.setattr(seg_tools, "_semantic_image_mask_samples", lambda *a: ([], {0: "bg"}, 1))
    monkeypatch.setattr(
        seg_tools.train_tools, "load_semantic_segmentation_split_config",
        lambda *a: {"classes": {0: "bg"}},
    )
    caches = []
    for shard_dir in shard_dirs:
        for split in splits:
            cache = shard_dir / split / "_prediction_cache" / "prediction.png"
            cache.parent.mkdir(parents=True, exist_ok=True)
            if save_visualization:
                Image.new("L", (8, 8)).save(cache)
                caches.append(cache)
            row = dict(image_path=str(image_path), mask_path=str(mask_path), valid_pixels=64,
                       vis_miou=1.0, vis_class=0,
                       prediction_path=str(cache) if save_visualization else None)
            seg_tools._write_semantic_shard_result(
                shard_dir / split, split=split, confusion=np.array([[64]]), rows=[row],
                class_names={0: "bg"}, num_samples=1, infer_time_sum_ms=2.0, failed=0,
            )
    args = SimpleNamespace(overwrite=overwrite, save_visualization=save_visualization,
                           vis_max_images=0, seg_train_type="semantic")
    seg_tools._merge_parallel_semantic(
        args, shard_dirs, splits=splits, final_output_dir=output,
        checkpoint_path=tmp_path / "model.pt", data_path=tmp_path / "data.yaml",
    )
    for split in splits:
        folder = output / split if len(splits) > 1 else output
        summary = json.loads((folder / "seg_semantic_eval_summary.json").read_text())
        assert summary["num_images"] == 2
        assert summary["metrics"]["miou"] == 1.0
        with (folder / "seg_semantic_eval_samples.csv").open() as stream:
            reader = csv.DictReader(stream)
            assert reader.fieldnames == ["image_path", "mask_path", "valid_pixels", "vis_miou", "vis_class"]
            assert len(list(reader)) == 2
        if save_visualization:
            manifest = json.loads((folder / "compare" / "manifest.json").read_text())
            assert manifest["rendered"] == 2
            assert all((folder / item["output"]).exists() for item in manifest["items"])
    assert all(cache.exists() for cache in caches)
    assert all((sd / split / "seg_semantic_shard_result.json").exists()
               for sd in shard_dirs for split in splits)


def test_sequential_semantic_eval_csv_and_overwrite(tmp_path, monkeypatch):
    rt.import_data_dependencies()
    row = dict(image_path="a.png", mask_path="m.png", valid_pixels=4,
               vis_miou=1.0, vis_class=0, prediction_path=None)
    monkeypatch.setattr(seg_tools, "_accumulate_semantic_confusion", lambda *a, **kw: {
        "confusion": np.array([[4]]), "rows": [row], "class_names": {0: "bg"},
        "num_classes": 1, "num_samples": 1, "infer_time_sum_ms": 1.0, "failed": 0,
    })
    output = tmp_path / "eval"
    kwargs = dict(model=None, data_path=tmp_path / "data.yaml", split="val",
                  output_dir=output, threshold=0.0, checkpoint_path=tmp_path / "model.pt",
                  save_visualization=False)
    summary = seg_tools._evaluate_semantic_split(**kwargs, overwrite=False)
    assert summary["metrics"]["miou"] == 1.0
    with pytest.raises(ValueError, match="Output directory is not empty"):
        seg_tools._evaluate_semantic_split(**kwargs, overwrite=False)
    stale = output / "stale.txt"
    stale.write_text("old")
    seg_tools._evaluate_semantic_split(**kwargs, overwrite=True)
    assert not stale.exists()
    assert "prediction_path" in row
