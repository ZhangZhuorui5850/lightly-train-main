from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from PIL import Image

from tool_lib import common as rt
from tool_lib import seg_shared
from tool_lib import seg_tools
from tool_lib.interactive import parse_cli_args


def test_eval_parser_accepts_shard_flags():
    args = parse_cli_args([
        "eval", "--task", "seg", "--data", "d.yaml",
        "--shard-index", "1", "--num-shards", "3", "--dry-run",
        "--skip-important-artifacts",
    ])
    assert args.shard_index == 1
    assert args.num_shards == 3
    assert args.dry_run is True
    assert args.skip_important_artifacts is True


def test_eval_parser_shard_defaults():
    args = parse_cli_args(["eval", "--task", "seg", "--data", "d.yaml"])
    assert args.shard_index is None
    assert args.num_shards == 1
    assert args.dry_run is False


def _roundtrip(mask):
    rle = seg_shared.rle_encode(mask)
    out = seg_shared.rle_decode(rle)
    assert out.dtype == bool
    assert out.shape == mask.shape
    assert np.array_equal(out, mask.astype(bool))


def test_rle_roundtrip_mixed():
    mask = np.array([[1, 1, 0], [0, 1, 1]], dtype=bool)
    _roundtrip(mask)


def test_rle_roundtrip_all_zero_and_all_one():
    _roundtrip(np.zeros((4, 5), dtype=bool))
    _roundtrip(np.ones((4, 5), dtype=bool))


def test_rle_roundtrip_non_square():
    rng = np.random.default_rng(0)
    _roundtrip(rng.integers(0, 2, size=(7, 13)).astype(bool))


def test_rle_size_field_matches_shape():
    mask = np.zeros((3, 8), dtype=bool)
    rle = seg_shared.rle_encode(mask)
    assert rle["size"] == [3, 8]


def _reference_load(mask_path, classes, ignore_classes):
    # 旧实现逻辑的参考副本，用于等价性对拍
    with Image.open(mask_path) as mask_image:
        mask_np = np.array(mask_image)
    compare_np = mask_np if mask_np.ndim == 3 else mask_np[:, :, None]
    target = np.full(mask_np.shape[:2], -100, dtype=np.int64)
    original_to_internal = {
        cid: i for i, cid in enumerate(sorted(set(classes) - ignore_classes))
    }
    for cid, internal in original_to_internal.items():
        for label in seg_tools._class_labels(classes, cid):
            label_tuple = tuple(int(v) for v in label) if isinstance(label, tuple) else (int(label),)
            target[np.all(compare_np == np.array(label_tuple), axis=2)] = internal
    return target


def test_load_semantic_mask_single_channel_matches_reference(tmp_path):
    rt.import_runtime_dependencies()
    classes = {0: "bg", 1: "cat", 2: "dog", 255: "ignore"}
    ignore = {255}
    arr = np.array([[0, 1, 2], [255, 1, 0]], dtype=np.uint8)
    p = tmp_path / "m.png"
    Image.fromarray(arr, mode="L").save(p)
    expected = _reference_load(p, classes, ignore)
    got = seg_tools._load_semantic_mask(p, classes, ignore)
    assert np.array_equal(got, expected)


def test_load_semantic_mask_rgb_labels_matches_reference(tmp_path):
    rt.import_runtime_dependencies()
    classes = {
        0: {"name": "bg", "labels": [[0, 0, 0]]},
        1: {"name": "road", "labels": [[128, 64, 128]]},
        2: {"name": "sky", "labels": [[70, 130, 180]]},
    }
    ignore = set()
    arr = np.array(
        [[[0, 0, 0], [128, 64, 128]], [[70, 130, 180], [0, 0, 0]]], dtype=np.uint8
    )
    p = tmp_path / "m.png"
    Image.fromarray(arr, mode="RGB").save(p)
    expected = _reference_load(p, classes, ignore)
    got = seg_tools._load_semantic_mask(p, classes, ignore)
    assert np.array_equal(got, expected)


def test_merge_semantic_confusion_equals_single(tmp_path):
    rt.import_runtime_dependencies()
    full = np.array([[5, 1, 0], [0, 4, 2], [1, 0, 6]], dtype=np.int64)
    a = np.array([[3, 1, 0], [0, 1, 1], [0, 0, 3]], dtype=np.int64)
    b = full - a
    d1 = tmp_path / "s0"; d2 = tmp_path / "s1"
    d1.mkdir(); d2.mkdir()
    names = {0: "a", 1: "b", 2: "c"}
    seg_tools._write_semantic_shard_result(d1, split="test", confusion=a, rows=[], class_names=names, num_samples=2, infer_time_sum_ms=10.0, failed=0)
    seg_tools._write_semantic_shard_result(d2, split="test", confusion=b, rows=[], class_names=names, num_samples=3, infer_time_sum_ms=20.0, failed=0)
    merged = seg_tools._merge_semantic_shard_results([d1, d2], split="test")
    assert np.array_equal(merged["confusion"], full)
    assert merged["num_samples"] == 5
    assert merged["infer_time_sum_ms"] == 30.0


def test_serialize_instance_entry_roundtrip():
    rt.import_runtime_dependencies()
    prediction = {
        "labels": torch.tensor([0, 2], dtype=torch.int64),
        "scores": torch.tensor([0.9, 0.5], dtype=torch.float32),
        "masks": torch.tensor(
            np.stack([np.eye(4, dtype=bool), np.ones((4, 4), dtype=bool)]),
        ),
    }
    target = {
        "labels": torch.tensor([0], dtype=torch.int64),
        "masks": torch.tensor(np.eye(4, dtype=bool)[None]),
    }
    entry = seg_tools._serialize_instance_entry(prediction, target)
    pred2, tgt2 = seg_tools._deserialize_instance_entry(entry)
    assert torch.equal(pred2["labels"], prediction["labels"])
    assert torch.allclose(pred2["scores"], prediction["scores"])
    assert torch.equal(pred2["masks"], prediction["masks"])
    assert torch.equal(tgt2["masks"], target["masks"])


def test_run_parallel_seg_eval_falls_back_when_few_gpus(monkeypatch):
    monkeypatch.setattr(
        seg_tools.gpu_parallel, "query_gpu_inventory",
        lambda: ([{"index": 0, "used_ratio": 0.1, "memory_used": 0, "memory_total": 1, "utilization": 0}], ""),
    )
    args = SimpleNamespace(device="auto", shard_index=None, num_shards=1, dry_run=False, data="d.yaml")
    assert seg_tools.run_parallel_seg_eval(args) is False


def test_run_parallel_seg_eval_skips_when_device_fixed():
    args = SimpleNamespace(device="cuda:0", shard_index=None, num_shards=1, dry_run=False, data="d.yaml")
    assert seg_tools.run_parallel_seg_eval(args) is False


def test_build_seg_eval_child_command_preserves_all_splits():
    # --split 是 nargs="+"，重复 --split 会被覆盖；子命令必须能解析回全部 split。
    args = SimpleNamespace(
        seg_train_type="semantic", experiment_dir=None, checkpoint=None,
        data="d.yaml", split=["val", "test"], classwise=False, overwrite=False,
    )
    command = seg_tools._build_seg_eval_child_command(
        args, shard_index=0, num_shards=2, output_dir="/tmp/shard_00", device="auto"
    )
    parsed = parse_cli_args(command[2:])  # drop [python, launcher.py]; keep "eval" subcommand
    assert parsed.split == ["val", "test"]
    assert parsed.shard_index == 0 and parsed.num_shards == 2


def test_run_parallel_seg_infer_falls_back_single_gpu(monkeypatch):
    monkeypatch.setattr(
        seg_tools.gpu_parallel, "query_gpu_inventory",
        lambda: ([{"index": 0, "used_ratio": 0.1, "memory_used": 0, "memory_total": 1, "utilization": 0}], ""),
    )
    args = SimpleNamespace(device="auto", shard_index=None, num_shards=1, dry_run=False)
    assert seg_tools.run_parallel_seg_infer(args) is False


def test_prefetch_iter_preserves_order_and_payload():
    def _load(x):
        return x * 10
    items = [1, 2, 3, 4]
    out = list(seg_tools._prefetch_iter(items, _load))
    assert out == [(1, 10), (2, 20), (3, 30), (4, 40)]


def test_prefetch_iter_empty():
    assert list(seg_tools._prefetch_iter([], lambda x: x)) == []


def test_write_seg_run_meta_records_core_fields(tmp_path):
    args = SimpleNamespace(
        seg_train_type="semantic", data="d.yaml", split=["test"],
        device="auto", overwrite=False, threshold=None,
    )
    path = tmp_path / "run_meta.json"
    seg_tools._write_seg_run_meta(
        path, action="eval", checkpoint_path=tmp_path / "ck.pt",
        output_dir=tmp_path, args=args, num_images=12, device_mode="cuda:0",
    )
    import json as _json
    payload = _json.loads(path.read_text(encoding="utf-8"))
    assert payload["task"] == "seg"
    assert payload["action"] == "eval"
    assert payload["num_images"] == 12
    assert payload["settings"]["device_mode"] == "cuda:0"
