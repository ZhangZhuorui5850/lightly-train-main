#!/usr/bin/env python3
"""按尺寸档 / 密集度 / 官方有效性分组的检测指标评估（纯后处理，不需要 GPU）。

输入是 `launcher.py eval/infer --save-json --score-threshold 0.0` 写出的每图 JSON
目录，外加数据集本身（用于读 GT）。输出分档 AP50 / 召回，用来回答：

- 原模型与 combined 的差距落在哪个尺寸档（而不是只看一个总 mAP）；
- 哪些 GT 在当前分辨率下原理上不可达（小于一个 ViT patch）；
- 排除官方协议会忽略的框之后，模型真实的表现是多少。

尺寸口径统一为「设备像素短边」= min(bw * S / W, bh * S / H)，其中 (W,H) 是原图尺寸、
S 是训练/整图推理的方形拉伸边长。这是模型在输入里实际看到的框短边，也是本仓库
唯一应当对外引用的口径（不要与保比缩放口径混用）。

用法：
    python band_eval.py --pred-json-dir out/.../json \
        --data datasets/face_detect/face_yolo_wider/data.yaml --split test
    python band_eval.py --self-test          # 用合成数据校验匹配与分档逻辑
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import math
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np
import yaml
from PIL import Image

Image.MAX_IMAGE_PIXELS = None

# 尺寸档（设备像素短边，单位 px）。上限用 inf 表示。
BANDS: list[tuple[str, float, float]] = [
    ("<8", 0.0, 8.0),
    ("8-16", 8.0, 16.0),
    ("16-32", 16.0, 32.0),
    (">=32", 32.0, math.inf),
]
BANDS.append(("all", 0.0, math.inf))


def device_short_side(
    boxes: np.ndarray, image_wh: tuple[int, int], image_size: int
) -> np.ndarray:
    """框在模型输入中的短边（像素）。boxes 为原图像素 xyxy。"""
    width, height = image_wh
    if boxes.size == 0:
        return np.zeros((0,), dtype=np.float64)
    scale_x, scale_y = image_size / width, image_size / height
    box_w = (boxes[:, 2] - boxes[:, 0]) * scale_x
    box_h = (boxes[:, 3] - boxes[:, 1]) * scale_y
    return np.minimum(box_w, box_h)


def load_split(data_yaml: Path, split: str) -> list[dict[str, Any]]:
    """读取一个 split 的原图尺寸与 GT（原图像素 xyxy，原生高度）。"""
    config = yaml.safe_load(data_yaml.read_text(encoding="utf-8"))
    root = Path(config.get("path", ".") or ".")
    if not root.is_absolute():
        root = data_yaml.parent / root
    rel = Path(str(config[split]))
    images_dir = rel if rel.is_absolute() else root / rel
    parts = list(rel.parts)
    parts[parts.index("images")] = "labels"
    labels_dir = Path(*parts) if Path(*parts).is_absolute() else root / Path(*parts)

    samples: list[dict[str, Any]] = []
    for image_path in sorted(images_dir.rglob("*")):
        if not image_path.is_file():
            continue
        label_path = labels_dir / image_path.relative_to(images_dir).with_suffix(".txt")
        with Image.open(image_path) as image:
            image_wh = image.size
        boxes: list[list[float]] = []
        if label_path.is_file():
            for line in label_path.read_text(encoding="utf-8").splitlines():
                parts_ = line.split()
                if len(parts_) < 5:
                    continue
                cx, cy, bw, bh = (float(v) for v in parts_[1:5])
                width, height = image_wh
                boxes.append(
                    [
                        (cx - bw / 2) * width,
                        (cy - bh / 2) * height,
                        (cx + bw / 2) * width,
                        (cy + bh / 2) * height,
                    ]
                )
        samples.append(
            {
                "image_path": image_path,
                "relative": image_path.relative_to(images_dir).with_suffix(""),
                "image_wh": image_wh,
                "gt": np.asarray(boxes, dtype=np.float64).reshape(-1, 4),
            }
        )
    return samples


def load_predictions(json_dir: Path, samples: Sequence[dict[str, Any]]) -> None:
    """把 --save-json 的产物挂到 samples 上，缺失的按空预测处理。"""
    missing = 0
    for sample in samples:
        json_path = (json_dir / sample["relative"]).with_suffix(".json")
        boxes: list[list[float]] = []
        scores: list[float] = []
        if json_path.is_file():
            payload = json.loads(json_path.read_text(encoding="utf-8"))
            for record in payload.get("predictions", []):
                box = record.get("bbox_xyxy")
                if not box or len(box) != 4:
                    continue
                boxes.append([float(v) for v in box])
                scores.append(float(record.get("score", 0.0)))
        else:
            missing += 1
        sample["pred"] = np.asarray(boxes, dtype=np.float64).reshape(-1, 4)
        sample["score"] = np.asarray(scores, dtype=np.float64)
    if missing:
        print(f"[warn] {missing}/{len(samples)} 张图没有预测 JSON，按空预测处理")


def evaluate_subset(
    samples: Sequence[dict[str, Any]],
    image_size: int,
    keep: Callable[[dict[str, Any], int], bool],
    max_dets: int,
    iou_threshold: float = 0.5,
) -> dict[str, float]:
    """在 keep(sample, gt_index) 选中的 GT 子集上算 AP50 / 召回。

    未被选中的 GT 标记为 crowd：既不计漏检、也不把命中它的预测算成误检
    （与 COCO 的 ignore 语义一致，也与 WIDER 官方忽略 <=10px 人脸的约定一致）。

    注意：本环境安装的 pycocotools 在 cocoeval.py:109 用 `iscrowd` 覆盖 `ignore`
    字段，只传 ignore=1 不生效，必须走 iscrowd。代价是被排除的 GT 按 crowd 区域
    用 IoA 匹配，对"排除出评分"这个用途是可以接受的近似。
    """
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval

    images: list[dict[str, Any]] = []
    annotations: list[dict[str, Any]] = []
    detections: list[dict[str, Any]] = []
    ann_id = 0
    total_kept = 0

    for image_id, sample in enumerate(samples, start=1):
        width, height = sample["image_wh"]
        images.append({"id": image_id, "width": width, "height": height})
        for index, box in enumerate(sample["gt"]):
            keep_it = keep(sample, index)
            total_kept += int(keep_it)
            ann_id += 1
            box_w, box_h = box[2] - box[0], box[3] - box[1]
            annotations.append(
                {
                    "id": ann_id,
                    "image_id": image_id,
                    "category_id": 1,
                    "bbox": [box[0], box[1], box_w, box_h],
                    "area": float(box_w * box_h),
                    "iscrowd": 0 if keep_it else 1,
                }
            )
        for box, score in zip(sample["pred"], sample["score"]):
            box_w, box_h = box[2] - box[0], box[3] - box[1]
            detections.append(
                {
                    "image_id": image_id,
                    "category_id": 1,
                    "bbox": [box[0], box[1], box_w, box_h],
                    "score": float(score),
                }
            )

    if not detections or total_kept == 0:
        return {"gt": float(total_kept), "ap50": float("nan"), "recall": float("nan")}

    # pycocotools 会往 stdout 打进度；这里屏蔽掉，只保留我们自己的报表。
    with contextlib.redirect_stdout(io.StringIO()):
        coco_gt = COCO()
        coco_gt.dataset = {
            "images": images,
            "annotations": annotations,
            "categories": [{"id": 1, "name": "face"}],
        }
        coco_gt.createIndex()
        coco_dt = coco_gt.loadRes(detections)
        coco_eval = COCOeval(coco_gt, coco_dt, iouType="bbox")
        coco_eval.params.iouThrs = np.array([iou_threshold])
        coco_eval.params.maxDets = [max_dets]
        coco_eval.params.areaRng = [[0.0, 1e10]]
        coco_eval.params.areaRngLbl = ["all"]
        coco_eval.evaluate()
        coco_eval.accumulate()

    # 直接从累积结果取数，绕开 summarize() 对固定参数表长度的假设。
    # precision 形状 (T, R, K, A, M)，recall 形状 (T, K, A, M)；此处 T=K=A=M=1。
    precision = coco_eval.eval["precision"][0, :, 0, 0, 0]
    recall = coco_eval.eval["recall"][0, 0, 0, 0]
    valid = precision[precision > -1]
    return {
        "gt": float(total_kept),
        "ap50": float(valid.mean()) if valid.size else float("nan"),
        "recall": float(recall),
    }


def report(
    samples: Sequence[dict[str, Any]],
    image_size: int,
    patch: int,
    max_dets: int,
    official_min_native_height: float,
) -> dict[str, Any]:
    device = [device_short_side(s["gt"], s["image_wh"], image_size) for s in samples]
    for sample, values in zip(samples, device):
        sample["device_short"] = values
        sample["native_h"] = (
            sample["gt"][:, 3] - sample["gt"][:, 1]
            if sample["gt"].size
            else np.zeros((0,))
        )

    def subset(name: str, keep: Callable[[dict[str, Any], int], bool]) -> dict[str, Any]:
        metrics = evaluate_subset(samples, image_size, keep, max_dets)
        metrics["name"] = name
        return metrics

    results: dict[str, Any] = {"max_dets": max_dets, "image_size": image_size}
    results["bands"] = {}
    for label, low, high in BANDS:
        metrics = subset(
            label,
            lambda s, i, low=low, high=high: bool(low <= s["device_short"][i] < high),
        )
        results["bands"][label] = metrics

    results["subsets"] = {}
    results["subsets"]["official_valid"] = subset(
        "official_valid",
        lambda s, i: bool(s["native_h"][i] > official_min_native_height),
    )
    results["subsets"]["official_valid_and_ge1patch"] = subset(
        "official_valid_and_ge1patch",
        lambda s, i: bool(
            s["native_h"][i] > official_min_native_height
            and s["device_short"][i] >= patch
        ),
    )
    results["subsets"]["dense_images_gt_over_100"] = subset(
        "dense_images_gt_over_100",
        lambda s, i: len(s["gt"]) > 100,
    )
    results["subsets"]["small_and_dense"] = subset(
        "small_and_dense",
        lambda s, i: bool(s["device_short"][i] < 16 and len(s["gt"]) > 100),
    )
    return results


def print_report(results: dict[str, Any]) -> None:
    def line(name: str, metrics: dict[str, Any]) -> str:
        return (
            f"{name:<32} GT={metrics['gt']:>8.0f}  "
            f"AP50={metrics['ap50']:.4f}  R={metrics['recall']:.4f}"
        )

    print(f"\nmaxDets={results['max_dets']}  方形拉伸边长={results['image_size']}")
    print("--- 尺寸档（设备像素短边）---")
    for label, _low, _high in BANDS:
        print(line(label, results["bands"][label]))
    print("--- 子集 ---")
    for name, metrics in results["subsets"].items():
        print(line(name, metrics))


def self_test() -> int:
    """用合成数据校验匹配、分档与 exclude 语义。图像为 640x640，原生尺寸=设备尺寸。"""
    samples: list[dict[str, Any]] = [
        {
            # A=6px（官方会忽略、<8 档）B=12px（官方有效、8-16 档）C=40px（>=32 档）
            "image_wh": (640, 640),
            "gt": np.array(
                [[10, 10, 16, 16], [100, 100, 112, 112], [300, 300, 340, 340]],
                dtype=np.float64,
            ),
            "pred": np.array(
                [[10, 10, 16, 16], [100, 100, 112, 112], [300, 300, 340, 340]],
                dtype=np.float64,
            ),
            "score": np.array([0.9, 0.8, 0.7]),
            "relative": Path("a"),
        },
        {
            # D=12px（官方有效、8-16 档）完全漏检
            "image_wh": (640, 640),
            "gt": np.array([[50, 50, 62, 62]], dtype=np.float64),
            "pred": np.zeros((0, 4)),
            "score": np.zeros((0,)),
            "relative": Path("b"),
        },
    ]
    results = report(
        samples, image_size=640, patch=16, max_dets=100, official_min_native_height=10.0
    )
    print_report(results)

    def get(group: str, name: str, key: str) -> float:
        return float(results[group][name][key])

    def close(value: float, expected: float) -> bool:
        return math.isclose(value, expected, rel_tol=0.0, abs_tol=1e-9)

    # COCO 的 AP 在 101 个召回点上取平均、未达到的召回段按 0 计，所以"漏检"会同时
    # 压低 AP 与召回：命中 k 个召回满档时 AP = (100*k+1)/101。这里按该公式校验。
    checks = [
        ("<8 档 GT=1", get("bands", "<8", "gt") == 1.0),
        (
            "<8 档 AP50=1 R=1",
            close(get("bands", "<8", "ap50"), 1.0) and close(get("bands", "<8", "recall"), 1.0),
        ),
        ("8-16 档 GT=2", get("bands", "8-16", "gt") == 2.0),
        ("8-16 档 R=0.5", close(get("bands", "8-16", "recall"), 0.5)),
        (
            "8-16 档 AP50=51/101（漏检压低 AP）",
            close(get("bands", "8-16", "ap50"), 51 / 101),
        ),
        ("16-32 档无 GT", get("bands", "16-32", "gt") == 0.0),
        (
            ">=32 档 AP50=1 R=1",
            close(get("bands", ">=32", "ap50"), 1.0) and close(get("bands", ">=32", "recall"), 1.0),
        ),
        (
            "all 档 GT=4 R=0.75 AP50=76/101",
            get("bands", "all", "gt") == 4.0
            and close(get("bands", "all", "recall"), 0.75)
            and close(get("bands", "all", "ap50"), 76 / 101),
        ),
        (
            "official_valid GT=3（6px 被排除）R=2/3",
            get("subsets", "official_valid", "gt") == 3.0
            and close(get("subsets", "official_valid", "recall"), 2 / 3),
        ),
        (
            "official_valid_and_ge1patch GT=1（只剩 40px）",
            get("subsets", "official_valid_and_ge1patch", "gt") == 1.0,
        ),
        ("dense 子集无 GT", get("subsets", "dense_images_gt_over_100", "gt") == 0.0),
    ]
    failed = [name for name, ok in checks if not ok]
    for name, ok in checks:
        print(f"  [{'ok' if ok else 'FAIL'}] {name}")
    if failed:
        print(f"\nself-test 失败: {failed}")
        return 1
    print("\nself-test 全部通过")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--pred-json-dir", type=Path)
    parser.add_argument("--data", type=Path)
    parser.add_argument("--split", default="test")
    parser.add_argument("--image-size", type=int, default=640, help="训练/整图推理的方形拉伸边长")
    parser.add_argument("--patch", type=int, default=16, help="backbone patch size")
    parser.add_argument("--max-dets", type=int, default=100)
    parser.add_argument("--official-min-native-height", type=float, default=10.0)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--self-test", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.self_test:
        return self_test()
    if not args.pred_json_dir or not args.data:
        raise SystemExit("需要 --pred-json-dir 与 --data（或用 --self-test）")
    samples = load_split(args.data, args.split)
    print(f"[info] {args.split}: {len(samples)} 张图")
    load_predictions(args.pred_json_dir, samples)
    scores = [s["score"] for s in samples if s["score"].size]
    if scores:
        merged = np.concatenate(scores)
        print(f"[info] 预测分数最小值={merged.min():.4f}（应接近 0，否则说明 JSON 被阈值过滤过）")
    else:
        print("[info] 没有任何预测，指标将全为 0（可用来核对分档 GT 计数）")
    results = report(
        samples,
        image_size=args.image_size,
        patch=args.patch,
        max_dets=args.max_dets,
        official_min_native_height=args.official_min_native_height,
    )
    print_report(results)
    output = args.output or args.pred_json_dir.parent / f"band_eval_{args.split}_maxdets{args.max_dets}.json"
    results["split"] = args.split
    results["pred_json_dir"] = str(args.pred_json_dir)
    output.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nwritten: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
