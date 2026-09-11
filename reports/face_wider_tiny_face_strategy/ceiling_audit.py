#!/usr/bin/env python3
"""量化 WIDER FACE 本地副本在当前训练/评测配置下的"可达上限"。

只读数据集，输出 JSON 到脚本同目录。用于解释为什么两个模型的绝对指标都低、
且差异只有约 1 个百分点。

口径说明（三者不可混用，本脚本全部显式标注）：
- native    : 原始图像像素。官方 WIDER 协议以人脸框高度 <=10px 作为 ignore 条件。
- device    : 模型实际输入像素。训练与整图推理都把图像方形拉伸到 640x640
              （ScaleJitter 使用 albumentations Resize 到 (S,S)，predict 使用
              resize 到 image_size），因此框的设备像素短边 = min(w*640/W, h*640/H)。
- patch     : 设备像素 / 16（DINOv3 ViT-S/16 的 patch 边长）；<1 patch 表示
              该框小于一个 token。
"""

from __future__ import annotations

import json
import os
from pathlib import Path

from PIL import Image

Image.MAX_IMAGE_PIXELS = None

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "datasets" / "face_detect" / "face_yolo_wider"
IMAGE_SIZE = 640
PATCH = 16
MAX_DETS = 100


def audit_split(split: str) -> dict[str, object]:
    images_dir = DATA / "images" / split
    labels_dir = DATA / "labels" / split
    per_image: list[tuple[int, int, int, int, int, int]] = []
    # (gt_total, native_h_le10, device_lt8, device_lt16, device_ge16, native_h_gt10_and_device_ge16)
    for name in sorted(os.listdir(images_dir)):
        label = labels_dir / f"{Path(name).stem}.txt"
        if not label.is_file():
            continue
        with Image.open(images_dir / name) as image:
            width, height = image.size
        scale_x, scale_y = IMAGE_SIZE / width, IMAGE_SIZE / height
        total = le10 = lt8 = lt16 = ge16 = reachable = 0
        for line in label.read_text(encoding="utf-8").splitlines():
            parts = line.split()
            if len(parts) < 5:
                continue
            box_w = float(parts[3]) * width
            box_h = float(parts[4]) * height
            device = min(box_w * scale_x, box_h * scale_y)
            total += 1
            le10 += box_h <= 10
            lt8 += device < 8
            lt16 += 8 <= device < 16
            ge16 += device >= 16
            # 官方协议会忽略的框不计入，且小于一个 patch 的框视为不可定位
            reachable += box_h > 10 and device >= 16
        per_image.append((total, le10, lt8, lt16, ge16, reachable))

    gt = sum(row[0] for row in per_image)
    images = len(per_image)
    dense = [row[0] for row in per_image if row[0] > MAX_DETS]
    dense_gt = sum(dense)
    # 每图最多评测 100 框时，超出容量的 GT 永远无法被命中
    capped = sum(max(0, row[0] - MAX_DETS) for row in per_image)
    beyond_queries = sum(max(0, row[0] - 300) for row in per_image)
    native_valid = gt - sum(row[1] for row in per_image)
    reachable_gt = sum(row[5] for row in per_image)
    return {
        "images": images,
        "boxes": gt,
        "dense_images_gt_over_100": len(dense),
        "dense_images_gt": dense_gt,
        "gt_beyond_maxdets100": capped,
        "gt_beyond_maxdets100_pct": round(capped / gt * 100, 2),
        "gt_beyond_query_budget300": beyond_queries,
        "gt_beyond_query_budget300_pct": round(beyond_queries / gt * 100, 2),
        "gt_device_short_lt8": sum(row[2] for row in per_image),
        "gt_device_short_8to16": sum(row[3] for row in per_image),
        "gt_device_short_ge16": sum(row[4] for row in per_image),
        "gt_device_short_lt16_pct": round(
            sum(row[2] + row[3] for row in per_image) / gt * 100, 2
        ),
        "gt_native_height_le10_official_ignore": sum(row[1] for row in per_image),
        "gt_native_height_le10_pct": round(
            sum(row[1] for row in per_image) / gt * 100, 2
        ),
        "gt_reachable_native_valid_and_ge1patch": reachable_gt,
        "gt_reachable_pct": round(reachable_gt / gt * 100, 2),
        "gt_native_valid_but_sub_patch": native_valid - reachable_gt,
        "gt_native_valid_but_sub_patch_pct": round(
            (native_valid - reachable_gt) / gt * 100, 2
        ),
    }


def main() -> int:
    result = {
        "image_size": IMAGE_SIZE,
        "patch_size": PATCH,
        "max_dets": MAX_DETS,
        "queries": 300,
        "data_root": str(DATA),
        "splits": {split: audit_split(split) for split in ("train", "val", "test")},
    }
    out = Path(__file__).with_name("ceiling_audit.json")
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result["splits"], ensure_ascii=False, indent=2))
    print(f"\nwritten: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
