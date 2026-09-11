#!/usr/bin/env python3
"""Generate a small synthetic YOLO-segmentation dataset for demoing the
YOLO-seg -> MVTec AD conversion.

Layout produced (mirrors datasets/neu_dataset/dataset_seg):

    sample_yoloseg/
      data.yaml
      classes.txt
      images/{train,val,test}/*.jpg
      labels/{train,val,test}/*.txt   # one polygon per line:
                                       #   cls x1 y1 x2 y2 ... xn yn  (normalized)

The set is deliberately small but covers the tricky cases the converter
must handle:
  * images with a single defect class
  * images with MULTIPLE classes (must be duplicated per class downstream)
  * "good" images with an empty label file (no defect at all)
"""
from __future__ import annotations

import argparse
import random
from pathlib import Path

import numpy as np
from PIL import Image

try:
    from .dataset_transaction import staged_output, validate_output_location
except ImportError:
    from dataset_transaction import staged_output, validate_output_location  # type: ignore[no-redef]

ROOT = Path(__file__).resolve().parent.parent / "sample_yoloseg"
W = H = 256
CLASSES = ["scratch", "dent", "crack", "stain"]

# Reproducible without Math.random-style nondeterminism.
RNG = random.Random(1234)
NP_RNG = np.random.default_rng(1234)


def rand_polygon(cx: float, cy: float, r: float, n: int) -> list[tuple[float, float]]:
    """A jittered convex-ish polygon (normalized coords) around (cx, cy)."""
    pts = []
    for k in range(n):
        ang = 2 * np.pi * k / n
        rr = r * (0.6 + 0.4 * RNG.random())
        x = min(max(cx + rr * np.cos(ang), 0.01), 0.99)
        y = min(max(cy + rr * np.sin(ang), 0.01), 0.99)
        pts.append((x, y))
    return pts


def make_image(seed: int) -> Image.Image:
    """A plausible-looking gray 'surface' with mild texture."""
    base = 110 + int(40 * RNG.random())
    arr = np.full((H, W, 3), base, dtype=np.int16)
    noise = NP_RNG.normal(0, 12, size=(H, W, 1))
    arr = np.clip(arr + noise, 0, 255).astype(np.uint8)
    return Image.fromarray(arr, "RGB")


# (split, stem, [(class_id, n_vertices), ...])  -- empty list = good image
SPEC = [
    ("train", "img_000", [(0, 6)]),
    ("train", "img_001", [(1, 5)]),
    ("train", "img_002", [(0, 7), (2, 6)]),          # multi-class
    ("train", "img_003", []),                         # good
    ("train", "img_004", [(3, 8)]),
    ("train", "img_005", [(1, 6), (3, 5)]),          # multi-class
    ("train", "img_006", []),                         # good
    ("train", "img_007", [(2, 6)]),
    ("val",   "img_100", [(0, 6)]),
    ("val",   "img_101", [(2, 5), (3, 6)]),          # multi-class
    ("val",   "img_102", []),                         # good
    ("test",  "img_200", [(1, 6)]),
    ("test",  "img_201", [(0, 7), (1, 5), (2, 6)]),  # 3-class
    ("test",  "img_202", []),                         # good
    ("test",  "img_203", [(3, 6)]),
]


def _generate(root: Path) -> None:
    global NP_RNG
    RNG.seed(1234)
    NP_RNG = np.random.default_rng(1234)
    for split in ("train", "val", "test"):
        (root / "images" / split).mkdir(parents=True, exist_ok=True)
        (root / "labels" / split).mkdir(parents=True, exist_ok=True)

    for i, (split, stem, defects) in enumerate(SPEC):
        img = make_image(i)
        img.save(root / "images" / split / f"{stem}.jpg", quality=92)

        lines = []
        for cls, nv in defects:
            cx, cy = 0.2 + 0.6 * RNG.random(), 0.2 + 0.6 * RNG.random()
            r = 0.08 + 0.10 * RNG.random()
            poly = rand_polygon(cx, cy, r, nv)
            coords = " ".join(f"{x:.6f} {y:.6f}" for x, y in poly)
            lines.append(f"{cls} {coords}")
        # Always write the label file (empty file == good image).
        (root / "labels" / split / f"{stem}.txt").write_text(
            "\n".join(lines) + ("\n" if lines else ""),
            encoding="utf-8",
        )

    (root / "classes.txt").write_text("\n".join(CLASSES) + "\n", encoding="utf-8")
    names = "\n".join(f'  {i}: "{c}"' for i, c in enumerate(CLASSES))
    (root / "data.yaml").write_text(
        "path: .\n"
        "train: images/train\n"
        "val: images/val\n"
        "test: images/test\n\n"
        "task: segment\n"
        f"nc: {len(CLASSES)}\n"
        f"names:\n{names}\n",
        encoding="utf-8",
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="生成可复现的 YOLO-seg 示例数据")
    parser.add_argument("--out", type=Path, default=ROOT, help="输出目录")
    parser.add_argument("--force", action="store_true", help="安全替换已有输出")
    parser.add_argument("--dry-run", action="store_true", help="显示示例数据生成计划，保持零写入")
    args = parser.parse_args(argv)
    output = validate_output_location(args.out, [])
    if args.dry_run:
        print(f"[dry-run] 计划生成 {len(SPEC)} 张图片，{len(CLASSES)} 个类别 -> {output}")
        return 0
    with staged_output(output, clean=args.force) as stage:
        _generate(stage)
    print(f"Wrote sample YOLO-seg dataset to {output}")
    print(f"  classes : {CLASSES}")
    print(f"  images  : {len(SPEC)} (incl. multi-class and good)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
