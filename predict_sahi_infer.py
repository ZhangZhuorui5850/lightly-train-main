#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""SAHI 切片推理脚本：用训练好的 LT-DETR 模型对高分辨率图做小目标检测。

SAHI 只用于推理（不训练、不改模型）：把大图切成重叠 tile + 一个全局缩放图，
并行前向后合并去重，从而看清在整图缩放下会糊掉的小目标。

用法：
    python predict_sahi_infer.py
直接改下面的「配置区」即可；也可命令行覆盖，见 build_args()。
"""

from __future__ import annotations

from pathlib import Path

# ===========================================================================
# 配置区（改这里）
# ===========================================================================
MODEL = "out/0414/NEU_train_unfreeze4/exported_models/exported_best.pt"  # 你训练导出的 .pt，或仓库模型名如 "dinov3/vitt16-ltdetr-coco"
IMAGE = "datasets/neu_dataset/dataset_det/images/val"  # 单张图，或一个文件夹（批量）
OUTPUT_DIR = "out/sahi_pred"  # 标注结果保存目录
DATA_YAML = "datasets/neu_dataset/dataset_det/data.yaml"  # 可选：用于把类别 ID 映射成名字；没有就设 None

THRESHOLD = 0.6  # 置信度阈值
OVERLAP = 0.2  # tile 重叠比例 [0,1)
NMS_IOU = 0.3  # tile 间 NMS 的 IoU 阈值
GLOBAL_LOCAL_IOU = 0.1  # 全局/局部一致性匹配阈值
COMPARE_PLAIN = True  # True 时同时跑普通 predict，打印两者框数对比


def main() -> None:
    args = build_args()

    import lightly_train

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    class_names = load_class_names(args.data_yaml)

    print(f"加载模型: {args.model}")
    model = lightly_train.load_model(args.model)  # 自动选 GPU/CPU；本地 .pt 或仓库名都行

    images = collect_images(args.image)
    if not images:
        raise FileNotFoundError(f"未找到图片: {args.image}")
    print(f"待推理图片: {len(images)} 张\n")

    # 模型的 tile 尺寸；比它小的图需要先放大再切片（见 run_sahi）。
    tile_size = getattr(model, "image_size", (640, 640))

    for img_path in images:
        sahi = run_sahi(model, img_path, args, tile_size)

        line = f"[{img_path.name}] SAHI 检出 {len(sahi['labels'])} 个框"
        if args.compare_plain:
            from PIL import Image

            with Image.open(img_path) as _im:
                plain_img = _im.convert("RGB")  # 同样转 RGB，避免 4 通道报错
            plain = model.predict(image=plain_img, threshold=args.threshold)  # type: ignore[call-arg]
            line += f" | 普通 predict 检出 {len(plain['labels'])} 个框"
        print(line)

        save_path = out_dir / f"{img_path.stem}_sahi.jpg"
        draw_and_save(img_path, sahi, class_names, save_path)

    print(f"\n完成，标注图已保存到: {out_dir}/")


# ===========================================================================
# 辅助函数（一般不用改）
# ===========================================================================
def build_args():
    import argparse

    p = argparse.ArgumentParser(description="SAHI 切片推理")
    p.add_argument("--model", default=MODEL)
    p.add_argument("--image", default=IMAGE)
    p.add_argument("--output", default=OUTPUT_DIR)
    p.add_argument("--data-yaml", dest="data_yaml", default=DATA_YAML)
    p.add_argument("--threshold", type=float, default=THRESHOLD)
    p.add_argument("--overlap", type=float, default=OVERLAP)
    p.add_argument("--nms-iou", dest="nms_iou", type=float, default=NMS_IOU)
    p.add_argument(
        "--global-local-iou",
        dest="global_local_iou",
        type=float,
        default=GLOBAL_LOCAL_IOU,
    )
    p.add_argument(
        "--no-compare", dest="compare_plain", action="store_false", default=COMPARE_PLAIN
    )
    return p.parse_args()


def run_sahi(model, img_path: Path, args, tile_size) -> dict:
    """对单张图跑 SAHI 切片推理。

    lightly_train 的 tile_image 在「图比 tile 小」时会对 uint8 张量做 bilinear 插值，
    而 F.interpolate 不支持整型 dtype，会直接报错。为了不改动 lightly_train 源码，
    这里先用 PIL 把过小的图放大到至少一个 tile 大小（保持 uint8），切片推理后再把框
    坐标按相同比例缩回原图尺寸。大图不受影响，走原始路径。
    """
    from PIL import Image

    with Image.open(img_path) as im:
        # 统一转 RGB：避免 RGBA(4 通道)/灰度(1 通道) 与模型 3 通道归一化不匹配而报错。
        im = im.convert("RGB")
        w, h = im.size
        tile_h, tile_w = int(tile_size[0]), int(tile_size[1])
        if h >= tile_h and w >= tile_w:
            image_arg = im.copy()  # 大图也传转好 RGB 的图，而不是原始路径
            scale = 1.0
        else:
            import math

            scale = max(tile_h / h, tile_w / w)
            new_w, new_h = math.ceil(w * scale), math.ceil(h * scale)
            image_arg = im.resize((new_w, new_h), Image.Resampling.BILINEAR)

    sahi = model.predict_sahi(  # type: ignore[call-arg]
        image=image_arg,
        threshold=args.threshold,
        overlap=args.overlap,
        nms_iou_threshold=args.nms_iou,
        global_local_iou_threshold=args.global_local_iou,
    )
    if scale != 1.0 and len(sahi["bboxes"]) > 0:
        sahi["bboxes"] = sahi["bboxes"] / scale  # 框坐标缩回原图
    return sahi


def collect_images(path: str) -> list[Path]:
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
    p = Path(path)
    if p.is_file():
        return [p]
    if p.is_dir():
        return sorted(f for f in p.rglob("*") if f.suffix.lower() in exts)
    return []


def load_class_names(data_yaml: str | None) -> dict[int, str]:
    if not data_yaml or not Path(data_yaml).exists():
        return {}
    import yaml

    with open(data_yaml, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    names = cfg.get("names", {}) if isinstance(cfg, dict) else {}
    if isinstance(names, list):
        return {i: str(n) for i, n in enumerate(names)}
    return {int(k): str(v) for k, v in names.items()}


def draw_and_save(
    img_path: Path, pred: dict, class_names: dict[int, str], save_path: Path
) -> None:
    from PIL import Image, ImageDraw

    img = Image.open(img_path).convert("RGB")
    draw = ImageDraw.Draw(img)

    labels = pred["labels"].cpu().tolist()
    boxes = pred["bboxes"].cpu().tolist()
    scores = pred["scores"].cpu().tolist()

    for label, box, score in zip(labels, boxes, scores):
        x1, y1, x2, y2 = box
        draw.rectangle([x1, y1, x2, y2], outline=(255, 0, 0), width=2)
        name = class_names.get(int(label), str(label))
        draw.text((x1, max(0, y1 - 12)), f"{name} {score:.2f}", fill=(255, 255, 0))

    img.save(save_path)


if __name__ == "__main__":
    main()
