#!/usr/bin/env python3
"""为 WIDER Face YOLO 数据生成定向 Copy-Paste、训练切片或组合训练集。"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import shutil
import sys
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path

import yaml
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter, ImageStat

try:
    from dataset_transaction import staged_output, validate_output_location
    from progress import tqdm
except ImportError:  # pragma: no cover - package-style import
    from .dataset_transaction import staged_output, validate_output_location
    from .progress import tqdm


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


@dataclass(frozen=True)
class Box:
    class_id: int
    x: float
    y: float
    w: float
    h: float

    def xyxy(self, width: int, height: int) -> tuple[float, float, float, float]:
        return (
            (self.x - self.w / 2) * width,
            (self.y - self.h / 2) * height,
            (self.x + self.w / 2) * width,
            (self.y + self.h / 2) * height,
        )

    def short_side_at(self, image_size: int = 640) -> float:
        return min(self.w, self.h) * image_size


@dataclass(frozen=True)
class Sample:
    image: Path
    label: Path
    width: int
    height: int
    boxes: tuple[Box, ...]


@dataclass(frozen=True)
class Donor:
    sample_index: int
    box_index: int


def read_boxes(path: Path) -> tuple[Box, ...]:
    boxes: list[Box] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        parts = line.split()
        if len(parts) != 5:
            raise ValueError(f"标签列数错误: {path}:{line_number}")
        class_value = float(parts[0])
        if not class_value.is_integer():
            raise ValueError(f"类别ID需要为整数: {path}:{line_number}: {parts[0]}")
        class_id = int(class_value)
        x, y, w, h = map(float, parts[1:])
        edges = (x - w / 2, y - h / 2, x + w / 2, y + h / 2)
        if (
            class_id != 0
            or w <= 0
            or h <= 0
            or not all(0 <= value <= 1 for value in (x, y, w, h))
            or not all(-1e-6 <= value <= 1 + 1e-6 for value in edges)
        ):
            raise ValueError(f"非法YOLO框: {path}:{line_number}: {line}")
        boxes.append(Box(class_id, x, y, w, h))
    return tuple(boxes)


def scan_split(source: Path, split: str) -> list[Sample]:
    image_dir = source / "images" / split
    label_dir = source / "labels" / split
    if not image_dir.is_dir() or not label_dir.is_dir():
        raise FileNotFoundError(f"缺少 {split} 目录: {image_dir} / {label_dir}")
    images = sorted(path for path in image_dir.iterdir() if path.suffix.lower() in IMAGE_SUFFIXES)
    if len({path.stem for path in images}) != len(images):
        raise ValueError(f"{image_dir} 中存在同名不同扩展名图片")
    samples: list[Sample] = []
    for image_path in tqdm(images, desc=f"扫描{split}", unit="张"):
        label_path = label_dir / f"{image_path.stem}.txt"
        if not label_path.is_file():
            raise FileNotFoundError(f"图片缺少标签: {image_path}")
        with Image.open(image_path) as image:
            width, height = image.size
            image.verify()
        samples.append(Sample(image_path, label_path, width, height, read_boxes(label_path)))
    extra_labels = {path.stem for path in label_dir.glob("*.txt")} - {
        path.stem for path in images
    }
    if extra_labels:
        raise ValueError(f"{label_dir} 中有 {len(extra_labels)} 个标签缺少图片")
    return samples


def write_boxes(path: Path, boxes: list[Box] | tuple[Box, ...]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = "".join(
        f"{box.class_id} {box.x:.8f} {box.y:.8f} {box.w:.8f} {box.h:.8f}\n"
        for box in boxes
    )
    path.write_text(text, encoding="utf-8")


def link_or_copy(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(source, destination)
    except OSError:
        shutil.copy2(source, destination)


def copy_base_dataset(stage: Path, splits: dict[str, list[Sample]]) -> None:
    for split, samples in splits.items():
        for sample in tqdm(samples, desc=f"复制{split}", unit="张"):
            link_or_copy(sample.image, stage / "images" / split / sample.image.name)
            label_destination = stage / "labels" / split / sample.label.name
            label_destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(sample.label, label_destination)
    (stage / "data.yaml").write_text(
        'path: .\ntrain: images/train\nval: images/val\ntest: images/test\n\ntask: detect\nnc: 1\nnames:\n  0: "face"\n',
        encoding="utf-8",
    )


def iou(left: tuple[float, float, float, float], right: tuple[float, float, float, float]) -> float:
    x1 = max(left[0], right[0])
    y1 = max(left[1], right[1])
    x2 = min(left[2], right[2])
    y2 = min(left[3], right[3])
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    left_area = max(0.0, left[2] - left[0]) * max(0.0, left[3] - left[1])
    right_area = max(0.0, right[2] - right[0]) * max(0.0, right[3] - right[1])
    union = left_area + right_area - intersection
    return intersection / union if union else 0.0


def covered_fraction(
    candidate: tuple[float, float, float, float],
    other: tuple[float, float, float, float],
) -> float:
    x1, y1 = max(candidate[0], other[0]), max(candidate[1], other[1])
    x2, y2 = min(candidate[2], other[2]), min(candidate[3], other[3])
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    candidate_area = max(0.0, candidate[2] - candidate[0]) * max(0.0, candidate[3] - candidate[1])
    return intersection / candidate_area if candidate_area else 0.0


def build_donors(samples: list[Sample], min_side: float, max_side: float) -> list[Donor]:
    donors: list[Donor] = []
    for sample_index, sample in enumerate(samples):
        for box_index, box in enumerate(sample.boxes):
            aspect = box.w / box.h
            x1, y1, x2, y2 = box.xyxy(sample.width, sample.height)
            if (
                min_side <= box.short_side_at() <= max_side
                and 0.65 <= aspect <= 1.55
                and x1 >= 2
                and y1 >= 2
                and x2 <= sample.width - 2
                and y2 <= sample.height - 2
            ):
                donors.append(Donor(sample_index, box_index))
    return donors


def crop_donor(sample: Sample, box: Box, context: float) -> tuple[Image.Image, tuple[float, float, float, float], float]:
    with Image.open(sample.image) as source:
        image = source.convert("RGB")
    x1, y1, x2, y2 = box.xyxy(sample.width, sample.height)
    pad_x = (x2 - x1) * context
    pad_y = (y2 - y1) * context
    crop_x1 = max(0, math.floor(x1 - pad_x))
    crop_y1 = max(0, math.floor(y1 - pad_y))
    crop_x2 = min(sample.width, math.ceil(x2 + pad_x))
    crop_y2 = min(sample.height, math.ceil(y2 + pad_y))
    crop = image.crop((crop_x1, crop_y1, crop_x2, crop_y2))
    edge_variance = ImageStat.Stat(crop.convert("L").filter(ImageFilter.FIND_EDGES)).var[0]
    return crop, (x1 - crop_x1, y1 - crop_y1, x2 - crop_x1, y2 - crop_y1), edge_variance


def paste_patch(
    recipient: Image.Image,
    donor_crop: Image.Image,
    donor_face: tuple[float, float, float, float],
    existing: list[Box],
    rng: random.Random,
    target_side: float,
    max_iou: float,
) -> tuple[Box, tuple[int, int, int, int]] | None:
    width, height = recipient.size
    face_w = donor_face[2] - donor_face[0]
    face_h = donor_face[3] - donor_face[1]
    current_equiv = min(face_w * 640 / width, face_h * 640 / height)
    scale = target_side / max(current_equiv, 1e-6)
    patch_w = max(3, round(donor_crop.width * scale))
    patch_h = max(3, round(donor_crop.height * scale))
    if patch_w >= width or patch_h >= height:
        return None
    patch = donor_crop.resize((patch_w, patch_h), Image.Resampling.LANCZOS)
    scaled_face = tuple(value * scale for value in donor_face)
    existing_xyxy = [box.xyxy(width, height) for box in existing]
    face_centers_x = [box.x * width for box in existing]
    face_centers_y = [box.y * height for box in existing]
    for _ in range(50):
        if face_centers_x:
            span_min = max(0.0, min(face_centers_x) - width * 0.18)
            span_max = min(float(width), max(face_centers_x) + width * 0.18)
            center_x = rng.uniform(span_min, span_max)
            left = round(center_x - (scaled_face[0] + scaled_face[2]) / 2)
            left = min(max(0, left), width - patch_w)
        else:
            left = rng.randint(0, width - patch_w)
        if face_centers_y:
            center_y = rng.gauss(sum(face_centers_y) / len(face_centers_y), height * 0.16)
            top = round(center_y - (scaled_face[1] + scaled_face[3]) / 2)
            top = min(max(0, top), height - patch_h)
        else:
            top = rng.randint(0, height - patch_h)
        face_xyxy = (
            left + scaled_face[0],
            top + scaled_face[1],
            left + scaled_face[2],
            top + scaled_face[3],
        )
        if any(
            iou(face_xyxy, other) > max_iou or covered_fraction(face_xyxy, other) > 0.10
            for other in existing_xyxy
        ):
            continue
        background = recipient.crop((left, top, left + patch_w, top + patch_h)).convert("L")
        source_mean = ImageStat.Stat(patch.convert("L")).mean[0]
        target_mean = ImageStat.Stat(background).mean[0]
        brightness = min(1.25, max(0.75, target_mean / max(source_mean, 1.0)))
        adjusted = ImageEnhance.Brightness(patch).enhance(brightness)
        feather = max(1, min(4, min(patch_w, patch_h) // 8))
        mask = Image.new("L", adjusted.size, 0)
        face_pad_x = (scaled_face[2] - scaled_face[0]) * 0.06
        face_pad_y = (scaled_face[3] - scaled_face[1]) * 0.06
        ImageDraw.Draw(mask).ellipse(
            (
                max(0, scaled_face[0] - face_pad_x),
                max(0, scaled_face[1] - face_pad_y),
                min(patch_w - 1, scaled_face[2] + face_pad_x),
                min(patch_h - 1, scaled_face[3] + face_pad_y),
            ),
            fill=255,
        )
        mask = mask.filter(ImageFilter.GaussianBlur(max(0.8, feather / 2)))
        recipient.paste(adjusted, (left, top), mask)
        x1, y1, x2, y2 = face_xyxy
        box = Box(0, (x1 + x2) / (2 * width), (y1 + y2) / (2 * height), (x2 - x1) / width, (y2 - y1) / height)
        return box, (left, top, left + patch_w, top + patch_h)
    return None


def generate_copy_paste(
    samples: list[Sample],
    stage: Path,
    manifest,
    rng: random.Random,
    *,
    fraction: float,
    recipient_min_boxes: int,
    recipient_max_mean_side: float,
    donor_min: float,
    donor_max: float,
    target_min: float,
    target_max: float,
    max_boxes: int,
    max_iou: float,
    min_sharpness: float,
) -> list[Path]:
    recipients = [
        (index, sample)
        for index, sample in enumerate(samples)
        if recipient_min_boxes <= len(sample.boxes) <= 5
        and all(box.short_side_at() >= 32 for box in sample.boxes)
        and sum(box.short_side_at() for box in sample.boxes) / len(sample.boxes)
        <= recipient_max_mean_side
    ]
    donors = build_donors(samples, donor_min, donor_max)
    if not donors:
        raise ValueError("符合尺寸、边界和宽高比条件的Copy-Paste供体数量为0")
    rng.shuffle(recipients)
    requested = round(len(recipients) * fraction)
    if requested == 0:
        raise ValueError("符合条件的Copy-Paste接收图片数量为0")
    recipients = recipients[:requested]
    donor_use: Counter[Donor] = Counter()
    outputs: list[Path] = []
    progress = tqdm(recipients, desc="生成Copy-Paste", unit="张")
    for output_index, (sample_index, sample) in enumerate(progress, 1):
        with Image.open(sample.image) as source:
            image = source.convert("RGB")
        boxes = list(sample.boxes)
        records: list[dict[str, object]] = []
        used_donors: set[Donor] = set()
        paste_count = min(rng.randint(1, 3), max_boxes - len(boxes))
        for _ in range(paste_count):
            for _attempt in range(80):
                donor = rng.choice(donors)
                if (
                    donor.sample_index == sample_index
                    or donor in used_donors
                    or donor_use[donor] >= 5
                ):
                    continue
                donor_sample = samples[donor.sample_index]
                donor_box = donor_sample.boxes[donor.box_index]
                crop, face, sharpness = crop_donor(donor_sample, donor_box, 0.10)
                if sharpness < min_sharpness:
                    continue
                target_side = rng.uniform(target_min, target_max)
                pasted = paste_patch(image, crop, face, boxes, rng, target_side, max_iou)
                if pasted is None:
                    continue
                new_box, patch_xyxy = pasted
                boxes.append(new_box)
                donor_use[donor] += 1
                used_donors.add(donor)
                records.append(
                    {
                        "donor_image": donor_sample.image.name,
                        "donor_box_index": donor.box_index,
                        "source_box": asdict(donor_box),
                        "target_box": asdict(new_box),
                        "patch_xyxy": patch_xyxy,
                        "sharpness": round(sharpness, 3),
                    }
                )
                break
        if not records:
            continue
        output_name = f"cp_{output_index:06d}_{sample.image.stem}.jpg"
        output_image = stage / "images" / "train" / output_name
        output_label = stage / "labels" / "train" / f"{Path(output_name).stem}.txt"
        image.save(output_image, quality=95, subsampling=0)
        write_boxes(output_label, boxes)
        manifest.write(
            json.dumps(
                {
                    "kind": "copy_paste",
                    "output_image": output_name,
                    "recipient_image": sample.image.name,
                    "recipient_index": sample_index,
                    "pastes": records,
                },
                ensure_ascii=False,
            )
            + "\n"
        )
        outputs.append(output_image)
        progress.set_postfix_str(f"有效={len(outputs)}", refresh=False)
    return outputs


def intersection_ratio(
    box: tuple[float, float, float, float], crop: tuple[int, int, int, int]
) -> tuple[float, tuple[float, float, float, float]]:
    x1, y1 = max(box[0], crop[0]), max(box[1], crop[1])
    x2, y2 = min(box[2], crop[2]), min(box[3], crop[3])
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area = max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])
    return (intersection / area if area else 0.0), (x1, y1, x2, y2)


def make_tile(
    sample: Sample,
    target: Box,
    rng: random.Random,
    tile_min: int,
    tile_max: int,
    max_boxes: int,
) -> tuple[Image.Image, list[Box], tuple[int, int, int, int]] | None:
    target_xyxy = target.xyxy(sample.width, sample.height)
    for _ in range(30):
        side = min(rng.randint(tile_min, tile_max), sample.width, sample.height)
        if side < 64:
            return None
        center_x = (target_xyxy[0] + target_xyxy[2]) / 2 + rng.uniform(-0.15, 0.15) * side
        center_y = (target_xyxy[1] + target_xyxy[3]) / 2 + rng.uniform(-0.15, 0.15) * side
        left = min(max(0, round(center_x - side / 2)), sample.width - side)
        top = min(max(0, round(center_y - side / 2)), sample.height - side)
        crop = (left, top, left + side, top + side)
        boxes: list[Box] = []
        for box in sample.boxes:
            ratio, clipped = intersection_ratio(box.xyxy(sample.width, sample.height), crop)
            if ratio < 0.50:
                continue
            x1, y1, x2, y2 = clipped
            boxes.append(
                Box(
                    box.class_id,
                    ((x1 + x2) / 2 - left) / side,
                    ((y1 + y2) / 2 - top) / side,
                    (x2 - x1) / side,
                    (y2 - y1) / side,
                )
            )
        target_short = min((target_xyxy[2] - target_xyxy[0]) * 640 / side, (target_xyxy[3] - target_xyxy[1]) * 640 / side)
        full_short = target.short_side_at()
        if 1 <= len(boxes) <= max_boxes and target_short >= 12 and target_short >= full_short * 1.25:
            with Image.open(sample.image) as source:
                image = source.convert("RGB").crop(crop)
            return image, boxes, crop
    return None


def generate_tiles(
    samples: list[Sample],
    stage: Path,
    manifest,
    rng: random.Random,
    *,
    count: int,
    tile_min: int,
    tile_max: int,
    max_boxes: int,
) -> list[Path]:
    candidates = [
        (sample_index, box_index)
        for sample_index, sample in enumerate(samples)
        for box_index, box in enumerate(sample.boxes)
        if box.short_side_at() < 32
    ]
    rng.shuffle(candidates)
    per_image: Counter[int] = Counter()
    accepted_crops: dict[int, list[tuple[int, int, int, int]]] = {}
    outputs: list[Path] = []
    attempts = 0
    with tqdm(total=count, desc="生成Tile", unit="张") as progress:
        for sample_index, box_index in candidates:
            if len(outputs) >= count:
                break
            if per_image[sample_index] >= 3:
                continue
            attempts += 1
            sample = samples[sample_index]
            result = make_tile(sample, sample.boxes[box_index], rng, tile_min, tile_max, max_boxes)
            if result is None:
                continue
            image, boxes, crop = result
            if any(iou(crop, previous) > 0.90 for previous in accepted_crops.get(sample_index, [])):
                continue
            accepted_crops.setdefault(sample_index, []).append(crop)
            per_image[sample_index] += 1
            output_name = f"tile_{len(outputs) + 1:06d}_{sample.image.stem}.jpg"
            output_image = stage / "images" / "train" / output_name
            output_label = stage / "labels" / "train" / f"{Path(output_name).stem}.txt"
            image.save(output_image, quality=95, subsampling=0)
            write_boxes(output_label, boxes)
            manifest.write(
                json.dumps(
                    {
                        "kind": "tile",
                        "output_image": output_name,
                        "source_image": sample.image.name,
                        "source_index": sample_index,
                        "crop_xyxy": crop,
                        "boxes": len(boxes),
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
            outputs.append(output_image)
            progress.update()
            progress.set_postfix_str(f"尝试={attempts}", refresh=False)
    if len(outputs) < count:
        print(f"  Tile有效样本达到 {len(outputs)}/{count}，候选尝试 {attempts}")
    return outputs


def draw_preview(stage: Path, generated: list[Path], seed: int, limit: int) -> None:
    if not generated or limit <= 0:
        return
    rng = random.Random(seed + 991)
    selected = rng.sample(generated, min(limit, len(generated)))
    thumb_size = 256
    columns = 5
    rows = math.ceil(len(selected) / columns)
    sheet = Image.new("RGB", (columns * thumb_size, rows * thumb_size), "white")
    for index, image_path in enumerate(selected):
        with Image.open(image_path) as source:
            image = source.convert("RGB")
        label_path = stage / "labels" / "train" / f"{image_path.stem}.txt"
        draw = ImageDraw.Draw(image)
        for box in read_boxes(label_path):
            draw.rectangle(box.xyxy(image.width, image.height), outline=(255, 40, 40), width=max(1, image.width // 400))
        image.thumbnail((thumb_size, thumb_size))
        tile = Image.new("RGB", (thumb_size, thumb_size), "white")
        tile.paste(image, ((thumb_size - image.width) // 2, (thumb_size - image.height) // 2))
        sheet.paste(tile, ((index % columns) * thumb_size, (index // columns) * thumb_size))
    sheet.save(stage / "augmentation_preview.jpg", quality=92)


def validate_stage(stage: Path, generated: list[Path]) -> dict[str, dict[str, int]]:
    result: dict[str, dict[str, int]] = {}
    for split in ("train", "val", "test"):
        image_dir = stage / "images" / split
        label_dir = stage / "labels" / split
        images = [path for path in image_dir.iterdir() if path.suffix.lower() in IMAGE_SUFFIXES]
        labels = list(label_dir.glob("*.txt"))
        if {path.stem for path in images} != {path.stem for path in labels}:
            raise ValueError(f"{split} 图片和标签文件名集合不一致")
        box_count = sum(
            len(read_boxes(path))
            for path in tqdm(labels, desc=f"校验{split}", unit="份")
        )
        result[split] = {"images": len(images), "boxes": box_count}
    for image_path in generated:
        with Image.open(image_path) as image:
            image.verify()
    return result


def plan(samples: list[Sample], args: argparse.Namespace) -> dict[str, int]:
    recipients = sum(
        args.recipient_min_boxes <= len(sample.boxes) <= 5
        and all(box.short_side_at() >= 32 for box in sample.boxes)
        and sum(box.short_side_at() for box in sample.boxes) / len(sample.boxes)
        <= args.recipient_max_mean_side
        for sample in samples
    )
    donors = len(build_donors(samples, args.donor_min_side, args.donor_max_side))
    tile_candidates = sum(box.short_side_at() < 32 for sample in samples for box in sample.boxes)
    return {
        "train_images": len(samples),
        "train_boxes": sum(len(sample.boxes) for sample in samples),
        "copy_paste_recipients": recipients,
        "planned_copy_paste": round(recipients * args.recipient_fraction),
        "donors": donors,
        "tile_candidates": tile_candidates,
        "planned_tiles": args.tile_count or round(len(samples) * args.tile_ratio),
    }


def run(args: argparse.Namespace) -> Path:
    validate_args(args)
    source = args.src.expanduser().resolve()
    if not (source / "data.yaml").is_file():
        raise FileNotFoundError(f"缺少数据配置: {source / 'data.yaml'}")
    config = yaml.safe_load((source / "data.yaml").read_text(encoding="utf-8"))
    if int(config.get("nc", 0)) != 1:
        raise ValueError("当前工具只处理单类别 WIDER Face YOLO 数据集")
    output = validate_output_location(args.out, [source])
    splits = {split: scan_split(source, split) for split in ("train", "val", "test")}
    stats = plan(splits["train"], args)
    if args.mode in {"copy-paste", "combined"}:
        if stats["donors"] == 0:
            raise ValueError("符合条件的Copy-Paste供体数量为0")
        if stats["planned_copy_paste"] == 0:
            raise ValueError("符合条件的Copy-Paste接收图片数量为0")
    if args.mode in {"tile", "combined"} and stats["planned_tiles"] == 0:
        raise ValueError("计划生成的Tile数量为0")
    print(json.dumps(stats, ensure_ascii=False, indent=2), flush=True)
    if args.dry_run:
        print("dry-run完成，输出目录保持原样。", flush=True)
        return output

    rng = random.Random(args.seed)
    with staged_output(output, clean=args.clean) as stage:
        copy_base_dataset(stage, splits)
        generated: list[Path] = []
        with (stage / "augmentation_manifest.jsonl").open("w", encoding="utf-8") as manifest:
            if args.mode in {"copy-paste", "combined"}:
                generated.extend(
                    generate_copy_paste(
                        splits["train"], stage, manifest, rng,
                        fraction=args.recipient_fraction,
                        recipient_min_boxes=args.recipient_min_boxes,
                        recipient_max_mean_side=args.recipient_max_mean_side,
                        donor_min=args.donor_min_side,
                        donor_max=args.donor_max_side,
                        target_min=args.target_min_side,
                        target_max=args.target_max_side,
                        max_boxes=args.copy_paste_max_boxes,
                        max_iou=args.max_iou,
                        min_sharpness=args.min_sharpness,
                    )
                )
            if args.mode in {"tile", "combined"}:
                generated.extend(
                    generate_tiles(
                        splits["train"], stage, manifest, rng,
                        count=stats["planned_tiles"],
                        tile_min=args.tile_min,
                        tile_max=args.tile_max,
                        max_boxes=args.tile_max_boxes,
                    )
                )
        draw_preview(stage, generated, args.seed, args.preview_count)
        verified = validate_stage(stage, generated)
        summary = {
            "mode": args.mode,
            "source": str(source),
            "seed": args.seed,
            "parameters": vars(args) | {"src": str(args.src), "out": str(args.out)},
            "plan": stats,
            "generated_images": len(generated),
            "generated_by_kind": {
                "copy_paste": sum(path.name.startswith("cp_") for path in generated),
                "tile": sum(path.name.startswith("tile_") for path in generated),
            },
            "storage": {"base_images": "hardlink_or_copy", "base_labels": "copy"},
            "verified": verified,
        }
        (stage / "preparation_summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2, default=str) + "\n",
            encoding="utf-8",
        )
    print(f"已生成: {output}", flush=True)
    print(json.dumps(verified, ensure_ascii=False, indent=2), flush=True)
    return output


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("copy-paste", "tile", "combined"), required=True)
    parser.add_argument("--src", type=Path, default=Path("datasets/face_yolo_wider"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--recipient-fraction", type=float, default=0.70)
    parser.add_argument("--recipient-min-boxes", type=int, default=2)
    parser.add_argument("--recipient-max-mean-side", type=float, default=128.0)
    parser.add_argument("--donor-min-side", type=float, default=24.0)
    parser.add_argument("--donor-max-side", type=float, default=96.0)
    parser.add_argument("--target-min-side", type=float, default=12.0)
    parser.add_argument("--target-max-side", type=float, default=28.0)
    parser.add_argument("--copy-paste-max-boxes", type=int, default=20)
    parser.add_argument("--max-iou", type=float, default=0.05)
    parser.add_argument("--min-sharpness", type=float, default=8.0)
    parser.add_argument("--tile-ratio", type=float, default=0.50)
    parser.add_argument("--tile-count", type=int, default=0)
    parser.add_argument("--tile-min", type=int, default=512)
    parser.add_argument("--tile-max", type=int, default=768)
    parser.add_argument("--tile-max-boxes", type=int, default=100)
    parser.add_argument("--preview-count", type=int, default=25)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--clean", action="store_true")
    return parser


def validate_args(args: argparse.Namespace) -> None:
    if not 0 < args.recipient_fraction <= 1:
        raise ValueError("--recipient-fraction 需要位于 (0, 1]")
    if args.recipient_min_boxes < 1 or args.recipient_min_boxes > 5:
        raise ValueError("--recipient-min-boxes 需要位于 [1, 5]")
    if args.recipient_max_mean_side <= 0:
        raise ValueError("--recipient-max-mean-side需要为正数")
    if args.donor_min_side <= 0 or args.target_min_side <= 0:
        raise ValueError("目标尺寸需要为正数")
    if args.donor_min_side > args.donor_max_side or args.target_min_side > args.target_max_side:
        raise ValueError("尺寸范围下限需要小于等于上限")
    if args.copy_paste_max_boxes <= 5 or args.tile_max_boxes < 1:
        raise ValueError("框数量上限过小")
    if not 0 <= args.max_iou <= 1:
        raise ValueError("--max-iou 需要位于 [0, 1]")
    if args.min_sharpness < 0 or args.tile_ratio < 0 or args.tile_count < 0:
        raise ValueError("锐度、Tile比例和Tile数量需要为非负数")
    if args.tile_min < 64 or args.tile_min > args.tile_max:
        raise ValueError("Tile尺寸需要满足 64 <= min <= max")
    if args.preview_count < 0:
        raise ValueError("--preview-count需要为非负数")


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    run(args)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileExistsError, FileNotFoundError, ValueError, RuntimeError) as exc:
        print(f"错误: {exc}", file=sys.stderr)
        raise SystemExit(2)
