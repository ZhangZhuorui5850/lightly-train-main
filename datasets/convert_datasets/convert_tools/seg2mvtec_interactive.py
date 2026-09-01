#!/usr/bin/env python3
"""Interactive YOLO-seg -> MVTec AD converter.

What it does
------------
1. Scans `datasets/` and identifies YOLO polygon labels from their content.
2. CONTENT-validates each candidate by actually reading its labels:
     * polygons (>=3 points / >=6 coords)  -> YOLO-seg, convertible
     * exactly 4 coords per line (cx cy w h) -> detection bbox, REFUSED
   So a mis-named `_seg` folder that really holds det boxes is rejected.
3. Shows a menu; you pick which to convert (`1`, `1,3`, or `all`).
4. Writes a NEW dataset next to each source using the shared operation suffix
   (`dataset_seg` -> `dataset_seg__to_mvtec_ad`). The original
   is never modified. Refuses to overwrite an existing output unless you
   confirm.

Run it (interactive):
    python datasets/convert_datasets/seg2mvtec_interactive.py

Non-interactive (convert everything it finds, no prompts):
    python datasets/convert_datasets/seg2mvtec_interactive.py --all --yes
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Reuse the validated core conversion logic.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from yoloseg_to_mvtec import convert, load_names  # noqa: E402
from text_encoding import read_text_auto  # noqa: E402
from dataset_detector import detect_datasets  # noqa: E402

from output_naming import default_output_dir

SPLITS = ("train", "val", "test")
def sample_label_lines(src: Path, limit: int = 200) -> list[list[str]]:
    """Collect tokenized non-empty label lines from across the splits."""
    lines: list[list[str]] = []
    for split in SPLITS:
        for txt in sorted((src / "labels" / split).glob("*.txt")):
            for ln in read_text_auto(txt).splitlines():
                toks = ln.split()
                if toks:
                    lines.append(toks)
                    if len(lines) >= limit:
                        return lines
    return lines


def inspect(src: Path) -> dict:
    """Classify a candidate dir. kind in {seg, det, empty, invalid}."""
    has_imgs = (src / "images").is_dir()
    has_lbls = (src / "labels").is_dir()
    has_meta = (src / "data.yaml").exists() or (src / "classes.txt").exists()
    if not (has_imgs and has_lbls and has_meta):
        miss = []
        if not has_imgs:
            miss.append("images/")
        if not has_lbls:
            miss.append("labels/")
        if not has_meta:
            miss.append("data.yaml|classes.txt")
        return {"kind": "invalid", "reason": "missing " + ", ".join(miss)}

    lines = sample_label_lines(src)
    if not lines:
        return {"kind": "empty", "reason": "no label content found"}

    # token count per line = 1 (class) + 2*n_points
    poly = sum(1 for t in lines if len(t) >= 7 and len(t) % 2 == 1)
    bbox = sum(1 for t in lines if len(t) == 5)
    n_imgs = sum(len(list((src / "images" / s).glob("*"))) for s in SPLITS)
    try:
        names = load_names(src)
    except SystemExit:
        names = []

    if poly > 0:
        return {"kind": "seg", "reason": f"{poly} polygon lines sampled",
                "n_imgs": n_imgs, "names": names}
    if bbox > 0:
        return {"kind": "det", "reason": f"{bbox} bbox lines (4 coords) -> not segmentation",
                "n_imgs": n_imgs, "names": names}
    return {"kind": "invalid", "reason": "labels are neither polygon nor bbox"}


def out_dir_for(src: Path) -> Path:
    return default_output_dir(src, "to-mvtec")


def scan(datasets_root: Path) -> list[tuple[Path, dict]]:
    candidates = [
        candidate
        for candidate in detect_datasets(
            datasets_root, kinds={"yolo_instance"}
        )
        if candidate.kind == "yolo_instance"
    ]
    return [
        (
            candidate.path,
            {
                "kind": "seg",
                "reason": "polygon 标签内容识别",
                "n_imgs": candidate.image_count,
                "names": [None] * candidate.class_count,
            },
        )
        for candidate in candidates
    ]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", type=Path, default=None,
                    help="datasets root (default: auto-detect ./datasets)")
    ap.add_argument("--all", action="store_true", help="select all convertible without prompting")
    ap.add_argument("--yes", action="store_true", help="接受默认选择和输出替换")
    ap.add_argument("--dry-run", action="store_true", help="扫描并显示全部转换计划，保持零写入")
    args = ap.parse_args(argv)

    here = Path(__file__).resolve()
    root = args.datasets or next(
        (p / "datasets" for p in here.parents if (p / "datasets").is_dir()), Path("datasets")
    )
    root = root.resolve()
    print(f"Scanning YOLO polygon datasets under: {root}\n")

    cands = scan(root)
    if not cands:
        print("No YOLO polygon datasets were found.")
        return 1

    convertible = []
    print(f"{'#':>2}  {'kind':<8}{'classes':>8}{'images':>8}  dataset")
    print("-" * 72)
    for i, (d, info) in enumerate(cands):
        rel = d.relative_to(root)
        nc = len(info.get("names", [])) if info.get("names") else "-"
        ni = info.get("n_imgs", "-")
        tag = {"seg": "OK", "det": "DET-skip", "empty": "empty", "invalid": "invalid"}[info["kind"]]
        idx = "-"
        if info["kind"] == "seg":
            idx = str(len(convertible) + 1)
            convertible.append((d, info))
        print(f"{idx:>2}  {tag:<8}{str(nc):>8}{str(ni):>8}  {rel}")
        if info["kind"] != "seg":
            print(f"        └─ {info['reason']}")
    print("-" * 72)

    if not convertible:
        print("\nNothing convertible (no valid YOLO-seg dataset). det/* is refused by design.")
        return 1

    if args.all:
        picks = list(range(len(convertible)))
    else:
        raw = input(f"\nConvert which? 1-{len(convertible)}, comma-separated, or 'all' (blank=cancel): ").strip()
        if not raw:
            print("Cancelled.")
            return 0
        if raw.lower() == "all":
            picks = list(range(len(convertible)))
        else:
            try:
                picks = [int(x) - 1 for x in raw.replace(" ", "").split(",")]
                assert all(0 <= p < len(convertible) for p in picks)
            except (ValueError, AssertionError):
                print("Bad selection. Aborting.")
                return 2

    for p in picks:
        src, _ = convertible[p]
        out = out_dir_for(src)
        print(f"\n=== {src.name}  ->  {out.name} ===")
        if out.exists() and not args.yes and not args.dry_run:
            ans = input(f"  {out} exists. Overwrite? [y/N]: ").strip().lower()
            if ans != "y":
                print("  skipped.")
                continue
        if args.dry_run:
            print(f"  [dry-run] {src} -> {out}")
        else:
            convert(src, out, clean=True, verbose=True)
            print(f"  done -> {out}")

    print("\nAll selected conversions finished. Originals untouched.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
