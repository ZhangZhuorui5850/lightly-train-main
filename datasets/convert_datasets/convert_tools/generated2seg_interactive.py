#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Interactive launcher for generated ``image/`` + ``fg/`` to YOLO-seg."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from generated_mask_to_yoloseg import (  # noqa: E402
    convert,
    discover_pairs,
    natural_key,
    resolve_class_names,
)
from dataset_discovery import (  # noqa: E402
    auto_datasets_root,
    scan_generated_mask_roots,
)
from output_naming import default_output_dir  # noqa: E402


def scan_candidates(root: Path) -> list[Path]:
    """Group leaves as ``root/object/defect/{image,fg}`` candidate roots."""
    candidates = scan_generated_mask_roots(root)
    return sorted(candidates, key=lambda p: natural_key(str(p.relative_to(root.resolve()))))


def inspect_candidate(path: Path) -> dict:
    try:
        pairs, issues = discover_pairs(path)
    except ValueError as exc:
        return {"pairs": 0, "issues": 1, "objects": [], "defects": [], "error": str(exc)}
    return {
        "pairs": len(pairs),
        "issues": len(issues),
        "objects": sorted({p.object_name for p in pairs}, key=natural_key),
        "defects": sorted({p.defect_name for p in pairs}, key=natural_key),
        "error": "",
    }


def choose_source(root: Path) -> Path | None:
    print(f"扫描 image/ + fg/ 数据: {root}\n")
    candidates = scan_candidates(root)
    valid: list[tuple[Path, dict]] = []
    if candidates:
        print(f"{'#':>3} {'配对':>8} {'问题':>8} {'物体':>8} {'缺陷':>8}  路径")
        print("-" * 88)
        for candidate in candidates:
            info = inspect_candidate(candidate)
            if info["pairs"]:
                valid.append((candidate, info))
                rel = candidate.relative_to(root)
                print(
                    f"{len(valid):>3} {info['pairs']:>8} {info['issues']:>8} "
                    f"{len(info['objects']):>8} {len(info['defects']):>8}  {rel}"
                )
        print("-" * 88)
    if not valid:
        print("扫描范围内暂未发现可配对数据。")

    prompt = "选择序号，输入 p 手动填写路径，回车退出: "
    raw = input(prompt).strip()
    if not raw:
        return None
    if raw.lower() in ("p", "path", "路径"):
        manual = input("输入数据路径: ").strip()
        return Path(manual).expanduser().resolve() if manual else None
    try:
        index = int(raw) - 1
        if 0 <= index < len(valid):
            return valid[index][0]
    except ValueError:
        pass
    print(f"无效选择: {raw}")
    return None


def default_output(src: Path) -> Path:
    return default_output_dir(src, "to-yolo-seg")


def print_analysis(src: Path, names_from: Path | None) -> tuple[list, list, list[str], Path | None]:
    pairs, issues = discover_pairs(src)
    names, config_source = resolve_class_names(src, pairs, names_from)
    objects = sorted({pair.object_name for pair in pairs}, key=natural_key)
    print("\n数据分析")
    print(f"  输入       : {src}")
    print(f"  有效配对   : {len(pairs)}")
    print(f"  配对问题   : {len(issues)}")
    print(f"  物体       : {len(objects)} ({', '.join(objects)})")
    print(f"  类别来源   : {config_source or '缺陷目录自动生成'}")
    print("  类别映射   :")
    for class_id, name in enumerate(names):
        sources = sorted(
            {f"{pair.object_name}/{pair.defect_name}" for pair in pairs if pair.defect_name == name},
            key=natural_key,
        )
        suffix = f"  [{', '.join(sources)}]" if sources else "  [配置保留类别]"
        print(f"    {class_id:>3} -> {name}{suffix}")
    return pairs, issues, names, config_source


def main() -> None:
    parser = argparse.ArgumentParser(description="交互式生成 image/fg -> YOLO Seg")
    parser.add_argument("--src", type=Path, help="输入目录；省略时自动扫描并选择")
    parser.add_argument("--out", type=Path, help="输出目录；省略时交互确认默认路径")
    parser.add_argument("--datasets", type=Path, help="自动扫描根目录，默认仓库 datasets/")
    parser.add_argument("--names-from", type=Path, help="可选 data.yaml/classes.txt")
    parser.add_argument("--threshold", type=int, default=127)
    parser.add_argument("--min-area", type=float, default=1.0)
    parser.add_argument("--epsilon", type=float, default=0.001)
    parser.add_argument("--no-auto-invert", action="store_true")
    parser.add_argument("--clean", action="store_true")
    parser.add_argument("--yes", action="store_true", help="使用默认值并直接执行")
    args = parser.parse_args()

    try:
        if args.src:
            src = args.src.expanduser().resolve()
        else:
            src = choose_source(args.datasets or auto_datasets_root())
            if src is None:
                print("已退出。")
                return

        print_analysis(src, args.names_from)
        default_out = default_output(src).resolve()
        if args.out:
            out = args.out.expanduser().resolve()
        elif args.yes:
            out = default_out
        else:
            raw_out = input(f"\nYOLO Seg 输出目录 [默认 {default_out}]: ").strip()
            out = Path(raw_out).expanduser().resolve() if raw_out else default_out

        clean = args.clean
        if out.exists() and any(out.iterdir()) and not clean:
            if args.yes:
                raise FileExistsError(f"输出目录已存在且含有文件: {out}；使用 --clean 可重建")
            clean = input(f"输出目录已有内容，清空并重建 {out}？[y/N]: ").strip().lower() in (
                "y", "yes", "是",
            )
            if not clean:
                print("已取消。")
                return

        if not args.yes:
            answer = input(f"开始转换到 {out}？[Y/n]: ").strip().lower()
            if answer in ("n", "no", "否"):
                print("已取消。")
                return

        convert(
            src, out, names_from=args.names_from, threshold=args.threshold,
            min_area=args.min_area, epsilon=args.epsilon,
            auto_invert=not args.no_auto_invert, clean=clean, verbose=True,
        )
    except (ValueError, FileExistsError) as exc:
        parser.error(str(exc))
    except (EOFError, KeyboardInterrupt):
        print("\n已退出。")


if __name__ == "__main__":
    main()
