#!/usr/bin/env python3
"""Fast, shared dataset detection frontend for every conversion workflow."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable, Sequence

try:
    from .dataset_discovery import (
        KIND_LABELS,
        DatasetCandidate,
        auto_datasets_root,
        dataset_path_diagnostics,
        find_dataset_config,
        format_candidate_count,
        format_modified_time,
        inspect_config_dataset,
        load_yaml,
        scan_datasets,
    )
except ImportError:
    from dataset_discovery import (  # type: ignore[no-redef]
        KIND_LABELS,
        DatasetCandidate,
        auto_datasets_root,
        dataset_path_diagnostics,
        find_dataset_config,
        format_candidate_count,
        format_modified_time,
        inspect_config_dataset,
        load_yaml,
        scan_datasets,
    )


TASK_KINDS = {
    "cls": {"image_classification"},
    "det": {"yolo_detection"},
    "seg": {"yolo_instance"},
    "semantic": {"semantic_mask"},
    "labelme": {"labelme"},
    "generated": {"generated_mask"},
    "mvtec": {"mvtec"},
    "yolo": {"yolo_detection", "yolo_instance"},
    "all": set(),
}


def _search_roots(search_root: Path | Sequence[Path]) -> list[Path]:
    values = [search_root] if isinstance(search_root, (str, Path)) else list(search_root)
    roots: list[Path] = []
    for value in values:
        root = Path(value).expanduser().resolve()
        if root not in roots:
            roots.append(root)
    return roots


def detect_datasets(
    search_root: Path | Sequence[Path],
    *,
    kinds: Iterable[str] | None = None,
    include_empty: bool = False,
    include_unknown: bool = False,
    show_progress: bool = True,
    count_limit: int | None = 200,
) -> list[DatasetCandidate]:
    """Return compatible candidates in latest-modified-first order."""
    allowed = set(kinds or ())
    result: list[DatasetCandidate] = []
    seen: set[tuple[Path, str]] = set()
    for root in _search_roots(search_root):
        for candidate in scan_datasets(
            root,
            show_progress=show_progress,
            count_limit=count_limit,
            kinds=allowed if allowed else None,
        ):
            if (
                allowed
                and candidate.kind not in allowed
                and not (include_unknown and candidate.kind == "unknown")
            ):
                continue
            if candidate.kind == "unknown" and not include_unknown:
                continue
            if candidate.image_count <= 0 and not include_empty:
                continue
            identity = ((candidate.config_path or candidate.path).resolve(), candidate.kind)
            if identity in seen:
                continue
            seen.add(identity)
            result.append(candidate)
    return sorted(result, key=lambda item: item.modified_time, reverse=True)


def inspect_dataset(
    source: Path,
    *,
    kinds: Iterable[str] | None = None,
    count_limit: int | None = 200,
) -> DatasetCandidate:
    """Inspect one explicit dataset path through the shared detector facade."""
    config_path = find_dataset_config(Path(source))
    candidate = inspect_config_dataset(
        config_path,
        show_progress=False,
        count_limit=count_limit,
    )
    allowed = set(kinds or ())
    if allowed and candidate.kind not in allowed:
        expected = ", ".join(sorted(allowed))
        raise ValueError(
            f"数据集格式为 {candidate.kind}，当前功能需要: {expected}"
        )
    return candidate


def choose_dataset(
    search_root: Path,
    *,
    kinds: Iterable[str],
    title: str,
    include_unknown: bool = False,
) -> DatasetCandidate | None:
    """Shared interactive chooser used by conversion tools."""
    allowed = set(kinds)
    candidates = detect_datasets(
        search_root,
        kinds=allowed,
        include_unknown=include_unknown,
    )
    print(f"\n{title}（最近修改优先）:\n")
    print(
        f"  {'#':>3} {'类型':<18}{'图片':>8}{'标注文件':>10}{'类别':>6}  "
        f"{'split':<16}{'最后修改':<16} 配置/路径"
    )
    for index, candidate in enumerate(candidates, start=1):
        image_count = format_candidate_count(
            candidate.image_count, candidate.image_count_is_exact
        )
        annotation_count = format_candidate_count(
            candidate.annotation_count, candidate.annotation_count_is_exact
        )
        split_text = ",".join(candidate.splits) or "-"
        print(
            f"  {index:>3} {KIND_LABELS.get(candidate.kind, candidate.kind):<18}"
            f"{image_count:>8}{annotation_count:>10}"
            f"{candidate.class_count:>6}  "
            f"{split_text:<16}"
            f"{format_modified_time(candidate.modified_time):<16} "
            f"{candidate.config_path or candidate.path}"
        )
    print("   p. 手动输入 data.yaml 或数据集目录")
    while True:
        try:
            raw = input(f"选择 [1-{len(candidates)}/p/q]: ").strip().casefold()
        except (EOFError, KeyboardInterrupt):
            print()
            return None
        if raw in {"q", "quit", "exit"}:
            return None
        if raw == "p":
            value = input("data.yaml 或数据集目录: ").strip()
            if not value:
                continue
            source = Path(value).expanduser().resolve()
            candidate = inspect_dataset(source, count_limit=200)
            if candidate.kind in allowed or (
                include_unknown and candidate.kind == "unknown"
            ):
                return candidate
            return DatasetCandidate(
                path=source,
                kind="unknown",
                config_path=candidate.config_path,
            )
        if raw.isdigit() and 1 <= int(raw) <= len(candidates):
            return candidates[int(raw) - 1]
        print("请输入有效序号、p 或 q。")


def _candidate_dict(candidate: DatasetCandidate) -> dict[str, object]:
    return {
        "path": str(candidate.path),
        "config_path": str(candidate.config_path) if candidate.config_path else None,
        "kind": candidate.kind,
        "image_count": candidate.image_count,
        "annotation_count": candidate.annotation_count,
        "class_count": candidate.class_count,
        "splits": list(candidate.splits),
        "modified_time": candidate.modified_time,
        "issues": list(candidate.issues),
        "counts_are_exact": candidate.counts_are_exact,
        "image_count_is_exact": candidate.image_count_is_exact,
        "annotation_count_is_exact": candidate.annotation_count_is_exact,
    }


def print_candidates(candidates: Sequence[DatasetCandidate]) -> None:
    print(
        f"{'#':>3} {'类型':<18}{'图片':>9}{'标注文件':>11}{'类别':>7}  "
        f"{'split':<16}{'最后修改':<16} 配置/路径"
    )
    print("-" * 135)
    for index, candidate in enumerate(candidates, start=1):
        issue = f"  [{'; '.join(candidate.issues)}]" if candidate.issues else ""
        image_count = format_candidate_count(
            candidate.image_count, candidate.image_count_is_exact
        )
        annotation_count = format_candidate_count(
            candidate.annotation_count, candidate.annotation_count_is_exact
        )
        split_text = ",".join(candidate.splits) or "-"
        print(
            f"{index:>3} {KIND_LABELS.get(candidate.kind, candidate.kind):<18}"
            f"{image_count:>9}{annotation_count:>11}"
            f"{candidate.class_count:>7}  "
            f"{split_text:<16}"
            f"{format_modified_time(candidate.modified_time):<16} "
            f"{candidate.config_path or candidate.path}{issue}"
        )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--datasets",
        type=Path,
        action="append",
        help="扫描根目录；可重复传入以检索多个磁盘或挂载点",
    )
    parser.add_argument("--task", choices=tuple(TASK_KINDS))
    parser.add_argument("--include-empty", action="store_true")
    parser.add_argument("--include-unknown", action="store_true")
    parser.add_argument(
        "--exact-counts",
        action="store_true",
        help="遍历全部文件并显示精确数量",
    )
    parser.add_argument("--json", action="store_true", help="输出 JSON")
    parser.add_argument(
        "--diagnose",
        type=Path,
        help="诊断一个 data.yaml 或数据集目录的路径解析",
    )
    return parser.parse_args(argv)


def _prompt_task() -> str:
    print("选择要检测的数据集类型:")
    print("  1. Det（YOLO Detection）")
    print("  2. Seg（YOLO Instance Segmentation）")
    print("  3. Semantic（PNG mask）")
    print("  4. Cls（图片分类）")
    print("  5. 全部类型")
    mapping = {
        "1": "det",
        "2": "seg",
        "3": "semantic",
        "4": "cls",
        "5": "all",
        "": "all",
    }
    while True:
        try:
            raw = input("选择 [1-5，默认 5]: ").strip().casefold()
        except (EOFError, KeyboardInterrupt):
            print()
            return "all"
        if raw in mapping:
            return mapping[raw]
        print("请输入 1、2、3、4 或 5。")


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.diagnose is not None:
        config_path = find_dataset_config(args.diagnose)
        config = load_yaml(config_path)
        print(f"配置文件: {config_path}")
        for line in dataset_path_diagnostics(config_path, config):
            print(f"  {line}")
        return 0
    task = args.task or _prompt_task()
    roots = _search_roots(args.datasets or [auto_datasets_root()])
    candidates = detect_datasets(
        roots,
        kinds=TASK_KINDS[task],
        include_empty=args.include_empty,
        include_unknown=args.include_unknown,
        count_limit=None if args.exact_counts else 200,
    )
    if args.json:
        print(
            json.dumps(
                [_candidate_dict(item) for item in candidates],
                ensure_ascii=False,
                indent=2,
            )
        )
    else:
        print("扫描根目录: " + ", ".join(str(root) for root in roots))
        print_candidates(candidates)
        print(f"\n共 {len(candidates)} 个数据集。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
