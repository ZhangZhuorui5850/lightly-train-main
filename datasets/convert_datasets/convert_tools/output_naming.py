"""Shared output-directory naming for dataset conversion operations."""

from __future__ import annotations

from pathlib import Path


OPERATION_SUFFIXES = {
    "to-semantic": "to_semantic",
    "to-mvtec": "to_mvtec_ad",
    "to-yolo": "to_yolo",
    "to-yolo-seg": "to_yolo_seg",
    "mvtec-to-yolo": "mvtec_to_yolo",
    "mirror-seg": "mirror_seg_subset",
    "sample-preview": "sample_preview",
    "to-mvtec-object": "to_mvtec_ad_object",
}


def operation_suffix(operation: str) -> str:
    try:
        return OPERATION_SUFFIXES[operation]
    except KeyError as exc:
        raise ValueError(f"未知数据集操作: {operation}") from exc


def default_output_dir(
    source: Path,
    operation: str,
    *,
    parent: Path | None = None,
) -> Path:
    """Return ``<source-name>__<operation>`` under the source's parent directory."""
    source = source.expanduser().resolve()
    output_parent = (parent or source.parent).expanduser().resolve()
    return output_parent / f"{source.name}__{operation_suffix(operation)}"
