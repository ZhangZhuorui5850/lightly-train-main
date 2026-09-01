"""Shared output-directory naming for dataset conversion operations."""

from __future__ import annotations

import hashlib
import re
import unicodedata
from pathlib import Path


OPERATION_SUFFIXES = {
    "to-semantic": "to_semantic",
    "edit-classes": "edit_classes",
    "edit-semantic": "edit_semantic",
    "merge-yolo": "merge_yolo",
    "merge-semantic": "merge_semantic",
    "merge-datasets": "merge_datasets",
    "to-mvtec": "to_mvtec_ad",
    "to-yolo": "to_yolo",
    "to-yolo-seg": "to_yolo_seg",
    "mvtec-to-yolo": "mvtec_to_yolo",
    "mirror-seg": "mirror_seg_subset",
    "sample-preview": "sample_preview",
    "to-mvtec-object": "to_mvtec_ad_object",
    "balanced-test": "balanced_test",
}

_WINDOWS_RESERVED = {
    "CON", "PRN", "AUX", "NUL",
    *(f"COM{index}" for index in range(1, 10)),
    *(f"LPT{index}" for index in range(1, 10)),
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


def flat_sample_stem(stem: str) -> str:
    """Flatten one logical relative stem into a direct-child filename stem."""
    parts = [
        part
        for part in stem.replace("\\", "/").split("/")
        if part and part not in {".", ".."}
    ]
    return "__".join(parts) or "sample"


def allocate_flat_sample_stem(stem: str, used: set[str]) -> str:
    """Allocate a case-insensitively unique flat stem and preserve free names."""
    base = flat_sample_stem(stem)
    candidate = base
    index = 2
    while candidate.casefold() in used:
        candidate = f"{base}__{index}"
        index += 1
    used.add(candidate.casefold())
    return candidate


def safe_path_component(value: object, *, fallback: str = "class") -> str:
    """Return one deterministic filesystem component without traversal syntax."""
    raw = unicodedata.normalize("NFKC", str(value)).strip()
    cleaned = re.sub(r"[\\/<>:\"|?*\x00-\x1f\x7f]+", "_", raw)
    cleaned = cleaned.strip(" .")
    if cleaned in {"", ".", ".."}:
        cleaned = fallback
    if cleaned.split(".", 1)[0].upper() in _WINDOWS_RESERVED:
        cleaned = f"{cleaned}_"
    changed = cleaned != raw
    if len(cleaned) > 96:
        cleaned = cleaned[:96].rstrip(" ._") or fallback
        changed = True
    if changed:
        digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()[:8]
        cleaned = f"{cleaned}__{digest}"
    return cleaned
