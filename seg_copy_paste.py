"""
在线 Copy-Paste 数据增强（语义分割版）—— 方案 A：零改动官方文件。

用法（在 train_seg.py 顶部、调用 train 之前）：

    import seg_copy_paste
    seg_copy_paste.enable(
        prob=0.5,                 # 多大概率对一张图做拼贴
        paste_classes=None,       # 允许被粘贴的类别 id 列表；None=除背景外全部
        background_classes=(0,),  # 视为背景/多数类、不参与粘贴的类别 id
        max_paste=3,              # 每张图最多粘贴几个连通块（实例）
        min_area_frac=0.001,      # 连通块面积小于全图该比例则丢弃（过滤噪点碎片）
        source_max_tries=20,      # 选源后裁剪可能裁掉目标类，最多重试多少次
        source_sample_tries=3,    # 源样本无有效连通块/位置时最多换几个源样本
        feather=0,                # 边缘羽化核大小(奇数,0=硬贴)，仅对图像做软化、标签仍硬边
        max_target_overlap=0.10,  # 粘贴区域最多覆盖多少已有前景
        verbose=True,
    )

原理：在 DataLoader 取样本时（dataset.__getitem__），随机挑一个"稀有类"，
从含该类的源图里把对应连通块抠出来，硬贴到当前图上，并同步覆盖 mask。
- 只在 **训练** 数据集生效（按 transform 类名判断），验证集不动。
- 官方文件一行不改：通过子类 + monkeypatch get_dataset_cls 注入。
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import random
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from filelock import FileLock

from lightly_train._data import cache, file_helpers
from lightly_train._data import mask_semantic_segmentation_dataset as _msd
from lightly_train._data.mask_semantic_segmentation_dataset import (
    MaskSemanticSegmentationDataset,
)
from lightly_train.types import MaskSemanticSegmentationDatasetItem
from tool_lib.progress import track

logger = logging.getLogger(__name__)

_CACHE_VERSION = 2
_MAX_BAD_MASK_EXAMPLES = 20
_DEFAULT_CONFIG: dict[str, Any] = {
    "enabled": False,
    "prob": 0.5,
    "paste_classes": None,
    "background_classes": (0,),
    "max_paste": 3,
    "min_area_frac": 0.001,
    "source_max_tries": 20,
    "source_sample_tries": 3,
    "feather": 0,
    "max_target_overlap": 0.10,
    "verbose": True,
    # 每处理多少张 mask 保存一次断点。扫描被中断后可从最近断点继续。
    "index_checkpoint_interval": 10_000,
}
_CONFIG: dict[str, Any] = dict(_DEFAULT_CONFIG)
_ORIGINAL_GET_DATASET_CLS = _msd.MaskSemanticSegmentationDatasetArgs.get_dataset_cls


def _distributed_is_initialized() -> bool:
    return torch.distributed.is_available() and torch.distributed.is_initialized()


def _is_global_rank_zero() -> bool:
    """仅让全局主进程输出进度，避免 DDP 多进程进度条互相覆盖。"""
    if _distributed_is_initialized():
        return torch.distributed.get_rank() == 0
    return int(os.environ.get("RANK", "0")) == 0


def _choose_source_index(
    indices: list[int],
    target_index: int,
    excluded: set[int] | None = None,
) -> int | None:
    """Choose a source sample while keeping target and source distinct."""
    blocked = {target_index, *(excluded or ())}
    candidates = [index for index in indices if index not in blocked]
    return random.choice(candidates) if candidates else None


class CopyPasteDataset(MaskSemanticSegmentationDataset):
    """在父类基础上，仅在训练集上插入在线 copy-paste。"""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self._cp_on = False
        self._cp_valid_indices: list[int] | None = None
        self._cp_bad_indices: list[int] = []
        self._cp_bad_examples: list[dict[str, Any]] = []
        super().__init__(*args, **kwargs)
        is_train = "train" in type(self.transform).__name__.lower()
        self._cp_on = bool(_CONFIG["enabled"]) and is_train
        bg = set(int(c) for c in _CONFIG["background_classes"])
        if _CONFIG["paste_classes"] is None:
            paste = set(int(c) for c in self.class_id_to_internal_class_id) - bg
        else:
            paste = set(int(c) for c in _CONFIG["paste_classes"])
        unknown_classes = paste - set(self.class_id_to_internal_class_id)
        if unknown_classes:
            raise ValueError(
                f"paste_classes 包含数据集未定义的类别: {sorted(unknown_classes)}"
            )
        self._cp_classes = paste

        self._cp_index = {}
        if self._cp_on:
            self._build_index()

    def _build_index(self) -> None:
        cache_path = self._index_cache_path()
        if _is_global_rank_zero():
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            with FileLock(str(cache_path) + ".lock"):
                cached = self._load_valid_cached_index(cache_path)
                if cached is None:
                    (
                        index,
                        manifest_digest,
                        valid_indices,
                        bad_indices,
                        bad_examples,
                    ) = self._scan_index(cache_path=cache_path)
                    self._save_cached_index(
                        cache_path=cache_path,
                        index=index,
                        manifest_digest=manifest_digest,
                        valid_indices=valid_indices,
                        bad_indices=bad_indices,
                        bad_examples=bad_examples,
                    )
                    cache_path.with_suffix(".partial.json").unlink(missing_ok=True)
                else:
                    index = cached["index"]
                    valid_indices = cached["valid_indices"]
                    bad_indices = cached["bad_indices"]
                    bad_examples = cached["bad_examples"]
                    if _CONFIG["verbose"]:
                        logger.info(f"[copy-paste] 复用索引缓存: '{cache_path}'")
            self._cp_index = index
            self._cp_valid_indices = valid_indices
            self._cp_bad_indices = bad_indices
            self._cp_bad_examples = bad_examples

        if _distributed_is_initialized():
            torch.distributed.barrier()

        if not _is_global_rank_zero():
            cached = self._read_cached_index(cache_path)
            if cached is None:
                raise RuntimeError(f"Copy-Paste 索引缓存读取失败: '{cache_path}'")
            self._cp_index = cached["index"]
            self._cp_valid_indices = cached["valid_indices"]
            self._cp_bad_indices = cached["bad_indices"]
            self._cp_bad_examples = cached["bad_examples"]

        if _CONFIG["verbose"] and _is_global_rank_zero():
            stat = {c: len(idxs) for c, idxs in sorted(self._cp_index.items())}
            logger.info(
                f"[copy-paste] 启用。可粘贴类别->图片数: {stat} "
                f"(prob={_CONFIG['prob']}, max_paste={_CONFIG['max_paste']})"
            )
            if not self._cp_index:
                logger.warning(
                    "[copy-paste] 训练集中没有任何可粘贴的类别，copy-paste 将不生效。"
                )
            if self._cp_bad_indices:
                logger.warning(
                    f"[copy-paste] 已隔离 {len(self._cp_bad_indices)} 张无法读取的训练 mask，"
                    f"有效样本数: {len(self._cp_valid_indices or [])}。"
                )
                for item in self._cp_bad_examples:
                    logger.warning(
                        "[copy-paste] 异常 mask: "
                        f"index={item['index']}, path='{item['path']}', "
                        f"error={item['error']}"
                    )

    def _index_cache_path(self) -> Path:
        classes = []
        for class_id, class_info in sorted(self.dataset_args.classes.items()):
            labels = sorted(repr(label) for label in class_info.labels)
            classes.append((int(class_id), labels))
        identity = {
            "version": _CACHE_VERSION,
            "image_dir": str(self.dataset_args.image_dir.expanduser().resolve()),
            "mask_source": str(
                Path(self.dataset_args.mask_dir_or_file).expanduser().resolve()
            ),
            "classes": classes,
            "ignore_classes": sorted(self.dataset_args.ignore_classes or ()),
            "ignore_index": int(self.dataset_args.ignore_index),
            "paste_classes": sorted(self._cp_classes),
        }
        digest = hashlib.sha256(
            json.dumps(identity, sort_keys=True).encode("utf-8")
        ).hexdigest()
        return cache.get_data_cache_dir() / "copy_paste" / f"{digest}.json"

    @staticmethod
    def _update_manifest(
        digest: Any,
        mask_path: Path,
    ) -> None:
        digest.update(os.path.abspath(mask_path).encode("utf-8"))
        digest.update(b"\0")
        try:
            stat = mask_path.stat()
        except OSError as exc:
            # 把稳定的失败信息纳入清单。文件恢复可读后摘要会变化，
            # 因而会自动触发索引重建。
            digest.update(b"STAT_ERROR\0")
            digest.update(type(exc).__name__.encode("utf-8"))
            digest.update(b"\0")
            digest.update(str(getattr(exc, "errno", "")).encode("ascii"))
        else:
            digest.update(str(stat.st_size).encode("ascii"))
            digest.update(b"\0")
            digest.update(str(stat.st_mtime_ns).encode("ascii"))
        digest.update(b"\n")

    def _manifest_digest(self) -> str:
        digest = hashlib.sha256()
        indices = track(
            range(len(self.image_info)),
            label="[copy-paste] 校验索引缓存",
            unit="mask",
            enable=bool(_CONFIG["verbose"]),
        )
        for i in indices:
            mask_path = Path(self.image_info[i]["mask_filepaths"])
            self._update_manifest(digest=digest, mask_path=mask_path)
        return digest.hexdigest()

    def _scan_index(
        self,
        cache_path: Path,
    ) -> tuple[
        dict[int, list[int]],
        str,
        list[int],
        list[int],
        list[dict[str, Any]],
    ]:
        manifest_digest = self._manifest_digest()
        partial_path = cache_path.with_suffix(".partial.json")
        partial = self._read_partial_index(
            partial_path=partial_path,
            manifest_digest=manifest_digest,
        )
        if partial is None:
            index = {c: [] for c in self._cp_classes}
            valid_indices: list[int] = []
            bad_indices: list[int] = []
            bad_examples: list[dict[str, Any]] = []
            start = 0
        else:
            index = partial["index"]
            valid_indices = partial["valid_indices"]
            bad_indices = partial["bad_indices"]
            bad_examples = partial["bad_examples"]
            start = partial["next_index"]
            if _CONFIG["verbose"]:
                logger.info(
                    f"[copy-paste] 从断点继续扫描: {start}/{len(self.image_info)}"
                )

        indices = track(
            range(start, len(self.image_info)),
            label="[copy-paste] 扫描训练 mask",
            unit="mask",
            enable=bool(_CONFIG["verbose"]),
        )
        checkpoint_interval = max(1, int(_CONFIG["index_checkpoint_interval"]))
        for i in indices:
            mask_path = Path(self.image_info[i]["mask_filepaths"])
            try:
                # stat 在这里提前检查路径与权限，便于记录精确原因。
                mask_path.stat()
                mask = file_helpers.open_mask_numpy(mask_path=mask_path)
                present = self._present_copy_paste_classes(mask)
            except MemoryError as exc:
                raise RuntimeError(
                    f"Copy-Paste 扫描 mask 时内存不足: "
                    f"index={i}, path='{mask_path}'"
                ) from exc
            except Exception as exc:
                bad_indices.append(i)
                if len(bad_examples) < _MAX_BAD_MASK_EXAMPLES:
                    bad_examples.append(
                        {
                            "index": i,
                            "path": str(mask_path),
                            "error": f"{type(exc).__name__}: {exc}",
                        }
                    )
                if _CONFIG["verbose"] and (
                    len(bad_indices) <= _MAX_BAD_MASK_EXAMPLES
                    or len(bad_indices) % 1000 == 0
                ):
                    logger.warning(
                        "[copy-paste] 隔离异常 mask: "
                        f"index={i}, path='{mask_path}', "
                        f"error={type(exc).__name__}: {exc}"
                    )
            else:
                valid_indices.append(i)
                for c in present:
                    index[int(c)].append(i)

            next_index = i + 1
            if next_index % checkpoint_interval == 0:
                self._save_partial_index(
                    partial_path=partial_path,
                    manifest_digest=manifest_digest,
                    next_index=next_index,
                    index=index,
                    valid_indices=valid_indices,
                    bad_indices=bad_indices,
                    bad_examples=bad_examples,
                )

        self._save_partial_index(
            partial_path=partial_path,
            manifest_digest=manifest_digest,
            next_index=len(self.image_info),
            index=index,
            valid_indices=valid_indices,
            bad_indices=bad_indices,
            bad_examples=bad_examples,
        )
        return (
            {c: idxs for c, idxs in index.items() if idxs},
            manifest_digest,
            valid_indices,
            bad_indices,
            bad_examples,
        )

    def _present_copy_paste_classes(self, mask: np.ndarray) -> set[int]:
        """Return paste classes present in a mask with a low-memory fast path."""
        if mask.ndim == 2 or (mask.ndim == 3 and mask.shape[2] == 1):
            raw_labels = set(np.unique(mask).tolist())
            present: set[int] = set()
            for class_id in self._cp_classes:
                class_info = self.dataset_args.classes.get(class_id)
                if class_info is None:
                    continue
                if any(
                    isinstance(label, (int, np.integer))
                    and int(label) in raw_labels
                    for label in class_info.labels
                ):
                    present.add(class_id)
            return present

        mapped = self.map_mask_labels_to_class_ids(mask)
        return set(np.unique(mapped).tolist()) & self._cp_classes

    def _read_partial_index(
        self,
        partial_path: Path,
        manifest_digest: str,
    ) -> dict[str, Any] | None:
        try:
            with partial_path.open("r", encoding="utf-8") as f:
                raw = json.load(f)
            if raw.get("version") != _CACHE_VERSION:
                return None
            if raw.get("manifest_digest") != manifest_digest:
                return None
            next_index = int(raw["next_index"])
            if next_index < 0 or next_index > len(self.image_info):
                return None
            index = self._parse_index(raw["index"], upper_bound=next_index)
            valid_indices = [int(i) for i in raw["valid_indices"]]
            bad_indices = [int(i) for i in raw["bad_indices"]]
            if not self._is_valid_partition(
                valid_indices=valid_indices,
                bad_indices=bad_indices,
                expected_size=next_index,
            ):
                return None
            return {
                "next_index": next_index,
                "index": index,
                "valid_indices": valid_indices,
                "bad_indices": bad_indices,
                "bad_examples": self._parse_bad_examples(raw.get("bad_examples", [])),
            }
        except (
            AttributeError,
            FileNotFoundError,
            KeyError,
            TypeError,
            ValueError,
            json.JSONDecodeError,
        ):
            return None

    @staticmethod
    def _save_partial_index(
        partial_path: Path,
        manifest_digest: str,
        next_index: int,
        index: dict[int, list[int]],
        valid_indices: list[int],
        bad_indices: list[int],
        bad_examples: list[dict[str, Any]],
    ) -> None:
        payload = {
            "version": _CACHE_VERSION,
            "manifest_digest": manifest_digest,
            "next_index": next_index,
            "index": index,
            "valid_indices": valid_indices,
            "bad_indices": bad_indices,
            "bad_examples": bad_examples,
        }
        CopyPasteDataset._atomic_write_json(path=partial_path, payload=payload)

    def _parse_index(
        self,
        raw_index: Any,
        upper_bound: int,
    ) -> dict[int, list[int]]:
        index = {
            int(class_id): [int(i) for i in indices]
            for class_id, indices in raw_index.items()
        }
        if any(
            class_id not in self._cp_classes
            or any(i < 0 or i >= upper_bound for i in indices)
            for class_id, indices in index.items()
        ):
            raise ValueError("Invalid Copy-Paste index")
        return index

    @staticmethod
    def _parse_bad_examples(raw_examples: Any) -> list[dict[str, Any]]:
        examples: list[dict[str, Any]] = []
        for raw in list(raw_examples)[:_MAX_BAD_MASK_EXAMPLES]:
            examples.append(
                {
                    "index": int(raw["index"]),
                    "path": str(raw["path"]),
                    "error": str(raw["error"]),
                }
            )
        return examples

    @staticmethod
    def _is_valid_partition(
        valid_indices: list[int],
        bad_indices: list[int],
        expected_size: int,
    ) -> bool:
        if len(valid_indices) + len(bad_indices) != expected_size:
            return False
        seen = bytearray(expected_size)
        for indices in (valid_indices, bad_indices):
            for index in indices:
                if index < 0 or index >= expected_size or seen[index]:
                    return False
                seen[index] = 1
        return True

    def _read_cached_index(self, cache_path: Path) -> dict[str, Any] | None:
        try:
            with cache_path.open("r", encoding="utf-8") as f:
                raw = json.load(f)
            if raw.get("version") != _CACHE_VERSION:
                return None
            index = self._parse_index(
                raw_index=raw["index"],
                upper_bound=len(self.image_info),
            )
            valid_indices = [int(i) for i in raw["valid_indices"]]
            bad_indices = [int(i) for i in raw["bad_indices"]]
            if not self._is_valid_partition(
                valid_indices=valid_indices,
                bad_indices=bad_indices,
                expected_size=len(self.image_info),
            ):
                return None
            return {
                "manifest_digest": str(raw["manifest_digest"]),
                "index": index,
                "valid_indices": valid_indices,
                "bad_indices": bad_indices,
                "bad_examples": self._parse_bad_examples(raw.get("bad_examples", [])),
            }
        except (
            AttributeError,
            FileNotFoundError,
            KeyError,
            TypeError,
            ValueError,
            json.JSONDecodeError,
        ):
            return None

    def _load_valid_cached_index(
        self,
        cache_path: Path,
    ) -> dict[str, Any] | None:
        cached = self._read_cached_index(cache_path)
        if cached is None:
            return None
        if cached["manifest_digest"] != self._manifest_digest():
            if _CONFIG["verbose"]:
                logger.info("[copy-paste] 数据集已更新，重新构建索引。")
            return None
        return cached

    @staticmethod
    def _save_cached_index(
        cache_path: Path,
        index: dict[int, list[int]],
        manifest_digest: str,
        valid_indices: list[int],
        bad_indices: list[int],
        bad_examples: list[dict[str, Any]],
    ) -> None:
        payload = {
            "version": _CACHE_VERSION,
            "manifest_digest": manifest_digest,
            "index": index,
            "valid_indices": valid_indices,
            "bad_indices": bad_indices,
            "bad_examples": bad_examples,
        }
        CopyPasteDataset._atomic_write_json(path=cache_path, payload=payload)

    @staticmethod
    def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
        temp_path = path.with_suffix(f".{os.getpid()}.tmp")
        try:
            with temp_path.open("w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False, separators=(",", ":"))
            temp_path.replace(path)
        finally:
            temp_path.unlink(missing_ok=True)

    def __len__(self) -> int:
        if self._cp_on and self._cp_valid_indices is not None:
            return len(self._cp_valid_indices)
        return super().__len__()

    def _load_transformed(
        self, index: int, require_classes: set[int] | None = None
    ) -> tuple[torch.Tensor, torch.Tensor] | None:
        row = self.image_info[index]
        image = file_helpers.open_image_numpy(
            image_path=Path(row["image_filepaths"]), mode=self.image_mode
        )
        mask = file_helpers.open_mask_numpy(mask_path=Path(row["mask_filepaths"]))
        if image.shape[:2] != mask.shape[:2]:
            raise ValueError(
                f"Shape mismatch: image {image.shape[:2]} vs mask {mask.shape[:2]}."
            )
        mask = self.map_mask_labels_to_class_ids(mask)
        tries = _CONFIG["source_max_tries"] if require_classes else 20
        transformed = None
        for _ in range(tries):
            transformed = self.transform({"image": image, "mask": mask})
            t_mask = transformed["mask"]
            if require_classes is not None:
                present = set(torch.unique(t_mask).tolist())
                if present & require_classes:
                    return transformed["image"], t_mask
            else:
                if self.is_mask_valid(t_mask):
                    return transformed["image"], t_mask
        if require_classes is not None:
            return None
        assert transformed is not None
        return transformed["image"], transformed["mask"]

    def _paste(
        self,
        tgt_img: torch.Tensor,
        tgt_mask: torch.Tensor,
        src_img: torch.Tensor,
        src_mask: torch.Tensor,
        selected_class: int,
    ) -> bool:
        h, w = tgt_mask.shape[-2:]
        min_area = max(1, int(_CONFIG["min_area_frac"] * h * w))
        src_mask_np = src_mask.cpu().numpy()
        cls_bool = (src_mask_np == selected_class).astype(np.uint8)
        n, labels = cv2.connectedComponents(cls_bool, connectivity=8)
        candidates = []
        for comp in range(1, n):
            comp_bool = labels == comp
            if int(comp_bool.sum()) >= min_area:
                candidates.append(comp_bool)
        if not candidates:
            return False
        random.shuffle(candidates)
        feather = int(_CONFIG["feather"])
        background_classes = tuple(int(c) for c in _CONFIG["background_classes"])
        max_overlap = float(_CONFIG["max_target_overlap"])
        pasted = False
        for comp_bool in candidates[: int(_CONFIG["max_paste"])]:
            x, y, comp_w, comp_h = cv2.boundingRect(comp_bool.astype(np.uint8))
            if comp_w > w or comp_h > h:
                continue
            local_mask_np = comp_bool[y : y + comp_h, x : x + comp_w]
            local_mask = torch.from_numpy(local_mask_np).to(tgt_mask.device)
            placement: tuple[int, int] | None = None
            for _ in range(30):
                left = random.randint(0, w - comp_w)
                top = random.randint(0, h - comp_h)
                target_view = tgt_mask[top : top + comp_h, left : left + comp_w]
                target_background = torch.zeros_like(target_view, dtype=torch.bool)
                for background_class in background_classes:
                    target_background |= target_view == background_class
                overlap = int((local_mask & ~target_background).sum().item())
                if overlap / int(local_mask.sum().item()) <= max_overlap:
                    placement = (left, top)
                    break
            if placement is None:
                continue
            left, top = placement
            target_image_view = tgt_img[:, top : top + comp_h, left : left + comp_w]
            source_image_view = src_img[:, y : y + comp_h, x : x + comp_w]
            if feather >= 3 and feather % 2 == 1:
                alpha_np = cv2.GaussianBlur(
                    local_mask_np.astype(np.float32), (feather, feather), 0
                )
                alpha = torch.from_numpy(alpha_np).to(
                    device=tgt_img.device, dtype=tgt_img.dtype
                )
                target_image_view.mul_(1 - alpha).add_(source_image_view * alpha)
            else:
                target_image_view[:, local_mask] = source_image_view[:, local_mask]
            target_mask_view = tgt_mask[top : top + comp_h, left : left + comp_w]
            target_mask_view[local_mask] = selected_class
            pasted = True
        return pasted

    def __getitem__(self, index: int) -> MaskSemanticSegmentationDatasetItem:
        if self._cp_on and self._cp_valid_indices is not None:
            index = self._cp_valid_indices[index]
        if not self._cp_on or not self._cp_index:
            return super().__getitem__(index)
        loaded = self._load_transformed(index)
        assert loaded is not None
        tgt_img, tgt_mask = loaded

        if random.random() < _CONFIG["prob"]:
            c = random.choice(list(self._cp_index.keys()))
            used_sources: set[int] = set()
            for _ in range(int(_CONFIG["source_sample_tries"])):
                src_index = _choose_source_index(
                    self._cp_index[c], index, excluded=used_sources
                )
                if src_index is None:
                    break
                used_sources.add(src_index)
                src = self._load_transformed(src_index, require_classes={c})
                if src is None:
                    continue
                src_img, src_mask = src
                pasted_img = tgt_img.clone()
                pasted_mask = tgt_mask.clone()
                if self._paste(
                    pasted_img,
                    pasted_mask,
                    src_img,
                    src_mask,
                    selected_class=c,
                ):
                    tgt_img, tgt_mask = pasted_img, pasted_mask
                    break

        binary_masks = self.get_binary_masks(tgt_mask)
        internal_mask = self.map_class_id_to_internal_class_id(tgt_mask)

        return MaskSemanticSegmentationDatasetItem(
            image_path=str(self.image_info[index]["image_filepaths"]),
            image=tgt_img,
            mask=internal_mask,
            binary_masks=binary_masks,
        )


def _validated_config(kwargs: dict[str, Any]) -> dict[str, Any]:
    unknown = sorted(set(kwargs) - (set(_DEFAULT_CONFIG) - {"enabled"}))
    if unknown:
        raise TypeError(f"未知 Copy-Paste 参数: {', '.join(unknown)}")
    config = dict(_DEFAULT_CONFIG)
    config.update(kwargs)

    for name in ("prob", "min_area_frac", "max_target_overlap"):
        if isinstance(config[name], bool):
            raise ValueError(f"{name} 需要为数值")
        try:
            config[name] = float(config[name])
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{name} 需要为数值") from exc
        if not 0 <= config[name] <= 1:
            raise ValueError(f"{name} 需要位于 [0, 1]")

    for name in (
        "max_paste",
        "source_max_tries",
        "source_sample_tries",
        "index_checkpoint_interval",
    ):
        value = config[name]
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"{name} 需要为正整数")

    feather = config["feather"]
    if (
        isinstance(feather, bool)
        or not isinstance(feather, int)
        or (feather != 0 and (feather < 3 or feather % 2 == 0))
    ):
        raise ValueError("feather 需要为 0 或不小于 3 的奇数")
    if not isinstance(config["verbose"], bool):
        raise ValueError("verbose 需要为布尔值")

    for name, allow_none in (("paste_classes", True), ("background_classes", False)):
        value = config[name]
        if value is None and allow_none:
            continue
        if isinstance(value, (str, bytes)):
            raise ValueError(f"{name} 需要为类别 ID 序列")
        try:
            class_ids = []
            for class_id in value:
                if isinstance(class_id, bool) or not isinstance(
                    class_id, (int, np.integer)
                ):
                    raise ValueError
                class_ids.append(int(class_id))
            config[name] = tuple(dict.fromkeys(class_ids))
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{name} 需要为类别 ID 序列") from exc
        if name == "background_classes" and not config[name]:
            raise ValueError("background_classes 至少需要一个类别 ID")
    if config["paste_classes"] is not None and set(config["paste_classes"]) & set(
        config["background_classes"]
    ):
        raise ValueError("paste_classes 与 background_classes 需要保持分离")

    config["enabled"] = True
    return config


def enable(**kwargs: Any) -> None:
    """校验配置并把 copy-paste 子类挂到官方的 get_dataset_cls 上。"""
    config = _validated_config(kwargs)
    _CONFIG.clear()
    _CONFIG.update(config)
    _msd.MaskSemanticSegmentationDatasetArgs.get_dataset_cls = staticmethod(
        lambda: CopyPasteDataset
    )
    logger.info("[copy-paste] 已注册 CopyPasteDataset（仅训练集生效）。")


def disable() -> None:
    """恢复官方数据集类并重置 Copy-Paste 配置。"""
    _CONFIG.clear()
    _CONFIG.update(_DEFAULT_CONFIG)
    _msd.MaskSemanticSegmentationDatasetArgs.get_dataset_cls = staticmethod(
        _ORIGINAL_GET_DATASET_CLS
    )
