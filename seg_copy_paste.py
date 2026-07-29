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
        feather=0,                # 边缘羽化核大小(奇数,0=硬贴)，仅对图像做软化、标签仍硬边
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

_CACHE_VERSION = 1
_CONFIG: dict[str, Any] = {
    "enabled": False,
    "prob": 0.5,
    "paste_classes": None,
    "background_classes": (0,),
    "max_paste": 3,
    "min_area_frac": 0.001,
    "source_max_tries": 20,
    "feather": 0,
    "verbose": True,
}


def _distributed_is_initialized() -> bool:
    return torch.distributed.is_available() and torch.distributed.is_initialized()


def _is_global_rank_zero() -> bool:
    """仅让全局主进程输出进度，避免 DDP 多进程进度条互相覆盖。"""
    if _distributed_is_initialized():
        return torch.distributed.get_rank() == 0
    return int(os.environ.get("RANK", "0")) == 0


class CopyPasteDataset(MaskSemanticSegmentationDataset):
    """在父类基础上，仅在训练集上插入在线 copy-paste。"""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        is_train = "train" in type(self.transform).__name__.lower()
        self._cp_on = bool(_CONFIG["enabled"]) and is_train
        bg = set(int(c) for c in _CONFIG["background_classes"])
        if _CONFIG["paste_classes"] is None:
            paste = set(int(c) for c in self.class_id_to_internal_class_id) - bg
        else:
            paste = set(int(c) for c in _CONFIG["paste_classes"])
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
                    index, manifest_digest = self._scan_index()
                    self._save_cached_index(
                        cache_path=cache_path,
                        index=index,
                        manifest_digest=manifest_digest,
                    )
                else:
                    index = cached
                    if _CONFIG["verbose"]:
                        logger.info(f"[copy-paste] 复用索引缓存: '{cache_path}'")
            self._cp_index = index

        if _distributed_is_initialized():
            torch.distributed.barrier()

        if not _is_global_rank_zero():
            cached = self._read_cached_index(cache_path)
            if cached is None:
                raise RuntimeError(f"Copy-Paste 索引缓存读取失败: '{cache_path}'")
            self._cp_index = cached["index"]

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
        stat = mask_path.stat()
        digest.update(os.path.abspath(mask_path).encode("utf-8"))
        digest.update(b"\0")
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

    def _scan_index(self) -> tuple[dict[int, list[int]], str]:
        index = {c: [] for c in self._cp_classes}
        digest = hashlib.sha256()
        indices = track(
            range(len(self.image_info)),
            label="[copy-paste] 扫描训练 mask",
            unit="mask",
            enable=bool(_CONFIG["verbose"]),
        )
        for i in indices:
            mask_path = Path(self.image_info[i]["mask_filepaths"])
            self._update_manifest(digest=digest, mask_path=mask_path)
            mask = file_helpers.open_mask_numpy(mask_path=mask_path)
            mask = self.map_mask_labels_to_class_ids(mask)
            present = set(np.unique(mask).tolist()) & self._cp_classes
            for c in present:
                index[int(c)].append(i)
        return (
            {c: idxs for c, idxs in index.items() if idxs},
            digest.hexdigest(),
        )

    def _read_cached_index(self, cache_path: Path) -> dict[str, Any] | None:
        try:
            with cache_path.open("r", encoding="utf-8") as f:
                raw = json.load(f)
            if raw.get("version") != _CACHE_VERSION:
                return None
            raw_index = raw["index"]
            index = {
                int(class_id): [int(i) for i in indices]
                for class_id, indices in raw_index.items()
            }
            if any(
                class_id not in self._cp_classes
                or any(i < 0 or i >= len(self.image_info) for i in indices)
                for class_id, indices in index.items()
            ):
                return None
            return {
                "manifest_digest": str(raw["manifest_digest"]),
                "index": index,
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
    ) -> dict[int, list[int]] | None:
        cached = self._read_cached_index(cache_path)
        if cached is None:
            return None
        if cached["manifest_digest"] != self._manifest_digest():
            if _CONFIG["verbose"]:
                logger.info("[copy-paste] 数据集已更新，重新构建索引。")
            return None
        return cached["index"]

    @staticmethod
    def _save_cached_index(
        cache_path: Path,
        index: dict[int, list[int]],
        manifest_digest: str,
    ) -> None:
        payload = {
            "version": _CACHE_VERSION,
            "manifest_digest": manifest_digest,
            "index": index,
        }
        temp_path = cache_path.with_suffix(f".{os.getpid()}.tmp")
        try:
            with temp_path.open("w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False, separators=(",", ":"))
            temp_path.replace(cache_path)
        finally:
            temp_path.unlink(missing_ok=True)

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
    ) -> None:
        h, w = tgt_mask.shape[-2:]
        min_area = max(1, int(_CONFIG["min_area_frac"] * h * w))
        src_mask_np = src_mask.cpu().numpy()
        candidates = []
        present = set(np.unique(src_mask_np).tolist()) & self._cp_classes
        for c in present:
            cls_bool = (src_mask_np == c).astype(np.uint8)
            n, labels = cv2.connectedComponents(cls_bool, connectivity=8)
            for comp in range(1, n):
                comp_bool = labels == comp
                if int(comp_bool.sum()) >= min_area:
                    candidates.append((int(c), comp_bool))
        if not candidates:
            return
        random.shuffle(candidates)
        feather = int(_CONFIG["feather"])
        for c, comp_bool in candidates[: _CONFIG["max_paste"]]:
            b = torch.from_numpy(comp_bool).to(tgt_mask.device)
            if feather >= 3 and feather % 2 == 1:
                alpha_np = cv2.GaussianBlur(
                    comp_bool.astype(np.float32), (feather, feather), 0
                )
                a = torch.from_numpy(alpha_np).to(tgt_img.device)
                tgt_img.mul_(1 - a).add_(src_img * a)
            else:
                tgt_img[:, b] = src_img[:, b]
            tgt_mask[b] = c

    def __getitem__(self, index: int) -> MaskSemanticSegmentationDatasetItem:
        if not self._cp_on or not self._cp_index:
            return super().__getitem__(index)
        loaded = self._load_transformed(index)
        assert loaded is not None
        tgt_img, tgt_mask = loaded

        if random.random() < _CONFIG["prob"]:
            c = random.choice(list(self._cp_index.keys()))
            src_index = random.choice(self._cp_index[c])
            src = self._load_transformed(src_index, require_classes={c})
            if src is not None:
                src_img, src_mask = src
                tgt_img = tgt_img.clone()
                tgt_mask = tgt_mask.clone()
                self._paste(tgt_img, tgt_mask, src_img, src_mask)

        binary_masks = self.get_binary_masks(tgt_mask)
        internal_mask = self.map_class_id_to_internal_class_id(tgt_mask)

        return MaskSemanticSegmentationDatasetItem(
            image_path=str(self.image_info[index]["image_filepaths"]),
            image=tgt_img,
            mask=internal_mask,
            binary_masks=binary_masks,
        )


def enable(**kwargs):
    """更新配置并把 copy-paste 子类挂到官方的 get_dataset_cls 上。"""
    _CONFIG.update(kwargs)
    _CONFIG["enabled"] = True
    _msd.MaskSemanticSegmentationDatasetArgs.get_dataset_cls = staticmethod(
        lambda: CopyPasteDataset
    )
    logger.info("[copy-paste] 已注册 CopyPasteDataset（仅训练集生效）。")
