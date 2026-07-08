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

import logging
import random
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch

from lightly_train._data import file_helpers
from lightly_train._data import mask_semantic_segmentation_dataset as _msd
from lightly_train._data.mask_semantic_segmentation_dataset import (
    MaskSemanticSegmentationDataset,
)
from lightly_train.types import MaskSemanticSegmentationDatasetItem

logger = logging.getLogger(__name__)

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
        index = {c: [] for c in self._cp_classes}
        for i in range(len(self.image_info)):
            mask_path = Path(self.image_info[i]["mask_filepaths"])
            mask = file_helpers.open_mask_numpy(mask_path=mask_path)
            mask = self.map_mask_labels_to_class_ids(mask)
            present = set(np.unique(mask).tolist()) & self._cp_classes
            for c in present:
                index[int(c)].append(i)
        self._cp_index = {c: idxs for c, idxs in index.items() if idxs}
        if _CONFIG["verbose"]:
            stat = {c: len(idxs) for c, idxs in sorted(self._cp_index.items())}
            logger.info(
                f"[copy-paste] 启用。可粘贴类别->图片数: {stat} "
                f"(prob={_CONFIG['prob']}, max_paste={_CONFIG['max_paste']})"
            )
            if not self._cp_index:
                logger.warning(
                    "[copy-paste] 训练集中没有任何可粘贴的类别，copy-paste 将不生效。"
                )

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
