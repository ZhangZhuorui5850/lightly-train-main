from typing import Any

import lightly_train

# ---------------------------------------------------------------------------
# train_det.py 的副本，用于「分层解冻」对比实验。原始 train_det.py 不动。
#
# 对比实验开关：解冻 backbone (ViT) 最后多少层。
#   None / 0 -> 整个 backbone 全冻结（库默认 backbone_freeze 行为）
#   4 / 6    -> 只放开最后 4 / 6 个 transformer block
# vits16 的 ViT 共 12 层。STA 适配层(sta/convs/norms)始终可训练。
# ---------------------------------------------------------------------------
TRAINABLE_BLOCKS = 4  # 跑 6 层实验时改成 6


def main():
    model_args: dict[str, Any] = {
        "backbone_weights": "weights/dinov3_vits16_pretrain_lvd1689m-08c60483.pth",
    }

    # 只有需要分层解冻时才打补丁并开启 backbone_freeze。
    if TRAINABLE_BLOCKS:
        _patch_partial_freeze(TRAINABLE_BLOCKS)
        # backbone_freeze=True 时，set_train_mode 才会调用(被替换的) freeze_backbone。
        model_args["backbone_freeze"] = True

    lightly_train.train_object_detection(
        out=f"out/0414/NEU_train_unfreeze{TRAINABLE_BLOCKS or 'full'}",
        model="dinov3/vits16-ltdetr",
        model_args=model_args,
        data="datasets/neu_dataset/dataset_det/data.yaml",
        overwrite=True,
        steps=200,
        batch_size=4,
        num_workers="auto",
    )


# ---------------------------------------------------------------------------
def _patch_partial_freeze(trainable_blocks: int) -> None:
    """运行时替换 freeze_backbone，只放开最后 N 个 ViT block。不修改库源码。"""
    from lightly_train._task_models.dinov3_ltdetr_object_detection.task_model import (
        DINOv3LTDETRObjectDetection,
    )

    def freeze_backbone(self) -> None:  # 替换原方法，签名一致(只有 self)
        vit = self.backbone.dinov3
        num_blocks = len(vit.blocks)
        if trainable_blocks > num_blocks:
            raise ValueError(
                f"TRAINABLE_BLOCKS ({trainable_blocks}) 超过 backbone 总层数 "
                f"({num_blocks})。"
            )
        first_trainable = num_blocks - trainable_blocks

        # 先冻结整个 ViT，再放开最后 N 个 block。
        vit.eval()
        vit.requires_grad_(False)
        for block in vit.blocks[first_trainable:]:
            block.train()
            block.requires_grad_(True)

        trainable = sum(p.numel() for p in vit.parameters() if p.requires_grad)
        total = sum(p.numel() for p in vit.parameters())
        print(
            f"[partial-freeze] ViT 共 {num_blocks} 层，放开最后 {trainable_blocks} 层 "
            f"(block {first_trainable}..{num_blocks - 1})；"
            f"ViT 可训练参数 {trainable:,}/{total:,}"
        )

    DINOv3LTDETRObjectDetection.freeze_backbone = freeze_backbone


if __name__ == "__main__":
    main()
