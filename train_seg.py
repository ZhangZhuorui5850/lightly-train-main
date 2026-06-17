from pathlib import Path

import yaml
import lightly_train

# ======================== 改这里 ========================

OUT = "out/seg/aeroscapes"                             # 输出目录
MODEL = "dinov3/vits16-eomt"                           # 模型: dinov3/vits16-eomt / vitb16-eomt / vitl16-eomt
BACKBONE_WEIGHTS = "weights/dinov3_vits16_pretrain_lvd1689m-08c60483.pth"  # backbone权重，None则用默认

DATA_YAML = "datasets/aeroscapes/dataset_semantic/data.yaml"  # 数据集配置

STEPS = 5000
BATCH_SIZE = 4

# =======================================================


def load_data(yaml_path: str) -> dict:
    """从 data.yaml 读取 classes 和 train/val 路径"""
    cfg_path = Path(yaml_path)
    with cfg_path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        raise ValueError(f"data.yaml 格式错误: {yaml_path}")

    base = cfg_path.parent

    # 读类别
    raw = cfg.get("names") or cfg.get("classes") or {}
    if isinstance(raw, list):
        classes = {i: str(n) for i, n in enumerate(raw)}
    elif isinstance(raw, dict):
        classes = {int(k): str(v) for k, v in raw.items()}
    else:
        raise ValueError(f"无法解析类别: {yaml_path}")

    # 读路径（相对于 yaml 所在目录）
    data: dict = {"classes": classes}
    for split in ("train", "val"):
        split_cfg = cfg.get(split)
        if split_cfg and isinstance(split_cfg, dict):
            data[split] = {
                "images": str(base / split_cfg["images"]),
                "masks":  str(base / split_cfg["masks"]),
            }

    return data


def main():
    data = load_data(DATA_YAML)

    # 验证路径
    for split in ("train", "val"):
        for key in ("images", "masks"):
            p = Path(data[split][key])
            if not p.exists():
                raise FileNotFoundError(f"{split}/{key} 不存在: {p}")

    lightly_train.train_semantic_segmentation(
        out=OUT,
        model=MODEL,
        model_args={"backbone_weights": BACKBONE_WEIGHTS} if BACKBONE_WEIGHTS else None,
        data=data,
        overwrite=True,
        steps=STEPS,
        batch_size=BATCH_SIZE,
        num_workers="auto",
    )


if __name__ == "__main__":
    main()
