from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import yaml
import lightly_train

import seg_copy_paste  # 在线 Copy-Paste 增强（必须在调用 train 之前 import）
from tool_lib.training_run import inspect_run_mode, prepare_run

# ======================== 改这里 ========================

OUT = "out/seg/aeroscapes"                             # 输出目录
MODEL = "dinov3/vits16-eomt"                           # 模型: dinov3/vits16-eomt / vitb16-eomt / vitl16-eomt
BACKBONE_WEIGHTS = "weights/dinov3_vits16_pretrain_lvd1689m-08c60483.pth"  # backbone权重，None则用默认

DATA_YAML = "datasets/aeroscapes/dataset_semantic/data.yaml"  # 数据集配置

STEPS = 2000
BATCH_SIZE = 4
FRESH = False              # True=覆盖同名目录重新训练；False=自动新建或续训

# ---- Copy-Paste 在线数据增强 ----
COPY_PASTE = True          # False=完全关闭，行为回到官方原样
COPY_PASTE_ARGS = dict(
    prob=0.5,              # 多大概率对一张图做拼贴
    paste_classes=(1, 2, 3, 4, 5, 6, 7),  # 当前数据中像素占比低于1%的类别
    background_classes=(0,),  # 背景/多数类，不参与粘贴
    max_paste=3,           # 每张图最多粘贴几个连通块
    min_area_frac=0.0005,  # 512输入约131像素，保留更多稀有小区域
    source_max_tries=20,   # 选源裁剪可能裁掉目标类，最多重试次数
    source_sample_tries=3, # 连通块/位置无效时最多更换3个源样本
    feather=0,             # 边缘羽化核(奇数,0=硬贴)，仅软化图像
    max_target_overlap=0.10,  # 粘贴区域最多覆盖10%的已有前景
    verbose=True,
)

# =======================================================


def load_data(yaml_path: str) -> dict:
    """从 data.yaml 读取 classes 和 train/val 路径"""
    cfg_path = Path(yaml_path)
    if not cfg_path.is_absolute():
        cfg_path = PROJECT_ROOT / cfg_path
    cfg_path = cfg_path.resolve()
    with cfg_path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        raise ValueError(f"data.yaml 格式错误: {yaml_path}")

    configured_root = Path(cfg.get("path", "."))
    base = (
        configured_root
        if configured_root.is_absolute()
        else cfg_path.parent / configured_root
    ).resolve()

    # 读类别
    raw = cfg.get("names") or cfg.get("classes") or {}
    if isinstance(raw, list):
        classes = {i: str(n) for i, n in enumerate(raw)}
    elif isinstance(raw, dict):
        classes = {int(k): v for k, v in raw.items()}
    else:
        raise ValueError(f"无法解析类别: {yaml_path}")

    # 读路径（相对于 yaml 所在目录）
    data: dict = {"classes": classes}
    for split in ("train", "val"):
        split_cfg = cfg.get(split)
        if not isinstance(split_cfg, dict):
            raise ValueError(f"{split} 需要包含 images 和 masks: {cfg_path}")
        try:
            images = Path(split_cfg["images"])
            masks = Path(split_cfg["masks"])
        except KeyError as exc:
            raise ValueError(f"{split} 缺少字段 {exc.args[0]}: {cfg_path}") from exc
        data[split] = {
            "images": str(images if images.is_absolute() else base / images),
            "masks": str(masks if masks.is_absolute() else base / masks),
        }

    return data


def resolve_run_mode(out: str, fresh: bool) -> tuple[bool, bool]:
    output = Path(out)
    if not output.is_absolute():
        output = PROJECT_ROOT / output
    return inspect_run_mode(output, fresh)


def enable_checkpoint_compatibility() -> None:
    """Register the legacy collate class name used by older checkpoints."""
    from lightly_train._data import task_batch_collation
    try:
        from lightly_train._transforms.semantic_segmentation_transform import (
            SemanticSegmentationCollateFunction,
        )
    except ImportError:
        return

    task_batch_collation.MaskSemanticSegmentationCollateFunction = (  # type: ignore[attr-defined]
        SemanticSegmentationCollateFunction
    )


def main():
    if STEPS <= 0 or BATCH_SIZE <= 0:
        raise ValueError("STEPS 和 BATCH_SIZE 需要为正整数")
    data_yaml_path = Path(DATA_YAML).expanduser()
    if not data_yaml_path.is_absolute():
        data_yaml_path = PROJECT_ROOT / data_yaml_path
    data_yaml_path = data_yaml_path.resolve()
    data = load_data(str(data_yaml_path))

    if COPY_PASTE:
        seg_copy_paste.enable(**COPY_PASTE_ARGS)
    else:
        seg_copy_paste.disable()

    output = Path(OUT)
    if not output.is_absolute():
        output = PROJECT_ROOT / output
    backbone_weights = Path(BACKBONE_WEIGHTS) if BACKBONE_WEIGHTS else None
    if backbone_weights is not None and not backbone_weights.is_absolute():
        backbone_weights = PROJECT_ROOT / backbone_weights
    if backbone_weights is not None and not backbone_weights.is_file():
        raise FileNotFoundError(f"backbone 权重不存在: {backbone_weights}")
    # 验证路径
    for split in ("train", "val"):
        for key in ("images", "masks"):
            p = Path(data[split][key])
            if not p.exists():
                raise FileNotFoundError(f"{split}/{key} 不存在: {p}")

    resume_interrupted, overwrite = prepare_run(
        output,
        fresh=FRESH,
        config={
            "task": "semantic_segmentation",
            "model": MODEL,
            "backbone_weights": str(backbone_weights) if backbone_weights else None,
            "data_yaml": str(data_yaml_path),
            "data": data,
            "steps": STEPS,
            "batch_size": BATCH_SIZE,
            "copy_paste": COPY_PASTE,
            "copy_paste_args": COPY_PASTE_ARGS if COPY_PASTE else None,
        },
    )
    if resume_interrupted:
        enable_checkpoint_compatibility()

    lightly_train.train_semantic_segmentation(
        out=str(output),
        model=MODEL,
        model_args={"backbone_weights": str(backbone_weights)} if backbone_weights else None,
        data=data,
        overwrite=overwrite,
        resume_interrupted=resume_interrupted,
        steps=STEPS,
        batch_size=BATCH_SIZE,
        num_workers="auto",
    )


if __name__ == "__main__":
    main()
