from __future__ import annotations

import math
import re
import sys
from pathlib import Path

from tool_lib import common as rt
from tool_lib.dispatch import dispatch
from tool_lib.interactive import (
    InputCancelled,
    build_interactive_args,
    cleanup_pending_optimize_temp_dirs,
    parse_cli_args,
    validate_args,
)

# ============================================================
# 统一配置区
# 常改的路径和脚本都放这里；底层模块会自动读取这些配置
# ============================================================
# 公共配置：项目通用输出目录
COMMON_SETTINGS = {
    "out_dir": "out",
    "experiment_root_dir": "out",
    # 自动发现会递归扫描这些根目录；可继续添加挂载盘或仓库外数据目录。
    # 临时扩展可设置 LIGHTLY_DATASET_SEARCH_ROOTS=/mnt/data1:/mnt/data2。
    "dataset_search_roots": ["datasets"],
    # 用于搜索历史推理结果；新结果默认归档到对应实验目录。
    "test_output_root_dir": "out",
    "eda_output_root_dir": "out/EDA",
    "all_report_root_dir": "out/all_report",
}

# 检测配置：保留在一个地方，通过注释和空行分组
DET_SETTINGS = {
    # 基础配置
    # 这两个最常改：改数据集或实验目录时，下面大部分 det 默认路径会自动跟着变
    "det_dataset_dir": "datasets/convert_datasets/NEU-DET811kuozeng1bei",
    # 留空 / None / "auto" 时，会自动从 out/ 下检索最近的 det 实验目录
    "det_experiment_dir": None,
    "det_default_split": "test",

    # 阈值配置
    # det_score_threshold:
    #   推理时的置信度阈值。默认 0.3 更偏向少漏检；误检多时可再调高。
    # det_report_iou_threshold:
    #   评估 test_report.json 时，预测框和标注框的 IoU 命中阈值。
    #   指标太严格就调低，例如 0.5 -> 0.4。
    "det_score_threshold": 0.3,
    "det_report_iou_threshold": 0.5,

    # det eval 默认只抽样输出少量 [原图|GT|预测] 对比图，完整指标仍覆盖全部图片。
    # 0 表示关闭数量限制。
    "det_eval_vis_max_images": 50,

    # 可选覆盖项
    # 如果你不填，系统会自动推导：
    # det_data_yaml          -> <det_dataset_dir>/data.yaml
    # det_infer_image_dir    -> <det_dataset_dir>/images/<det_default_split>
    # det_infer_output_dir   -> <experiment_dir>/infer/<dataset>-<id>/<split>
    # det_eval_output_dir    -> <experiment_dir>/eval/<dataset>-<id>/<split>
    # det eval 会在同一目录生成对应 split 的指标与 *_report.json。
    # det_export_report_json 默认自动匹配当前数据集最近的 eval 报告。
    #
    # 只有你确实想和自动规则不一样时，再单独打开某一项覆盖。
    # "det_data_yaml": "datasets/wuwanPic_dataset/dataset_det/data.yaml",
    # "det_infer_image_dir": "datasets/wuwanPic_dataset/dataset_det/images/test",
    # "det_infer_output_dir": "out/2026-04-14/NEU_train/infer-neu-test",
    # "det_eval_output_dir": "out/2026-04-14/NEU_train/eval-neu-test",
    # "det_export_report_json": "out/2026-04-14/NEU_train/eval-neu-test/eval-neu-test-test_report.json",

    # export 默认配置
    # det_export_source_data 不填时，默认跟 det_data_yaml 一致
    # det_export_good_class_threshold:
    #   类别 AP 参考阈值。
    #   自动模式下它进入分析报告；严格模式下它参与删类。
    # det_export_min_class_images:
    #   一个类别至少要有多少张图才参与导出。
    #   自动模式下会结合 8:1:1 重划分要求做保底推导，默认会倾向保留至少能覆盖 train/val/test 的类别。
    # det_export_min_class_boxes:
    #   一个类别至少要有多少个框才参与导出。
    #   填 0 时按 det_export_balance_ratio 自动推导。
    #   高密度图片过滤后还会再复核一次，避免某个类别最后只剩几个框。
    # det_export_target_images_per_class:
    #   train 导出时，每个类别尽量靠近这个图数。
    #   填 0 时自动取可用类别图数的中位数。
    # det_export_target_total_images:
    #   导出后总图片数目标。填 0 表示由筛选结果自动决定。
    #   填入后，系统会优先围绕这个总量预算自动推导类别下限、框密度阈值和每类目标框数。
    # det_export_split_ratio:
    #   导出后重新划分的 train:val:test 比例。
    #   交互模式固定使用 8:1:1；用户只需要输入总图数，其余阈值优先走自动分析。
    # det_export_target_boxes_per_class:
    #   train 导出时，每个类别尽量靠近这个框数。
    #   填 0 时按 det_export_balance_ratio 自动推导。
    # det_export_balance_ratio:
    #   导出后 train 中最多和最少类别的框数比例上限。
    # det_export_auto_balance:
    #   导出前先做标签分布分析，并自动推导保留阈值和每类框数上限。
    # det_export_auto_relax_class_threshold:
    #   开启后，AP 阈值进入参考分析，类别保留主要看标签数量和平衡规则。
    #   关闭后，AP 阈值进入严格删类。
    # det_export_max_boxes_per_image:
    #   单张图保留类别的总框数上限。填 0 时自动模式会按总图数预算和框密度分布推导。
    # det_export_max_boxes_per_class_per_image:
    #   单张图里同一类别的框数上限。填 0 时自动模式会按总图数预算和框密度分布推导。
    # det_export_box_density_penalty:
    #   选图时对高框密度图片的惩罚强度，越大越偏向少框图片。
    #   自动模式下会以这个值为基础，按筛选压力向上调整。
    # "det_export_source_data": "datasets/wuwanPic_dataset/dataset_det/data.yaml",
    "det_export_target_total_images": 0,
    "det_export_good_class_threshold": 0.0,
    "det_export_auto_balance": True,
    "det_export_auto_relax_class_threshold": True,
    "det_export_balance_ratio": 0.0,
    "det_export_min_class_images": 0,
    "det_export_min_class_boxes": 0,
    "det_export_target_images_per_class": 0,
    "det_export_split_ratio": "8:1:1",
    "det_export_target_boxes_per_class": 0,
    "det_export_max_boxes_per_image": 0,
    "det_export_max_boxes_per_class_per_image": 0,
    "det_export_box_density_penalty": 0.0,
    # det_export_size_ratio:
    #   目标尺寸占比 小:中:大（排除 tiny<16²）。默认 "1:1:1"（均衡三档），填 "" 关闭尺寸感知、回退旧选图。
    "det_export_size_ratio": "1:1:1",
    # det_export_size_balance_weight: 尺寸赤字相对类别赤字的联合权重，默认 1.0。
    "det_export_size_balance_weight": 1.0,
    # det_export_avg_boxes_per_image_min/max: 平均每图框数软目标区间，默认 5~15；都填 0 关闭。
    "det_export_avg_boxes_per_image_min": 5,
    "det_export_avg_boxes_per_image_max": 15,
    # det_export_trim_boxes: 选图后把超出均衡窗口的多数类多余框从标签里删掉，让各类
    #   框数真正落入 balance_ratio 窗口。只选图受共现物理底限制时(多数类作"乘客"被动
    #   超配额)，这是把类别压到完全均衡的硬手段。代价：被删的框变成无标签实例(训练时
    #   潜在漏标)。默认 False；要求严格类别均衡时设 True。
    "det_export_trim_boxes": False,
    "det_export_suffix": "_A",
}

# SAHI 切片推理配置：只影响 det infer，且仅当命令行带 --sahi 时才启用。
# 不加 --sahi 时整条 infer 流程与以前完全一致，这里的参数也不会生效。
# 这里只调默认切片参数；要不要切片由 --sahi 决定，所以 det_sahi_enabled 保持 False。
SAHI_SETTINGS = {
    "det_sahi_enabled": False,        # 保持 False：是否切片由命令行 --sahi 决定
    "det_sahi_overlap": 0.2,          # tile 重叠比例 [0,1)
    "det_sahi_nms_iou": 0.3,          # tile 间 NMS 的 IoU 阈值
    "det_sahi_global_local_iou": 0.1, # 全局/局部一致性匹配阈值
    # True：短边 < 模型 tile(通常640) 的小图自动跳过 SAHI、回退普通推理。
    # 小图上 SAHI 反而更差，开启后混合尺寸数据集里大图切片、小图走普通推理。
    "det_sahi_skip_small_images": True,
}

# 分类配置：后面如果要统一 cls 默认路径，就加在这里
CLS_SETTINGS = {
    # 分类推理/评估默认阈值。预测太保守时可以调低。
    "cls_threshold": 0.5,
}

# 分割配置：和 det 一样，常改的路径和导出参数都放在这里
SEG_SETTINGS = {
    # 基础配置
    # 改 seg 数据集时，下面 seg_export_* 的源数据会自动跟着变
    "seg_dataset_dir": "datasets/convert_datasets/sample_yoloseg",
    "seg_train_type": "instance",

    # seg_experiment_dir:
    #   infer/eval 未显式给 --experiment-dir/--checkpoint 时的默认实验目录。
    #   "auto"（默认）自动发现 out/ 下最近一次含 checkpoint 的 seg 实验（train_seg.py 的 OUT）。
    #   也可写死具体目录，如 "out/xxx"；或每次用 --experiment-dir 覆盖。
    "seg_experiment_dir": "auto",
    # 语义分割默认指向真实存在的数据集（neu_dataset 下没有 dataset_semantic）。
    "semantic_seg_dataset_dir": "datasets/aeroscapes/dataset_semantic",
    "semantic_seg_data_yaml": "datasets/aeroscapes/dataset_semantic/data.yaml",

    # 阈值配置
    # seg_threshold:
    #   推理/评估默认阈值。掩码太少就调低，噪声太多就调高。
    "seg_threshold": 0.8,

    # seg_eval_vis_max_images:
    #   实例评估对比图最多出多少张，好/差各半、类别尽量全，避免占满磁盘。
    #   0 表示不限制、给每张有标注的图都出对比图（旧行为）。
    "seg_eval_vis_max_images": 100,

    # export 默认配置（参数语义对齐 det，关键词把"框"换成"实例"）
    # seg_export_source_data 不填时，默认 <seg_dataset_dir>/data.yaml
    # seg_export_report_json 不填或填 "auto" 时，自动检索最近的 seg_eval_summary.json
    # seg_export_good_class_threshold:
    #   类别 AP 参考阈值。自动模式下进入分析报告；严格模式下参与删类。
    # seg_export_min_class_images:
    #   一个类别至少要有多少张图才参与导出。
    # seg_export_min_class_instances:
    #   一个类别至少要有多少个实例（多少条 polygon）才参与导出。
    #   填 0 时按 seg_export_balance_ratio 自动推导。
    # seg_export_target_images_per_class:
    #   train 导出时，每个类别尽量靠近这个图数。填 0 自动取中位数。
    # seg_export_target_total_images:
    #   导出后总图片数目标。填 0 表示由筛选结果自动决定；
    #   填入后系统会围绕这个总量预算自动推导类别下限、实例密度阈值和每类目标实例数。
    # seg_export_split_ratio:
    #   导出后重新划分的 train:val:test 比例（交互模式固定 8:1:1）。
    # seg_export_target_instances_per_class:
    #   train 导出时每个类别尽量靠近这个实例数。填 0 按 balance_ratio 自动推导。
    # seg_export_balance_ratio:
    #   导出后 train 中最多和最少类别的实例数比例上限。
    # seg_export_auto_balance:
    #   导出前先做标签分布分析，并自动推导保留阈值和每类实例数上限。
    # seg_export_auto_relax_class_threshold:
    #   开启后 AP 阈值仅作参考；关闭后 AP 阈值进入严格删类。
    # seg_export_max_instances_per_image:
    #   单张图保留类别的实例总数上限。填 0 自动按总图数预算和密度分布推导。
    # seg_export_max_instances_per_class_per_image:
    #   单张图里同一类别的实例数上限。填 0 自动推导。
    # seg_export_instance_density_penalty:
    #   选图时对高实例密度图片的惩罚强度，越大越偏向少实例图片。
    #   自动模式下会以这个值为基础，按筛选压力向上调整。
    # "seg_export_source_data": "datasets/neu_dataset/dataset_seg/data.yaml",
    # "seg_export_report_json": "auto",
    "seg_export_target_total_images": 0,
    "seg_export_good_class_threshold": 0.0,
    "seg_export_auto_balance": True,
    "seg_export_auto_relax_class_threshold": True,
    "seg_export_balance_ratio": 0.0,
    "seg_export_min_class_images": 0,
    "seg_export_min_class_instances": 0,
    "seg_export_target_images_per_class": 0,
    "seg_export_split_ratio": "8:1:1",
    "seg_export_target_instances_per_class": 0,
    "seg_export_max_instances_per_image": 0,
    "seg_export_max_instances_per_class_per_image": 0,
    "seg_export_instance_density_penalty": 0.0,
    # seg_export_size_ratio: 目标掩码尺寸占比 小:中:大（排除 tiny）。默认 "1:1:1"，"" 关闭。
    "seg_export_size_ratio": "1:1:1",
    "seg_export_size_balance_weight": 1.0,
    "seg_export_avg_instances_per_image_min": 5,
    "seg_export_avg_instances_per_image_max": 15,
    "seg_export_suffix": "_A",
}

def build_user_settings() -> dict[str, object]:
    groups = [COMMON_SETTINGS, CLS_SETTINGS, DET_SETTINGS, SAHI_SETTINGS, SEG_SETTINGS]
    seen: set[str] = set()
    duplicates: set[str] = set()
    for group in groups:
        for key in group:
            if key in seen:
                duplicates.add(key)
            seen.add(key)
    if duplicates:
        raise ValueError(f"配置项重复定义: {', '.join(sorted(duplicates))}")

    # 把分组配置合并成底层模块统一使用的一份 settings
    settings: dict[str, object] = {}
    for group in groups:
        settings.update(group)

    for key, value in settings.items():
        if isinstance(value, str) and value.strip().casefold() in {"true", "false"}:
            raise TypeError(f"布尔配置 {key} 必须写 True/False，不能写字符串 {value!r}")
        if key.endswith("_balance_ratio"):
            if type(value) not in {int, float}:
                raise TypeError(f"配置 {key} 必须是数字，实际为 {value!r}")
            ratio = float(value)
            if not math.isfinite(ratio):
                raise ValueError(f"配置 {key} 必须是有限数字，实际为 {value}")
            if ratio != 0.0 and ratio < 1.0:
                raise ValueError(f"配置 {key} 必须为 0（自动）或大于等于 1，实际为 {value}")
            continue
        if key.endswith(("_threshold", "_overlap", "_iou")):
            if "_auto_" in key and isinstance(value, bool):
                continue
            if type(value) not in {int, float}:
                raise TypeError(f"配置 {key} 必须是数字，实际为 {value!r}")
            number = float(value)
            if not math.isfinite(number) or not 0.0 <= number <= 1.0:
                raise ValueError(f"配置 {key} 必须位于 [0, 1]，实际为 {value}")
            if key.endswith("_overlap") and number >= 1.0:
                raise ValueError(f"配置 {key} 必须位于 [0, 1)，实际为 {value}")
        if key.endswith(("_workers", "_images", "_instances", "_boxes")):
            if type(value) is bool:
                continue
            if type(value) is not int:
                raise TypeError(f"配置 {key} 必须是整数，实际为 {value!r}")
            if value < 0:
                raise ValueError(f"配置 {key} 不能为负数，实际为 {value}")
        if key.endswith(("_penalty", "_weight")):
            if type(value) not in {int, float} or not math.isfinite(float(value)):
                raise TypeError(f"配置 {key} 必须是有限数字，实际为 {value!r}")
            if float(value) < 0.0:
                raise ValueError(f"配置 {key} 不能为负数，实际为 {value}")
    roots = settings.get("dataset_search_roots")
    if not isinstance(roots, (str, Path, list, tuple)):
        raise TypeError("配置 dataset_search_roots 必须是路径字符串或路径列表")
    if isinstance(roots, (list, tuple)) and any(
        not isinstance(root, (str, Path)) for root in roots
    ):
        raise TypeError("配置 dataset_search_roots 中的每一项都必须是路径")
    if settings.get("det_default_split") not in {"train", "val", "test"}:
        raise ValueError("配置 det_default_split 必须是 train、val 或 test")
    if settings.get("seg_train_type") not in {"instance", "semantic"}:
        raise ValueError("配置 seg_train_type 必须是 instance 或 semantic")
    for prefix, unit in (("det_export", "框"), ("seg_export", "实例")):
        minimum = float(settings[f"{prefix}_avg_{'boxes' if prefix == 'det_export' else 'instances'}_per_image_min"])
        maximum = float(settings[f"{prefix}_avg_{'boxes' if prefix == 'det_export' else 'instances'}_per_image_max"])
        if not math.isfinite(minimum) or not math.isfinite(maximum):
            raise ValueError(f"配置 {prefix} 的平均每图{unit}数必须是有限数字")
        if minimum < 0 or maximum < 0:
            raise ValueError(f"配置 {prefix} 的平均每图{unit}数不能为负数")
        if minimum > maximum and maximum > 0:
            raise ValueError(f"配置 {prefix} 的平均每图{unit}数最小值不能大于最大值")
        split_ratio = str(settings[f"{prefix}_split_ratio"]).strip()
        parts = [part.strip() for part in split_ratio.split(":")]
        if len(parts) != 3:
            raise ValueError(f"配置 {prefix}_split_ratio 必须使用 train:val:test 三段格式")
        try:
            ratios = [float(part) for part in parts]
        except ValueError as exc:
            raise ValueError(f"配置 {prefix}_split_ratio 必须包含三个数字") from exc
        if any(not math.isfinite(ratio) or ratio < 0 for ratio in ratios) or sum(ratios) <= 0:
            raise ValueError(f"配置 {prefix}_split_ratio 必须是总和大于 0 的非负有限数字")
        size_ratio = str(settings[f"{prefix}_size_ratio"]).strip()
        if size_ratio:
            size_parts = [part for part in re.split(r"[\s/:,]+", size_ratio) if part]
            if len(size_parts) != 3:
                raise ValueError(f"配置 {prefix}_size_ratio 必须包含小、中、大三个数字")
            try:
                size_values = [float(part) for part in size_parts]
            except ValueError as exc:
                raise ValueError(f"配置 {prefix}_size_ratio 必须包含三个数字") from exc
            if any(not math.isfinite(ratio) or ratio < 0 for ratio in size_values) or sum(size_values) <= 0:
                raise ValueError(f"配置 {prefix}_size_ratio 必须是总和大于 0 的非负有限数字")
    return settings


def main(argv: list[str] | None = None) -> int:
    if argv is None:
        argv = sys.argv[1:]
    debug = "--debug" in argv
    argv = [arg for arg in argv if arg != "--debug"]
    if any(arg in {"-h", "--help"} for arg in argv):
        parse_cli_args(argv)
        return 0
    try:
        rt.apply_user_settings(build_user_settings())
        if debug:
            print(f"[debug] LightlyTrain 源码目录: {rt.SRC_DIR}", file=sys.stderr)
        args = parse_cli_args(argv) if argv else build_interactive_args()
        if args is None:
            return 0
        dispatch(validate_args(args))
        return 0
    except InputCancelled as exc:
        print(f"\n{exc}", file=sys.stderr)
        return 0
    except KeyboardInterrupt:
        print("\n已取消。", file=sys.stderr)
        return 130
    except Exception as exc:
        if debug:
            raise
        print(f"错误: {exc}", file=sys.stderr)
        print("使用 --debug 查看完整 traceback。", file=sys.stderr)
        return 2
    finally:
        cleanup_pending_optimize_temp_dirs()


if __name__ == "__main__":
    raise SystemExit(main())
