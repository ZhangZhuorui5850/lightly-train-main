#!/usr/bin/env python3
"""数据集质检抽样导出 CLI。

用法:
    # 基础模式（纯几何检测）
    python dataset_review_export.py --data datasets/.../data.yaml --out out/review_subset

    # 含模型分析
    python dataset_review_export.py --data datasets/.../data.yaml --out out/review_subset \
        --experiment-dir out/2026-06-01/NEU_test

    # 交互模式（推荐）
    python dataset_review_export.py --interactive
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# 确保项目根目录在 sys.path
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tool_lib import common as rt
from tool_lib.det_review_sample import run_review_pipeline, run_review_pipeline_with_model


def main() -> None:
    parser = argparse.ArgumentParser(
        description="数据集质检抽样：扫描 → 问题检测 → 贪心选图 → 导出 X-AnyLabeling 复核子集"
    )
    # 基础参数
    parser.add_argument("--data", help="data.yaml 路径")
    parser.add_argument("--out", help="输出目录")
    parser.add_argument("--interactive", action="store_true", help="交互模式（推荐）")

    # 模型分析参数
    parser.add_argument("--experiment-dir", type=str, default=None, help="实验目录（含推理结果）")
    parser.add_argument("--infer-output-dir", type=str, default=None, help="推理输出目录")
    parser.add_argument("--report-path", type=str, default=None, help="test_report.json 路径")

    # 功能开关
    parser.add_argument("--enable-geometry", dest="enable_geometry", action="store_true", default=True)
    parser.add_argument("--disable-geometry", dest="enable_geometry", action="store_false")
    parser.add_argument("--enable-model-analysis", dest="enable_model_analysis", action="store_true", default=True)
    parser.add_argument("--disable-model-analysis", dest="enable_model_analysis", action="store_false")
    parser.add_argument("--enable-outlier-class", dest="enable_outlier_class", action="store_true", default=True)
    parser.add_argument("--disable-outlier-class", dest="enable_outlier_class", action="store_false")
    parser.add_argument("--enable-visualization", dest="enable_visualization", action="store_true", default=False)
    parser.add_argument("--disable-visualization", dest="enable_visualization", action="store_false")

    # 选图参数
    parser.add_argument("--k", type=int, default=3, help="每类最少抽样数 (默认 3)")
    parser.add_argument("--alpha", type=float, default=3.0, help="配额上浮系数 (默认 3.0)")
    parser.add_argument("--cap", type=int, default=10, help="每类配额上限 (默认 10)")
    parser.add_argument("--max-images", type=int, default=500, help="最大导出图数上限 (默认 500)")
    parser.add_argument("--min-only", action="store_true", help="最小覆盖模式：每类仅需 1 张")
    parser.add_argument("--problems-only", action="store_true", help="仅导出问题图（不做额外覆盖补充）")

    # 几何检测参数
    parser.add_argument("--groups", type=str, default=None, help="class_groups.yaml 路径")
    parser.add_argument("--groups-section", type=str, default="coco80", help="class_groups.yaml 的 section 名")
    parser.add_argument("--cache", type=str, default=None, help="图尺寸缓存 JSON 路径")
    parser.add_argument("--seed", type=int, default=42, help="随机种子 (默认 42)")
    parser.add_argument("--iou-dup", type=float, default=0.9, help="重复标注 IoU 阈值 (默认 0.9)")
    parser.add_argument("--iou-conflict", type=float, default=0.5, help="冲突标注 IoU 阈值 (默认 0.5)")
    parser.add_argument("--min-box-px", type=float, default=4, help="最小框像素阈值 (默认 4)")
    parser.add_argument("--dense-top-n", type=int, default=None, help="dense 标记 Top-N（默认自适应）")

    # 模型分析参数
    parser.add_argument("--match-iou-threshold", type=float, default=0.5, help="匹配IoU阈值 (默认 0.5)")
    parser.add_argument("--low-conf-threshold", type=float, default=0.3, help="低置信度阈值 (默认 0.3)")
    parser.add_argument("--outlier-class-ap", type=float, default=0.1, help="劣质类别AP阈值 (默认 0.1)")
    parser.add_argument("--outlier-class-gt", type=int, default=5, help="劣质类别GT数阈值 (默认 5)")

    # 其他
    parser.add_argument("--no-embed-images", action="store_true", help="LabelMe JSON 不嵌入 base64 图片数据")

    args = parser.parse_args()

    # 交互模式
    if args.interactive:
        from tool_lib.interactive import build_interactive_review_sample_args
        args = build_interactive_review_sample_args()
        if args is None:
            return

    # 验证必需参数
    if not args.data:
        print("错误: 必须指定 --data 参数", file=sys.stderr)
        sys.exit(1)

    data_path = Path(args.data).expanduser().resolve()
    out_dir = Path(args.out).expanduser().resolve() if args.out else None
    groups_path = Path(args.groups).expanduser().resolve() if args.groups else None
    cache_path = Path(args.cache).expanduser().resolve() if args.cache else None
    experiment_dir = Path(args.experiment_dir).expanduser().resolve() if args.experiment_dir else None
    infer_output_dir = Path(args.infer_output_dir).expanduser().resolve() if args.infer_output_dir else None
    report_path = Path(args.report_path).expanduser().resolve() if args.report_path else None

    if not data_path.exists():
        print(f"错误: data.yaml 不存在: {data_path}", file=sys.stderr)
        sys.exit(1)
    if groups_path and not groups_path.exists():
        print(f"警告: class_groups.yaml 不存在: {groups_path}，跳过冲突检测", file=sys.stderr)
        groups_path = None

    # 如果指定了experiment_dir但没有infer_output_dir，自动查找
    if experiment_dir and not infer_output_dir:
        from tool_lib.det_problem_export import discover_infer_runs
        runs = discover_infer_runs(experiment_dir)
        if runs:
            infer_output_dir = runs[0].get("output_dir")
            if report_path is None:
                report_path = runs[0].get("report_path")

    # 确定输出目录
    if out_dir is None:
        import time
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        out_dir = rt.EXPERIMENT_ROOT_DIR / f"review_subset_{timestamp}"

    # 运行质检抽样
    result = run_review_pipeline_with_model(
        data_path=data_path,
        out_dir=out_dir,
        experiment_dir=experiment_dir,
        infer_output_dir=infer_output_dir,
        report_path=report_path,
        # 功能开关
        enable_geometry=args.enable_geometry,
        enable_model_analysis=args.enable_model_analysis,
        enable_outlier_class=args.enable_outlier_class,
        enable_visualization=args.enable_visualization,
        # 选图参数
        k=args.k,
        alpha=args.alpha,
        cap=args.cap,
        max_images=args.max_images,
        min_only=args.min_only,
        problems_only=args.problems_only,
        # 几何检测参数
        iou_dup=args.iou_dup,
        iou_conflict=args.iou_conflict,
        min_box_px=args.min_box_px,
        dense_top_n=args.dense_top_n,
        # 模型分析参数
        match_iou_threshold=args.match_iou_threshold,
        low_conf_threshold=args.low_conf_threshold,
        outlier_class_ap=args.outlier_class_ap,
        outlier_class_gt=args.outlier_class_gt,
        # 其他
        groups_path=groups_path,
        groups_section=args.groups_section,
        cache_path=cache_path,
        seed=args.seed,
        embed_images=not args.no_embed_images,
    )

    print(f"\n{'=' * 63}")
    print(f"  完成！")
    print(f"{'=' * 63}")
    print(f"\n  输出目录: {result}")
    print(f"\n  ── 下一步 ──")
    print(f"  1. 将 review_subset/ 拷贝到本地")
    print(f"  2. 用 X-AnyLabeling 打开 images/ 文件夹")
    print(f"  3. 按 description 字段排序，优先复核高风险 flag")
    print(f"  4. 复核完毕后可回灌训练")
    print(f"\n  详细报告: {result / 'qc_report.md'}")


if __name__ == "__main__":
    main()

