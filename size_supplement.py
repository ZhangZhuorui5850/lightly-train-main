"""交互式入口：目标尺寸均衡补充。"""
from __future__ import annotations

import sys
from pathlib import Path

from tool_lib import common as rt
from tool_lib import interactive as it
from tool_lib import det_size_supplement as ss


def main() -> None:
    rt.import_runtime_dependencies()

    print("\n=== 目标尺寸均衡补充 ===\n")

    # 1. 一次列出所有候选，让用户分别选基底和源池
    candidates = it.list_dataset_yaml_candidates(task="det")
    if len(candidates) < 2:
        print("需要至少 2 个 det 数据集才能进行补充。")
        return

    print("可用数据集:")
    for idx, path in enumerate(candidates, start=1):
        print(f"  {idx}. {it.compact_display_path(path)}")

    base_idx = int(input("\n基底数据集编号: ").strip()) - 1
    source_idx = int(input("源数据池编号: ").strip()) - 1
    base_yaml = candidates[base_idx]
    source_yaml = candidates[source_idx]

    # 2. 基底信息预览
    base_summary = ss.summarize_size(base_yaml)
    print(f"\n基底数据集：{base_yaml}")
    print(f"  图数：{base_summary['image_count']}")
    print(f"  尺寸分布：{base_summary['size_buckets']}")

    # 3. 目标比例 & 总图数
    ratio_text = input("\n目标尺寸占比 小/中/大（如 33/33/33）：").strip()
    target_ratio = ss.parse_size_ratio(ratio_text)
    target_total_images = int(input("目标总图数（含基底）：").strip())

    # 4. 确认
    print(f"\n目标比例：{target_ratio}")
    print(f"目标总图数：{target_total_images}")
    confirm = input("确认开始补充？(y/n)：").strip().lower()
    if confirm != "y":
        print("已取消。")
        return

    # 5. 执行
    out_root = ss.run_size_supplement(
        base_data_yaml=base_yaml,
        source_data_yaml=source_yaml,
        target_ratio=target_ratio,
        target_total_images=target_total_images,
        allow_new_classes=True,
    )
    print(f"\n补充完成！输出目录：{out_root}")


if __name__ == "__main__":
    main()
