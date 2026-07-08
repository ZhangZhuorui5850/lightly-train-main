#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""数据集转换工具的统一启动台 (single entry point).

只需记住这一个脚本。它本身不做转换，只负责：
  1) 列出所有转换/整理工具(按流水线阶段分组，每个带一句话说明)；
  2) 交互式询问你要做什么，然后把控制权交给对应工具。

用法:
    python convert.py                 # 交互式菜单(记不住用哪个就跑这个)
    python convert.py <command> ...   # 直接运行某工具，后面参数原样转发
    python convert.py <command> -h    # 看该工具自己的参数
    python convert.py list            # 只打印菜单

底层工具都在 ./convert_tools/ 下，每个也能单独运行。
"""
from __future__ import annotations

import shlex
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
TOOLS_DIR = HERE / "convert_tools"


class Tool:
    """一个可分发的工具的登记项。"""

    def __init__(self, script: str, desc: str, stage: str, usage: str,
                 interactive: bool, primary: bool = True):
        self.script = script          # convert_tools/ 下的文件名
        self.desc = desc              # 一句话说明(菜单里显示)
        self.stage = stage            # 流水线阶段(菜单分组)
        self.usage = usage            # 参数提示(选中后显示)
        self.interactive = interactive  # True=直接回车即进入自身交互流程
        self.primary = primary        # False=归到"更多单步/辅助工具"，默认不展开


# command -> Tool。改这里就能增删工具。
# 3 个主流程(primary)平时只用这几个；其余是它们的内部分步或辅助工具(primary=False)，
# 已被主流程包含或很少直接用，默认折叠，`convert.py more` 才展开。
COMMANDS: dict[str, Tool] = {
    # ---- 主流程 ----
    "oneclick": Tool(
        "one_click_convert.py",
        "整理散图/LabelMe → YOLO det/cls/seg 一步到位(内部已含 sync + labelme2yolo)",
        "整理 + 转 YOLO",
        "回车走默认(当前目录源)，或 --help 看参数",
        interactive=True,
    ),
    "to-mvtec": Tool(
        "seg2mvtec_interactive.py",
        "缺陷版(legacy):扫描 *seg 数据集，每个缺陷=一个 category，交互式转 MVTec AD",
        "转 MVTec AD",
        "回车=交互扫描并选择；或加 --all --yes 转全部",
        interactive=True,
    ),
    "to-mvtec-obj": Tool(
        "objectseg_to_mvtec.py",
        "物体版:按人工分好的物体文件夹 + 源seg 生成 MVTec(category=物体，内含各缺陷)",
        "转 MVTec AD",
        "需 --staging <物体文件夹根> --src <seg源> --out <输出> [--clean]",
        interactive=False,
    ),
    "det2seg": Tool(
        "mirror_det_subset_to_seg.py",
        "把挑出的 det 子集按图片名镜像成对应的 seg 子集",
        "子集对齐",
        "回车=扫描 datasets/ 并按相关性排序供选择；或 --det-subset/--seg-source 直接指定",
        interactive=True,
    ),
    # ---- 更多：单步 / 特殊输入 / 辅助(默认折叠) ----
    "coco2sync": Tool(
        "coco_to_synced.py",
        "COCO 2017 标注 → train/val/test 结构(特殊输入，之后接 oneclick)",
        "更多",
        "--help 看参数；需 COCO 根目录",
        interactive=False, primary=False,
    ),
    "sync": Tool(
        "sync_picture.py",
        "只做整理：零散源文件夹 → train/val/test(oneclick 的第 1 步)",
        "更多",
        "--help 看参数；一般需指定源目录与输出目录",
        interactive=False, primary=False,
    ),
    "labelme2yolo": Tool(
        "LabelMeToYOLO.py",
        "只做转换：LabelMe → YOLO det/cls/seg(oneclick 的第 2 步)",
        "更多",
        "--help 看参数；需 LabelMe 源目录",
        interactive=False, primary=False,
    ),
    "seg2mvtec": Tool(
        "yoloseg_to_mvtec.py",
        "只转单个 YOLO-seg 数据集(to-mvtec 的非交互内核)",
        "更多",
        "需 --src <seg目录> --out <输出目录>",
        interactive=False, primary=False,
    ),
    "sample-browse": Tool(
        "seg_sample_browse.py",
        "物体版第0步(可选):抽样摊图,帮你归纳源数据里有哪些物体",
        "更多",
        "需 --src <seg源> --out <浏览目录> [--limit 500]",
        interactive=False, primary=False,
    ),
    "make-sample": Tool(
        "make_sample_yoloseg.py",
        "造一份示例 YOLO-seg 数据(仅 demo/测试用)",
        "更多",
        "回车走默认输出",
        interactive=True, primary=False,
    ),
    "inspect-labelme": Tool(
        "inspect_labelme_shapes.py",
        "统计 / 检查 LabelMe 标注内容(排查用)",
        "更多",
        "--help 看参数；需 LabelMe 目录",
        interactive=False, primary=False,
    ),
}

STAGE_ORDER = ["整理 + 转 YOLO", "转 MVTec AD", "子集对齐", "更多"]


def _ordered_commands(show_all: bool) -> list[tuple[str, Tool]]:
    """按 STAGE_ORDER 稳定排序后的 (command, Tool) 列表；show_all=False 只含主流程。"""
    items: list[tuple[str, Tool]] = []
    for stage in STAGE_ORDER:
        for cmd, tool in COMMANDS.items():
            if tool.stage == stage and (show_all or tool.primary):
                items.append((cmd, tool))
    return items


def print_menu(numbered: bool = False, show_all: bool = False) -> list[str]:
    """打印菜单；numbered=True 时给每项编号，返回编号→command 的顺序表。"""
    print((__doc__ or "").strip().splitlines()[0])
    scope = "全部工具" if show_all else "常用工具"
    print(f"\n{scope}:\n")
    order: list[str] = []
    last_stage = None
    for cmd, tool in _ordered_commands(show_all):
        if tool.stage != last_stage:
            print(f"  [{tool.stage}]")
            last_stage = tool.stage
        order.append(cmd)
        prefix = f"{len(order):>2}. " if numbered else "    "
        print(f"  {prefix}{cmd:<16} {tool.desc}")
    if not show_all:
        print("\n  (输入 more 展开单步/辅助工具，详见 README.md)")
    return order


def _run(tool: Tool, args: list[str]) -> int:
    script = TOOLS_DIR / tool.script
    if not script.exists():
        print(f"脚本不存在: {script}")
        return 1
    # 子进程运行:每个工具自己的 argparse / 交互 / 进度条都原样生效，stdin 继承。
    return subprocess.call([sys.executable, str(script), *args])


def interactive() -> int:
    show_all = False
    while True:
        order = print_menu(numbered=True, show_all=show_all)
        print()
        try:
            raw = input("选择要做的处理 [序号/命令，more 展开全部，q 退出]: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return 0
        if not raw or raw.lower() in ("q", "quit", "exit"):
            return 0
        if raw.lower() in ("more", "all", "更多") and not show_all:
            show_all = True
            print()
            continue
        break
    # 支持序号或命令名
    cmd = None
    if raw.isdigit():
        idx = int(raw)
        if 1 <= idx <= len(order):
            cmd = order[idx - 1]
    elif raw in COMMANDS:
        cmd = raw
    if cmd is None:
        print(f"无效选择: {raw!r}")
        return 2
    tool = COMMANDS[cmd]
    print(f"\n>> {cmd} — {tool.desc}")
    print(f"   {tool.usage}")
    hint = "直接回车进入其交互流程" if tool.interactive else "按提示输入参数(可先输 -h 看帮助)"
    try:
        arg_line = input(f"参数({hint}): ").strip()
    except (EOFError, KeyboardInterrupt):
        print()
        return 0
    args = shlex.split(arg_line) if arg_line else []
    print()
    return _run(tool, args)


def main() -> int:
    argv = sys.argv[1:]
    # 无参数 → 交互菜单
    if not argv:
        return interactive()
    first = argv[0]
    if first in ("-h", "--help", "help", "list", "menu"):
        print_menu()
        return 0
    if first in ("more", "all"):
        print_menu(show_all=True)
        return 0
    if first not in COMMANDS:
        print(f"未知命令: {first!r}\n")
        print_menu()
        return 2
    # 直接分发:剩余参数原样转发给目标工具。
    return _run(COMMANDS[first], argv[1:])


if __name__ == "__main__":
    raise SystemExit(main())
