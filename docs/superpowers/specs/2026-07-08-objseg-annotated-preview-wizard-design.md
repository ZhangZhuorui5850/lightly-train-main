# 标注预览 + 物体版向导 设计

日期: 2026-07-08
状态: 已批准设计,待写实现计划
关联: 建立在 `2026-07-08-objectseg-to-mvtec-design.md`(物体版转换)之上。

## 背景与需求

物体版流水线里,人要按"物体"把图分进 `staging/<中文物体名>/` 文件夹。难点:源数据只有**缺陷**标签、没有物体标签,一张图上如果有分属不同物体的多种缺陷,就没法干净地归到单个物体文件夹。用户需要**直观地看到每张图的标注**,才能判断"这张图是不是只对应一个物体"。

需求(用户原话归纳):
1. 抽出的样本图要**把标注画在图上**(缺陷多边形 + 类别名),便于肉眼判断物体/缺陷是不是一对多。
2. 图**右边加一栏信息**,从几个角度写这张图的信息(类别等)。
3. 整个过程做成**交互引导式、分两步**:第1步挑数据(生成预览)→ 用户下载、本地分图、上传服务器 → 第2步跑脚本按文件名+源数据转换。

## 已确认决策

| 决策点 | 选择 |
|---|---|
| 中文字体 | 复用 `tool_lib/msyh.ttc`,经 `tool_lib/common.load_cjk_font()`;`--font` 可覆盖;缺字体退化 PIL 默认(已内置兜底) |
| 渲染库 | PIL(Pillow 12.2 已装);当前 seg_sample_browse 的 cv2 读图改为 PIL |
| 预览布局 | 每图一张:左=原图+缺陷多边形+中文名,右=丰富信息栏 |
| 信息栏内容 | 文件名、来源split、缺陷种类数(>1 标 ⚠)、逐类"● 中文名 ×数量"带色块、多边形总数 |
| 颜色 | `class_color(cls)` 固定调色板,整轮一致(同类同色) |
| 编排 | 新增交互向导 `objseg_wizard.py`,分两步引导;底层两命令仍可单用 |

## 非目标

- 不自动判定"多物体"(工具只有缺陷标签,做不到)——只把强信号(几种缺陷、各在哪)摆出来给人判断。
- 不做一个不间断的脚本会话(中间有下载/分图/上传人工往返)——向导是"引导式两步"。
- 不改动物体版转换核心 `objectseg_to_mvtec.py` 的转换逻辑。

## 架构:三个组件

### 组件 A — 升级 `convert_tools/seg_sample_browse.py`

当前:拷贝原图 + 写 `sample_index.csv`(stem, split, defects)。升级为生成**标注预览图**。

新增/修改的函数:

- `class_color(cls: int) -> tuple[int,int,int]`:从固定调色板(≥12 色)按 `cls % len` 取色,确定性。同一 cls 永远同色。
- `render_preview(img_path, polys, names, font, panel_w=320) -> PIL.Image`:
  - 用 PIL 打开原图转 RGB。
  - 在一张 RGBA overlay 上:每个多边形填充半透明该类色(alpha≈70),描边 2px 实色;`Image.alpha_composite` 合回原图。
  - 在每个多边形的最上顶点附近画中文类别名,带该类色的小底框(用 `_safe_text_size` 量文字尺寸)。
  - 右侧信息栏(白底,宽 `panel_w`,高=图高):逐行画
    - `文件: <stem>`、`来源: <split>`
    - 分隔线
    - `缺陷种类: <N>`;当 N>1 追加 ` ⚠`
    - 每个出现的类别一行:`● <中文名> ×<该类多边形数>`,行首画一个该类色的实心小方块
    - 分隔线
    - `多边形总数: <M>`
  - 左图 + 右栏横向拼成一张 canvas 返回。
- `browse(src, out, limit=500, font_path=None) -> int`:
  - 沿用现有抽样(跨 split 取前 `limit` 个标签)。
  - 对每张:找图、`parse_label` 得多边形;调 `render_preview` 存 `<out>/<stem>.png`。
  - 找不到图的样本跳过(沿用现有行为)。
  - 写 `sample_index.csv`,列: `stem, split, defects, n_defect_classes, n_polygons`。
  - 返回成功渲染的张数。
  - 字体:`font_path or tool_lib/msyh.ttc`,经 `load_cjk_font`(size 用两档:标题/正文)。
- `main()`:`--src --out --limit`,新增 `--font`。

依赖:`from PIL import Image, ImageDraw`。**不 import `tool_lib.common`**——把 `load_cjk_font` / `_safe_text_size` 这两个极小助手直接复制进本文件,字体默认路径用 `tool_lib/msyh.ttc` 的绝对路径解析(理由见"风险")。复用现有 `IMG_EXTS, SPLITS, load_names, parse_label`。

### 组件 B — 新增 `convert_tools/objseg_wizard.py`

交互向导,把两步用文字引导串起来。

- `run_generate(src, out, limit, font_path=None) -> int`:调 `seg_sample_browse.browse(...)`;打印引导文案(预览图位置、下载→本地按物体分图→上传到 `staging/`→回来跑第2步)。返回张数。
- `run_build(staging, src, out, clean) -> dict`:调 `objectseg_to_mvtec.convert(...)`;返回统计。
- `main()`:交互——打印两步总览,`input()` 问选哪步;按选择再 `input()` 收该步参数,调对应 `run_*`。交互层保持薄。
- `convert.py` 注册 `obj-wizard`(stage "转 MVTec AD",primary=True,interactive=True)。底层 `sample-browse`、`to-mvtec-obj` 保留。

### 组件 C — 文档

- README 更新:把"物体版第0步"改成用 `obj-wizard` 向导的说明 + 预览图长啥样一句话;`--font` 说明。

## 测试

`convert_tools/tests/`:

- `test_class_color_deterministic`:同 cls 两次调用同色;两个不同 cls 不同色。
- `test_browse_renders_annotated_preview`:构造 1 张多类别图(锈蚀+裂纹)的源+图,`browse` 后:
  - `<out>/<stem>.png` 存在,且其宽度 > 原图宽度(证明拼了右栏)。
  - `sample_index.csv` 含列 `n_defect_classes/n_polygons`,该图 `n_defect_classes==2`、`n_polygons==2`。
- `test_browse_missing_image_skipped`:标签有、图缺 → 不计数、不报错。
- `test_wizard_run_generate_forwards`(monkeypatch `seg_sample_browse.browse`)与 `test_wizard_run_build_forwards`(monkeypatch `objectseg_to_mvtec.convert`):验证参数透传;不深测 `input()`。

## 边界与错误处理

| 情况 | 处理 |
|---|---|
| 字体文件缺失 | `load_cjk_font` 退回 PIL 默认字体(中文可能变框,但不崩) |
| 样本图无法打开 | 跳过 + 汇总,不影响其余 |
| 空标签图(good) | 预览照出,信息栏 `缺陷种类: 0`;不特殊处理 |
| 多边形坐标越界 | 画之前 clip 到图尺寸内 |
| 类别 id 越界(names 无此 id) | 信息栏/标注显示 `未知类别<id>`,不崩(与转换核心的硬失败不同,预览要尽量出图) |

## 风险 / 权衡

- **引入 tool_lib.common 依赖**:`load_cjk_font/_safe_text_size` 很小。为避免把 `convert_tools`(数据转换工具,尽量自足)耦合到 `tool_lib`(训练/分析主库,重依赖),**决定:在 `seg_sample_browse.py` 内复制这两个极小助手 + 字体路径解析**(默认指向 `tool_lib/msyh.ttc` 的绝对路径),不 import common。若字体不在该路径,`--font` 兜底。
- 预览是给人看的中间物,渲染细节(配色、字号)可后续微调,不追求一次完美。

## 待实现清单(交给 writing-plans)

1. `seg_sample_browse.py`:`class_color`、`render_preview`、升级 `browse`(标注+新CSV列)、`--font`、字体助手。
2. `objseg_wizard.py`:`run_generate`、`run_build`、`main` 交互。
3. `convert.py`:注册 `obj-wizard`。
4. 测试(组件 C)。
5. README 更新。
