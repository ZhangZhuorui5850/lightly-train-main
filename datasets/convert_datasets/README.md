# datasets/convert_datasets — 数据集转换工具集

> 记不住用哪个脚本？**只记一个**：`python convert.py`。
> 不带参数会进入**交互菜单**，问你要做什么、给出一句话说明，再帮你调起对应工具。

## 目录结构

```
convert_datasets/
├── convert.py            ← 启动台(交互菜单 + 命令分发)，平时只用这个
├── README.md
└── convert_tools/        ← 所有转换/整理工具(可单独运行，也被 convert.py 调起)
    ├── sync_picture.py
    ├── coco_to_synced.py
    ├── one_click_convert.py
    ├── LabelMeToYOLO.py
    ├── seg2mvtec_interactive.py
    ├── yoloseg_to_mvtec.py
    ├── objectseg_to_mvtec.py
    ├── seg_sample_browse.py
    ├── make_sample_yoloseg.py
    ├── mirror_det_subset_to_seg.py
    └── inspect_labelme_shapes.py
```

## 一图看懂(流水线)

```
零散源文件夹 / COCO / LabelMe
        │  ① 采集整理
        ▼
   train/val/test 结构
        │  ② 转 YOLO
        ▼
 dataset_det / dataset_cls / dataset_seg   ← YOLO 格式
        │  ③ 转 MVTec(只吃 *seg)          │  ④ 子集对齐
        ▼                                  ▼
   dataset_mvtec_ad                    det 子集 ⇄ 对应 seg 子集
   ← 交给多模态团队
```

## 怎么用

```
python convert.py                 # 交互菜单：问你做什么(推荐)
python convert.py <command> ...   # 直接运行某工具，参数原样转发
python convert.py <command> -h    # 看该工具自己的参数
python convert.py list            # 只打印菜单
```

交互模式下：先选序号或命令名 → 再按提示输入参数(纯交互的工具直接回车即可)。
菜单默认只列 **3 个主流程**；输入 `more` 展开单步/辅助工具。

## 主流程(平时只用这 3 个)

| 你想做的事 | 命令 |
|---|---|
| 整理散图/LabelMe → YOLO **det/cls/seg** 一步到位 | `oneclick` |
| 把 `*seg` 数据交互式转成 **MVTec AD**(单个/全部都在这里选) | `to-mvtec` |
| 把挑出的 **det 子集 → 对应 seg 子集** | `det2seg --det-subset <dir> --seg-source <dir>` |

## 更多(单步 / 特殊输入 / 辅助，`convert.py more` 展开)

| 命令 | 说明 |
|---|---|
| `coco2sync` | COCO 2017 → train/val/test(特殊输入，之后接 `oneclick`) |
| `sync` | 只做整理(= `oneclick` 第 1 步) |
| `labelme2yolo` | 只做 LabelMe → YOLO(= `oneclick` 第 2 步) |
| `seg2mvtec` | 只转单个 seg 数据集(= `to-mvtec` 的非交互内核) |
| `make-sample` | 造示例 seg 数据(demo/测试) |
| `inspect-labelme` | 检查 LabelMe 标注内容(排查用) |

> `oneclick` 内部已 import 并串起 `sync` + `labelme2yolo`；`to-mvtec` 内部已包 `seg2mvtec`。
> 所以"更多"里的多是它们的内部分步——保留成独立文件供单独调用，日常不用直接碰。
> 参数会原样转发给 `convert_tools/` 下对应工具，`convert.py` 只负责分发。

## ③ 转 MVTec AD 的约定(交付多模态团队用)

- **结构 = 方案 B**：每个缺陷类 → 一个顶层 MVTec category(20+ 类就 20+ 个文件夹)。
- **good = 相对定义**：对类别 C，所有不含 C 的图算作 C 的 "good"
  (源 train → `train/good/`，源 val/test → `test/good/`)。
- **一图多类**：复制进每个相关 category，各自只保留本类多边形掩码。
- **掩码**：二值 PNG `{0,255}`，与原图同尺寸，命名 `<stem>_mask.png`。
- **只吃 `*seg`**：目录名以 `seg` 结尾才会被扫描；且会**读内容校验**，
  若标签其实是 det 的 4 点 bbox 会被标记 `DET-skip` 拒绝转换。
- **不动原数据**：输出写到同级新目录 `dataset_seg → dataset_mvtec_ad`。

## 物体版 MVTec AD(推荐,category=物体)

适用:多模态/零样本异常检测。category 是**物体**,物体内按缺陷分子文件夹。

三步:
1.(可选)抽样定物体:
   `python convert.py sample-browse` → 看 `<out>/sample_index.csv` 归纳有哪些物体。
2. 人工分图:在 `staging/` 下建**中文物体名**文件夹,把对应图片放进去(一张图只放一个物体)。
3. 生成:
   `python convert.py to-mvtec-obj`(或直接
   `python convert_tools/objectseg_to_mvtec.py --staging <staging> --src <seg源> --out <输出> --clean`)

产出 `<out>/<物体>/{train/good(空), test/good(空), test/<缺陷>/, ground_truth/<缺陷>/}`,
外加 `<out>/object_manifest.csv` 审计清单。脚本按文件名回源查标注、自动发现缺陷、一图多缺陷会复制进各缺陷子文件夹(mask 分拆)。

> 旧的 `to-mvtec`(缺陷版,每个缺陷=一个 category)保留为 legacy。
> 说明:`test/good` 为空 → 图像级 AUROC 无法计算,仅支持像素级定位(符合零样本用途)。

## ④ det 子集 → 对应 seg 子集(`det2seg`)

用 launcher 的 det export 挑出 200 张后，想拿到这 200 张**对应的 seg 数据**：

```
python convert.py det2seg          # 交互式(推荐)
```

交互流程(全程只需回车/选序号，不用手输路径)：
1. 扫描 `datasets/`，列出所有 **det** 数据集 → 选你挑出的那份子集；
2. 列出所有 **seg** 数据集，**按与该 det 子集的"图片名重合度"从高到低排序**
   (★=最匹配，回车即选它)；
3. 选复制方式(copy / symlink / hardlink)→ 确认执行。

也支持非交互直接指定：
```
python convert.py det2seg --det-subset <det子集> --seg-source <seg全量> [--copy-mode symlink]
```

按图片名 stem 匹配，自动识别实例/语义 seg 格式，**沿用 det 子集的 train/val/test 划分**，
输出 `dataset_seg_A_from_det/`(含 data.yaml / classes.txt / mirror_mapping.json)。

## 工具角色速查

| 工具(convert_tools/) | 角色 |
|---|---|
| `../convert.py` | **启动台**：交互菜单 + 命令分发(平时只用这个) |
| `seg2mvtec_interactive.py` | ③ 入口：扫描 + 交互 |
| `yoloseg_to_mvtec.py` | ③ 核心库(被上面 import，也可单独 CLI) |
| `objectseg_to_mvtec.py` | 物体版：按人工分好的物体文件夹 + 源seg 生成 MVTec(category=物体)；对应 `convert.py to-mvtec-obj` |
| `seg_sample_browse.py` | (可选)抽样摊图，帮人归纳源数据有哪些物体；对应 `convert.py sample-browse` |
| `make_sample_yoloseg.py` | ③ 造示例数据 |
| `one_click_convert.py` / `LabelMeToYOLO.py` | ② 转 YOLO 入口 |
| `sync_picture.py` / `coco_to_synced.py` | ① 采集整理入口 |
| `mirror_det_subset_to_seg.py` | ④ det 子集 → 对应 seg 子集 |
| `inspect_labelme_shapes.py` | 辅助：检查 LabelMe |
