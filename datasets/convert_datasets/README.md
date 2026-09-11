# datasets/convert_datasets — 数据集转换工具集

> 记不住用哪个脚本？**只记一个**：`python convert.py`。
> 不带参数会进入**交互菜单**，问你要做什么、给出一句话说明，再帮你调起对应工具。

数据转换不再通过仓库根目录的 `launcher.py` 提供入口，避免两套参数和行为漂移。
`--dry-run` 保证零写入；会产生数据集的主流程先写同级 staging 目录，校验成功后再原子发布。
普通文本读取不会自动改写源文件，编码规范化必须由调用方显式请求。
输出路径需要与源数据目录完全分离；现有非空输出默认触发安全拒绝，
`--clean` / `--overwrite` / `--force` 用于原子替换。转换失败时保留已发布的旧输出。

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
    ├── objseg_wizard.py
    ├── seg_sample_browse.py
    ├── make_sample_yoloseg.py
    ├── mirror_det_subset_to_seg.py
    ├── generated_mask_to_yoloseg.py
    ├── generated2seg_interactive.py
    ├── mvtec_to_yolo.py
    ├── dataset_discovery.py
    ├── dataset_detector.py
    ├── text_encoding.py
    ├── interactive_helpers.py
    ├── output_naming.py
    ├── conversion_wizard.py
    ├── yoloseg_to_semantic.py
    ├── semantic_to_xanylabeling.py
    ├── class_editor.py
    ├── dataset_merger.py
    ├── semantic_class_editor.py
    ├── semantic_dataset_merger.py
    ├── yolo_class_editor.py
    ├── yolo_dataset_merger.py
    ├── balanced_test_selector.py
    ├── rebalance_yolo_splits.py
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
python convert.py doctor          # 检查注册脚本和统一安全契约
python convert.py datasets        # 快速检测、筛选并诊断数据集路径
python convert.py auto            # 扫描项目数据集并显示可执行转换
python convert.py datasets --datasets /data/a --datasets /mnt/b
python convert.py auto --datasets /data/a --datasets /mnt/b --list
python convert.py auto --dry-run  # 完成交互选择并把零写入预览传给目标工具
python convert.py yolo2semantic   # YOLO polygon txt 转 PNG 语义掩码
python convert.py semantic2xany   # PNG 语义 mask 生成 X-AnyLabeling 导入映射
python convert.py merge-datasets  # 合并 Det/PNG Semantic/YOLO Seg 数据集
python convert.py edit-classes    # 检查、删除、合并、重命名数据集类别并重排 ID
python convert.py balanced-test   # 长尾诊断后，从指定 split 挑选固定数量 Det/Seg 测试图片
python convert.py rebalance-splits # 汇总 labels/masks 的 train/val/test 并均衡重划分
python convert.py mvtec2yolo ...  # 标准 MVTec 同时转 YOLO Seg 和 Det
```

交互模式下：先选序号或命令名 → 再按提示输入参数。`b/back` 返回菜单，`q/quit` 退出；
命令或参数输入错误会留在当前启动台继续修正。非交互工具的参数处直接回车会显示帮助。

菜单中的能力标签含义：`只读` 表示只做检测，`dry-run` 表示支持零写入预览，
`原子替换` 表示写入 staging 校验后发布。所有已注册的写入命令都提供 `--dry-run`；
`convert.py doctor` 会检查注册表，并实际运行全部工具的 `--help`，确认启动成功，同时检查写入工具公开 `--dry-run` 参数。
输出目录会在路径解析前检查软链接；重新划分工具的 `--dry-run` 同样校验源输出目录分离。
菜单默认列出常用主流程；输入 `more` 展开单步/辅助工具。

## 主流程

| 你想做的事 | 命令 |
|---|---|
| 快速检测 Cls/Det/Seg/Semantic/LabelMe/MVTec 数据集并诊断路径 | `datasets` |
| 自动检索项目中的数据集、识别格式、选择转换目标 | `auto` |
| YOLO Seg polygon txt → PNG 语义掩码 | `yolo2semantic` |
| PNG 语义 mask → X-AnyLabeling MASK 导入映射 JSON | `semantic2xany` |
| 多个同格式数据集 → 一个统一 Det/Semantic/YOLO Seg 数据集 | `merge-datasets` |
| PNG 语义/YOLO Det/YOLO Seg → 检查/删除/合并/重命名类别 | `edit-classes` |
| 整理散图/LabelMe → YOLO **det/cls/seg** 一步到位 | `oneclick` |
| 把 `*seg` 数据交互式转成 **MVTec AD**(单个/全部都在这里选) | `to-mvtec` |
| 把挑出的 **det 子集 → 对应 seg 子集** | `det2seg --det-subset <dir> --seg-source <dir>` |
| YOLO Det/Seg 候选池 → 长尾过滤与固定数量代表性测试集 | `balanced-test` |
| 已有 YOLO labels 或 Semantic masks → 汇总后按类别均衡重划分 | `rebalance-splits` |
| 图生图返回的 **image + fg 掩码 → YOLO Seg** | `generated2seg` |
| 标准 **MVTec AD → YOLO Seg + Det** | `mvtec2yolo --src <目录> --out <目录>` |

## 统一数据集检测器 (`datasets`)

所有转换、编辑、合并和重划分功能共用 `dataset_detector.py`。默认快速模式对每个候选最多
统计 200 张图片和 200 份标注，并从最多 24 个 YOLO 标签文件采样格式；列表中的 `≥200`
表示已达到快速扫描上限。数据集按最后修改时间从新到旧排列。

交互终端使用动态进度条；nohup、CI 和日志重定向环境每 5 秒输出一次包含阶段、数量、速度和
耗时的快照。通过 `LIGHTLY_PROGRESS_SNAPSHOT_INTERVAL=<秒>` 调整间隔，通过
`LIGHTLY_PROGRESS_MODE=off` 关闭显示。

```bash
python convert.py datasets --task det
python convert.py datasets --task cls
python convert.py datasets --task seg --datasets /path/to/datasets
python convert.py datasets --task all --include-empty --include-unknown
python convert.py datasets --task det --exact-counts
python convert.py datasets --diagnose /path/to/dataset/data.yaml
```

路径解析同时支持 `images/{train,val,test}` + `labels/{train,val,test}` 和
`{train,val,test}/images` + `{train,val,test}/labels`，也识别 `valid`、`validation` 等别名、
TXT 图片清单、列表式 split 和自定义名称的 YAML。YAML 内的历史绝对 `path` 失效时，会依据
配置文件所在目录恢复本地数据集根目录。`--diagnose` 会列出每个 split 的配置路径、常规
候选路径及存在状态，用于定位重复目录名、迁移路径和空目录。

扫描器会继续检查已识别数据集旁边的项目子目录，因此上层 `data.yaml` 与下层
`dataset_det/data.yaml`、`dataset_seg/data.yaml` 可以同时进入候选列表。软链接数据目录
同样支持，扫描时通过设备号与 inode 去重来处理链接回环。Det/Seg 专项扫描会在
`images/labels/masks/train/val/test` 内容树入口停止深入，以保持大型数据集上的响应速度。

## 自动扫描与转换向导 (`auto`)

```bash
python convert.py auto
python convert.py auto --list
python convert.py auto --datasets /path/to/datasets
```

向导按以下顺序执行：

1. 选择转换功能。
2. 扫描并展示支持该功能的数据集。
3. 选择数据集并确认输出目录。

扫描器依据目录内容和标签行识别格式，覆盖 YOLO 检测、YOLO 实例分割、PNG 语义分割、
LabelMe、标准 MVTec AD，以及 `image/fg` 生成数据。候选按数据文件最后修改时间从新到旧
排列，列表显示图像数、标注数、类别数、修改时间和数据集路径。

YAML 发现支持 `data.yaml`、`data.yml`、`dataset.yaml`、`dataset.yml` 的大小写变体，以及
包含 train/val/test 配置的自定义 `.yaml/.yml` 文件。split 目录兼容 `images/train`、
`train/images`、`valid`、`validation`，配置路径失效时会检查数据集根目录的本地布局。
LabelMe 同时支持图片与 JSON 同目录、`images + labels` 分离布局和
`train/{images,labels}` split-first 布局。

共享能力位于 `dataset_detector.py`、`dataset_discovery.py` 和 `interactive_helpers.py`。
各转换脚本复用同一套扫描、数据集根目录解析、split 路径解析、YOLO 标签目录映射、格式识别
和交互选择。

文本编码处理由 `text_encoding.py` 统一负责。读取 `data.yaml`、`classes.txt`、YOLO TXT、
LabelMe/COCO JSON 等输入时，会识别 UTF-8 BOM、UTF-16、GB18030/GBK、Big5、CP1252 等编码。
日常解析保持源文件字节不变；显式调用 `read_text_auto(path, rewrite=True)` 时才会原子规范化为
UTF-8，终端同时显示 `[编码转换]` 提示。可使用 `encoding=` 处理已知的编码。

所有向导输出目录采用统一格式：

```text
<源数据集名>__<操作名>
```

当前操作名包括：

```text
dataset_seg__to_semantic
dataset_det__merge_yolo
dataset_semantic__merge_semantic
dataset_semantic__edit_classes
dataset_seg__to_mvtec_ad
labelme_data__to_yolo
generated_data__to_yolo_seg
mvtec_data__mvtec_to_yolo
dataset_det_A__mirror_seg_subset
dataset_seg__sample_preview
dataset_seg__to_mvtec_ad_object
```

## PNG 语义 mask → X-AnyLabeling (`semantic2xany`)

生成 X-AnyLabeling 导入单通道语义 mask 所需的灰度映射 JSON：

```bash
python convert.py semantic2xany \
  --src ../aeroscapes/dataset_semantic
```

直接运行 `python convert.py semantic2xany` 会自动扫描项目 `datasets/`，展示可用的
PNG 语义分割数据集并通过编号选择。`--datasets <目录>` 可以指定交互扫描范围。

默认输出 `<数据集>/mask_grayscale_map.json`，并全量检查图片/mask 配对、尺寸、
单通道格式与像素类别。像素 ID `0` 默认作为画布背景；`--include-background` 可把它
加入导入映射。

映射以 mask 实际出现的像素 ID 为准。YAML 中缺失的 ID 自动使用 `class_<ID>`，重复
类别名自动追加 `__id_<ID>`，例如第二个 `none` 会写成 `none__id_20`。终端会逐项显示
这些自动处理。`--strict-classes` 可启用训练配置完整性检查。

在 X-AnyLabeling 中按 split 导入：

1. 打开 `images/train`（或 `images/val`、`images/test`）。
2. 选择 `导入标注` → `MASK`。
3. 映射文件选择 `mask_grayscale_map.json`。
4. mask 目录选择对应的 `masks/train`（或 `masks/val`、`masks/test`）。

导入完成后，X-AnyLabeling 会在当前图片目录生成同名标注 JSON，随后可以查看和编辑
语义区域。

## YOLO Seg → PNG 语义掩码 (`yolo2semantic`)

交互扫描并选择：

```bash
python convert.py yolo2semantic
```

直接转换：

```bash
python convert.py yolo2semantic \
  --src /path/to/dataset_seg \
  --out /path/to/dataset_semantic
```

转换结果可直接作为 `train_seg.py` 的 `DATA_YAML`：

```text
dataset_semantic/
├── images/{train,val,test}/
├── masks/{train,val,test}/*.png
├── data.yaml
├── classes.txt
└── conversion_report.json
```

默认类别映射为 `background=0`、`YOLO class 0 → semantic class 1`，其余类别依次偏移。
缺少 txt 或空 txt 的图片生成全背景 mask。实例重叠区域默认采用标签文件中的后一条标注，
`--overlap first` 和 `--overlap larger` 提供先标注优先、较大实例优先策略。图片传输支持
`--image-mode copy|hardlink|symlink`。

## 合并多个数据集 (`merge-datasets`)

```bash
python convert.py merge-datasets
python convert.py merge-datasets \
  --format semantic \
  --src /path/to/original \
  --src /path/to/generated \
  --out /path/to/merged
```

交互运行时先选择格式：

```text
1. YOLO 目标检测（bbox TXT）
2. PNG 语义分割（单通道 mask，类似 dataset_semantic）
3. YOLO 实例分割（polygon TXT）
```

工具随后为每个来源建立独立临时类别 ID，并自动展示同名、近似名称候选。用户可以
循环输入任意数量的类别操作：

```text
m 0,5 = car               合并临时类别 0、5
m 1,4 = truck             继续添加另一组合并
d 8,9                     删除类别及其全部标注行
r 3 = bus                 重命名类别
show                      查看当前完整计划
reset                     清空计划
done                      完成计划并进入一次性生成
```

`done` 之前只维护操作计划，不生成中间数据集。确认后一次性生成：

- YOLO Det/Seg：重写全部 TXT 第一列，保留 bbox 或 polygon 坐标。
- PNG Semantic：重写全部 mask 像素，删除类别写为 `255`。
- 合并结果直接写入 `images/{train,val,test}` 与对应的 `labels/` 或 `masks/`
  目录。文件名唯一时保留原名；同一 split 出现同名文件时依次追加 `__2`、
  `__3`，图片与标注同步改名。
- 最终只生成一个 `data.yaml`。`--id-policy compact` 生成连续 ID，`preserve`
  保留可保留的原 ID，`explicit` 使用计划文件中的 `explicit_ids`。

旧命令 `merge-yolo` 继续提供 YOLO Det/Polygon Seg 专用入口。

批处理模式支持以下安全与策略参数：

- `--taxonomy namespace|strict|union-by-name|mapping-file` 控制跨来源类别体系；
  `mapping-file` 同时传入 `--mapping-file <yaml>`。
- `--plan <class_plan.yaml>` 复用来源指纹绑定的完整类别计划。计划保存全部最终类别
  ID、名称与来源引用，调整 `--src` 顺序后仍保持映射稳定。
- `--id-policy compact|preserve|explicit` 控制最终类别 ID；`explicit` 从计划 YAML
  的 `explicit_ids` 读取精确映射。
- `--duplicate-policy keep|error|hash-dedupe|drop|merge-annotations` 处理同 split
  重复图片；
  `hash-dedupe` 同时校验重映射后的 TXT 或 mask，标注冲突会终止事务。
- YOLO 的 `merge-annotations` 合并重复图片标注；同类同几何自动去重，冲突通过
  `--duplicate-annotation-conflict-policy` 和
  `--duplicate-annotation-conflict-iou` 控制。
- `--split-leakage-policy warn|error|drop` 处理跨 split 图片重复，
  `--workers N` 并行计算内容摘要。
- YOLO 使用 `--empty-image-policy keep|drop|quarantine` 和
  `--orphan-policy error|drop`；Semantic 使用
  `--all-ignore-policy keep|drop|quarantine` 与可重复的
  `--source-ignore-label SOURCE=ID|none`。
- `--require-train-val` 要求最终 train/val 均可用；`--dry-run` 执行零写入分析；
  `--clean` 通过同级 staging 目录原子替换已有输出。
- `--image-mode reflink` 优先使用文件系统 CoW 克隆，并在缺少支持时复制；同一输出
  目录使用跨进程写锁。

## 统一数据集类别编辑 (`edit-classes`)

```bash
python convert.py edit-classes
python convert.py edit-classes --src /path/to/dataset --out /path/to/output
```

工具自动识别 PNG 语义分割、YOLO 目标检测和 YOLO 实例分割。检查项覆盖空名称、
`None/null/void/unlabeled`、重复名称、带来源前缀的近似名称、稀疏 ID、YAML 未定义
ID、YAML 中未使用类别和无效标注。

交互命令：

```text
d 7,8,12                 删除类别
m 1,5,9 = road           合并任意类别并指定输出名称
r 3 = paved_road         重命名类别
u ignore                 处理 YAML 未定义的标注 ID
u error                  YAML 未定义的标注 ID 作为错误
show                     查看当前计划
done                     按选定 ID 策略开始生成
```

PNG 语义分割删除类别时将对应像素写为 ignore label，默认值为 `255`。YOLO
Det/Seg 删除类别时移除对应标注行；合并与重排只重写每行首列类别 ID，坐标保持原值。
输出采用新目录，包含同步重写的 `masks/` 或 `labels/`、`data.yaml`、`classes.txt`、
`class_edit_mapping.yaml` 和 `class_analysis.json`。图片支持复制、硬链接、软链接和
CoW reflink。Semantic 同时支持多整数 `labels`/`values`、RGB/RGBA label 映射，并
统一输出单通道 PNG。
单数据集编辑同样采用 staging 事务写出，并支持 `--dry-run`、`--clean`、`--yes`
和 `--require-train-val`。`--id-policy compact|preserve|explicit` 支持连续、保留和
计划指定 ID。YOLO 的目录列表与 TXT manifest split 会按配置清单处理。
旧命令 `edit-semantic` 继续提供 PNG 语义数据集专用入口。

## 长尾诊断并挑选测试集 (`balanced-test`)

该命令面向 YOLO Detection 与 YOLO Instance Segmentation，从大型候选池挑出固定数量
测试图片。交互流程先选择 Det 或 Seg，再自动扫描对应数据集，并按最后修改时间从新到旧
排列。长尾诊断按少于 10、20、30……100 张分阶段显示类别数量与占比，然后设置任意类别
图片数门槛。候选图片可以来自 `val test`，也可以来自 `train val test`。输出统一写入
`test` split。

```bash
python convert.py balanced-test
python convert.py balanced-test \
  --source /path/to/pool/data.yaml \
  --task detect \
  --splits val test \
  --min-class-images 80 \
  --strategy representative \
  --count 200 \
  --out /path/to/final_test \
  --yes
```

`--strategy representative` 按候选池类别分布抽样，适合模型指标评估，也是默认策略。
`balanced` 拉平通过门槛的类别，`coverage` 优先覆盖这些类别。小批量导出由门槛和目标
分布共同约束，门槛边缘类别按所选目标分布获得权重。

`--rare-class-policy exclude-images` 会整图过滤含低频类别的样本，保持评估标注完整；
`keep-incidental` 允许合格类别驱动混合图片入选，并保留图片中的全部 Det 框或 Seg polygon。
工具始终保留入选图片的完整标注。输出 YAML 沿用参考类别 ID，便于直接评估已有模型。

`--previous` 是可选的图片排除集合，支持 YOLO `data.yaml`、数据集目录、普通图片目录和
单张图片。图片内容使用 SHA-256 排重，改名后的同图仍可识别。输出包含
`images/test`、`labels/test`、`selected_test.txt`、`remaining_source.txt`、
`filtered_by_class_threshold.txt`、类别映射和 JSON/Markdown 报告。报告记录 split 范围、
10–100 分阶段长尾占比、每类可用/入选图片数、门槛策略及随机种子。`--dry-run` 用于预览。
快速模式读取标签并按路径去重。`--deep-validate` 会逐张解码图片并执行完整校验；
`--deduplicate-source` 会计算候选池全部图片摘要，并按
`--duplicate-annotation-policy merge|error|keep-first` 处理同图标注。

## 汇总并均衡重划分 labels/masks 数据集 (`rebalance-splits`)

```bash
python convert.py rebalance-splits \
  --src /path/to/dataset/data.yaml \
  --ratios 0.8 0.1 0.1 \
  --out /path/to/dataset_balanced \
  --yes
```

工具汇总源数据集的 train、val、test，自动识别 YOLO Detection、YOLO Instance
Segmentation 和 PNG Semantic Segmentation。YOLO 从 TXT 读取图片类别与对象数；Semantic
从 mask 的实际像素值读取图片类别与像素数，并支持整数、RGB/RGBA mask、`labels`/`values`
映射与 `ignore_label`。图片含有多个类别时会作为整体分配；类别图片数达到有效 split 数时，
算法优先让该类别覆盖每个 split，再贴近目标比例。默认保留全部输入样本，因此600张输入
对应600张输出。显式使用 `--deduplicate` 时会通过图片内容摘要排重；YOLO 同图不同标注
会合并为一份完整 TXT，并去除完全重复的标注行。

YOLO 输出 `images/` + `labels/`，Semantic 输出 `images/` + `masks/`。结果同时包含
`data.yaml`、源文件去向 `split_mapping.json`、重复图片清单，以及 JSON/Markdown
类别分布报告。
先运行 `--dry-run` 可以预览各 split 数量和仍缺失的类别覆盖。

## 图生图返回数据 → YOLO Seg (`generated2seg`)

支持以下返回结构，路径可以放在任意位置：

```text
<输入根>/
└── <物体>/
    └── <缺陷>/
        ├── image/   # 生成图片
        │   ├── 0.png
        │   └── ...
        └── fg/      # 同 stem 的 PNG 缺陷掩码
            ├── 0.png
            └── ...
```

交互运行：

```bash
python convert.py generated2seg
```

工具会扫描 `datasets/` 中所有含同级 `image/`、`fg/` 的目录，显示配对数、问题数、
物体数和缺陷数。菜单支持序号选择，也支持输入 `p` 后填写任意路径。

直接指定路径：

```bash
python convert.py generated2seg --src /path/to/generated --out /path/to/output_seg --yes
```

转换规则：

- `<缺陷>` 目录名自动成为 YOLO 类别；多个物体下的同名缺陷共用 class ID。
- 根目录内存在有效 `data.yaml` 或 `classes.txt` 时沿用其类别顺序；其余缺陷自动追加。
- 缺少类别配置时按缺陷目录自然排序生成 `data.yaml`、`classes.txt` 和
  `class_mapping.json`。
- 所有生成样本默认写入 train；val/test 空目录会一并创建。
- 输出文件名采用 `<物体>__<缺陷>__<原stem>`，用于区分各目录重复的 `0.png`。
- fg 掩码经过二值化、白底自动反相、外轮廓提取和归一化后写成 YOLO polygon。
- `conversion_report.csv` 记录缺失配对、尺寸冲突、空掩码、轮廓数和前景比例。

常用参数：`--threshold 127`、`--min-area 1`、`--epsilon 0.001`、`--clean`。

## 标准 MVTec AD → YOLO Seg / Det (`mvtec2yolo`)

支持标准 MVTec 目录，输入路径可以是整个数据集根目录，也可以是单个 category：

```text
<MVTec根>/
└── <物体>/
    ├── train/good/*.png
    ├── test/good/*.png
    ├── test/<缺陷>/*.png
    └── ground_truth/<缺陷>/*_mask.png
```

交互扫描并选择：

```bash
python convert.py mvtec2yolo
```

在 `convert.py` 主菜单选择 `mvtec2yolo` 后，参数处直接回车即可扫描仓库 `datasets/`。
工具会列出每个候选的图片数、掩码数、类别数和路径，再提示选择输入与输出目录。

同时生成 YOLO Seg 和 Det：

```bash
python convert.py mvtec2yolo \
  --src /path/to/mvtec \
  --out /path/to/mvtec_yolo
```

输出结构：

```text
mvtec_yolo/
├── dataset_seg/{images,labels}/{train,val,test}/
├── dataset_det/{images,labels}/{train,val,test}/
├── dataset_seg/data.yaml
├── dataset_det/data.yaml
└── conversion_report.csv
```

只生成一种任务：

```bash
python convert.py mvtec2yolo --src /path/to/mvtec --out /path/to/dataset_seg --task segment
python convert.py mvtec2yolo --src /path/to/mvtec --out /path/to/dataset_det --task detect
```

转换规则：

- 默认 `--class-mode defect`，以 `<缺陷>` 目录名作为 YOLO 类别。
- `--class-mode object` 以物体名作为类别，`object-defect` 生成“物体+缺陷”类别。
- 默认 `--split-mode all-train`，让多模态生成的异常样本直接进入监督训练集。
- `--split-mode preserve` 保留 MVTec 的 train/test 归属。
- good 图片生成空 txt，作为 YOLO 负样本。
- 每个 mask 连通区域生成一个 Seg polygon 和一个 Det bbox。
- 缺失掩码、尺寸冲突和空掩码写入 `conversion_report.csv`。

## 更多(单步 / 特殊输入 / 辅助，`convert.py more` 展开)

| 命令 | 说明 |
|---|---|
| `coco2sync` | COCO 2017 → train/val/test(特殊输入，之后接 `oneclick`) |
| `sync` | 只做整理(= `oneclick` 第 1 步) |
| `labelme2yolo` | 只做 LabelMe → YOLO(= `oneclick` 第 2 步) |
| `seg2mvtec` | 只转单个 seg 数据集(= `to-mvtec` 的非交互内核) |
| `make-sample` | 造示例 seg 数据；支持 `--out`，已有输出使用 `--force` 原子替换 |
| `inspect-labelme` | 检查 LabelMe 标注内容(排查用) |

> `oneclick` 内部已 import 并串起 `sync` + `labelme2yolo`；`to-mvtec` 内部已包 `seg2mvtec`。
> 所以"更多"里的多是它们的内部分步——保留成独立文件供单独调用，日常不用直接碰。
> 参数会原样转发给 `convert_tools/` 下对应工具，`convert.py` 只负责分发。

`oneclick --task all --seg-type semantic` 会同时生成 `dataset_semantic`、
`dataset_det` 和 `dataset_cls`；各组件先在隔离目录完成，再一次性发布。

辅助命令 `labelme2yolo`、`seg2mvtec`、`sample-browse`、`make-sample` 同样支持
`--dry-run`。`labelme2yolo` 预览输入格式与类别映射，`seg2mvtec` 检查图片与多边形并
统计转换数量，`sample-browse` 检查配对与标签，`make-sample` 显示示例生成计划。
上述预览保持输出目录、锁文件及源数据零写入。`labelme2yolo --clean` 可显式原子替换旧输出。

## ③ 转 MVTec AD 的约定(交付多模态团队用)

- **结构 = 方案 B**：每个缺陷类 → 一个顶层 MVTec category(20+ 类就 20+ 个文件夹)。
- **good = 相对定义**：对类别 C，所有不含 C 的图算作 C 的 "good"
  (源 train → `train/good/`，源 val/test → `test/good/`)。
- **一图多类**：复制进每个相关 category，各自只保留本类多边形掩码。
- **无标注样本**：图片目录为样本清单，缺少 TXT 或空 TXT 都按 good 处理。
- **目录名**：`good` 专供正常样本；缺陷名与其冲突时自动追加序号（如 `good__2`）。
  重复名称、大小写及 Unicode 规范化后重名的缺陷也会分配独立目录，物体版采用相同规则。
- **掩码**：二值 PNG `{0,255}`，与原图同尺寸，命名 `<stem>_mask.png`。
- **内容识别**：扫描器依据 YOLO polygon 标签内容选出数据集，目录名可以自定义；
  det 的 4 坐标 bbox 会标记为 `DET-skip`。
- 输出写到同级新目录：`dataset_seg → dataset_seg__to_mvtec_ad`。

## 物体版 MVTec AD(推荐,category=物体)

适用:多模态/零样本异常检测。category 是**物体**,物体内按缺陷分子文件夹。

推荐用交互向导一站式走完:

    python convert.py obj-wizard

- **第1步(生成标注预览)**:输入源 seg 数据集 + 输出目录,生成每张图的**标注预览**
  (左=原图画上缺陷多边形+中文类别名,右=信息栏:文件名、缺陷种类数、逐类计数、多边形总数)
  和 `sample_index.csv`(含 `n_defect_classes` 列,可先筛多类别图)。
- **人工分图**:把预览下载到本地,逐张看图判断属于哪个物体,建**中文物体名**文件夹,把
  **预览图直接**分进去即可(一张图只放一个物体;多物体/说不清的先别放),再把 `staging/`
  上传回服务器。脚本**只认文件名**——图和标注都从源数据集按名字取原图,预览图的合成内容
  不会进 MVTec,所以放预览图/占位文件都行,**别改文件名**就好。
- **第2步(转换)**:再跑一次向导选第2步,按文件名回源数据查原图+标注、生成物体版 MVTec。

也可跳过向导直接用单命令:`python convert.py sample-browse`(生成预览)、
`python convert.py to-mvtec-obj`(转换)。中文字体默认用仓库自带 `tool_lib/msyh.ttc`,
可用 `--font` 覆盖。

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
| `objseg_wizard.py` | 物体版向导(交互)：分两步生成预览+转换；对应 `convert.py obj-wizard` |
| `seg_sample_browse.py` | (可选)抽样摊图，帮人归纳源数据有哪些物体；对应 `convert.py sample-browse` |
| `make_sample_yoloseg.py` | ③ 造示例数据 |
| `one_click_convert.py` / `LabelMeToYOLO.py` | ② 转 YOLO 入口 |
| `sync_picture.py` / `coco_to_synced.py` | ① 采集整理入口 |
| `mirror_det_subset_to_seg.py` | ④ det 子集 → 对应 seg 子集 |
| `generated2seg_interactive.py` | 扫描/选择图生图返回目录并转 YOLO Seg |
| `generated_mask_to_yoloseg.py` | image + fg 掩码转换核心，也支持独立 CLI |
| `mvtec_to_yolo.py` | 标准 MVTec AD → YOLO Seg/Det |
| `inspect_labelme_shapes.py` | 辅助：检查 LabelMe |
