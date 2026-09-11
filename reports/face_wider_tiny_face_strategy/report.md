# 极小脸难以提升的根因与行动方案：WIDER FACE 派生数据 + DINOv3 ViT-S/16 LTDETR

调研日期：2026-09-10。承接 [face_wider_combined_audit](../face_wider_combined_audit/report_source.md)。
本报告回答三个问题：数据是不是太差；"针对小脸增强反而降低小脸比重"是否构成负优化；以及这套数据 + 模型上真正该做什么。
所有本地数字来自只读统计，脚本为同目录 [ceiling_audit.py](ceiling_audit.py)，输出 [ceiling_audit.json](ceiling_audit.json)。

> **本目录文档导航**：[worklog_2026-09-10.md](worklog_2026-09-10.md)（当日工作记录与交接起点）· 本文（根因分析）· [action_plan.md](action_plan.md)（优先级与行动清单）· [conversion_audit.md](conversion_audit.md)（数据转换与划分审计）· [band_eval.py](band_eval.py)（分档评测工具）

---

## 0. 结论摘要

1. **不是数据太差。** 能查的标注质量检查全部通过：tile 几何重建 0 差异、0 个越界框、退化框占比 0.1%、val/test 与原始版本逐字节一致。数据集真正的硬伤在**划分方式**，不在标注。
2. **本地评测不是官方 WIDER 协议。** 本地三分是官方 train+val 合并后按图随机 80/10/10（seed 42）；61 个事件在三份里 100% 重叠；test 的 1,615 张图 **100%** 与 train 共享连拍子场景。官方 test（16,097 张）完全不在本地。任何与公开榜单的数字对比都不成立。
3. **当前 mAP 低，主因是分辨率，不是增强。** 在 640 输入、patch=16 下，**test 有 72.1% 的 GT 框小于一个 ViT token（<16 设备像素）**；另有 **15.0% 的 GT 高度 ≤10px，属于标注约定里标为 ignore 的难以辨认框**。两者叠加（排除 ≤10px 且 ≥1 patch）后，**只有 27.93% 的 test GT 在原理上可被这个模型检出**。剩下一大半不是"模型没学好"，是"模型看不到"。
4. **负优化成立，但准确的表述不是"增强小脸错了"，而是"三个操作把训练分布整体推离了评测分布"。** 切片把脸放大、关闭 RandomZoomOut 删掉了唯一会制造更小人脸的增强、Copy-Paste 只补 12–28px 的"看得清的小脸"。净效果：每图采样得到的 <8px 框期望从 4.91 降到 3.53（−28%），密集图（>100 脸）采样概率从 1.82% 降到 1.13%（−38%）。而 test 分布没变。
5. **关键认知：数据增强改变的是"模型练习什么"，改变不了"模型能看清什么"。** 在 72% 的 GT 小于一个 token 的情况下，任何数据侧调整都无法把 mAP 拉起来；能拉动它的是输入分辨率、特征层级和输出容量。
6. **最高杠杆的动作顺序：先把测量修对（分尺寸档召回 + 可达子集指标 + 多 seed），再恢复训练分布（过采样密集/极小脸图 + 修正切片边界），然后是分辨率（640 → 800 → 1024，且必须同时改多尺度列表，否则等于没改），最后才是结构改动。** 详见第 6、7 节。
7. 一条**反直觉但已核实**的更正：DETR 系损失按 batch 内框总数归一化，不是按图归一化。所以"一张图 300 张脸会让每张脸学得更少"**不成立**。密集图的真正代价在**输出容量**：模型最多 300 个 query，评测每图最多 100 框，test 有 23.13% 的 GT 因此永远无法被命中。

---

## 1. 先把两个前提摆正

上一轮讨论"为什么掉了 0.94 个百分点"，但有两个前提会让这个数字的含义被高估。它们不改变"combined 略低于原模型"的结论，但决定了这个结论该被赋予多少权重。

### 1.1 本地 test 不是官方 test，而是官方 train+val 的随机 10%

证据链：`datasets/face_detect/face_yolo/split_manifest.csv` 与 `reports/face_dataset_research/artifact.json` 记录了 `convert.py oneclick` 用固定种子 42 对 22,094 张图全局洗牌、按 80/10/10 划分。映射到 `face_yolo_wider` 后：

| 本地 split | 来自官方 WIDER_train | 来自官方 WIDER_val | 合计 |
|---|---:|---:|---:|
| train | 10,318 | 2,565 | 12,883 |
| val | 1,271 | 325 | 1,596 |
| test | 1,283 | 332 | 1,615 |
| 合计 | 12,872 | 3,222 | 16,094 |

三点直接推论：

- **官方 test 的 16,097 张图（标签不公开）完全不在本地。** 本地"test"是随机留出的 10%，其中 1,283 张来自官方 train。
- **事件级重叠 100%。** 61 个事件在 train/val/test 三份中全部出现。把文件名末尾连拍序号去掉后，train↔test 共享 162 个子场景前缀，**test 的 1,615 张图 100% 落入与 train 共享的连拍序列**（train 侧 99.3%）。也就是说每张测试图在训练集里都有同场景的相邻帧。
- 后果：这个 benchmark 度量的是"认出几乎见过的场景"，不是泛化。绝对指标会被抬高，而真实改进可能被掩盖——**两组模型共享这个偏差，所以 A/B 比较仍然可用；但任何"离 SOTA 还差多少"的判断都必须先降级。**

另有一处文件级泄漏（此前审计漏检）：`wider_train_41_Swimming_Swimming_41_184.jpg` 与 test 的 `wider_train_41_Swimming_Swimmer_41_745.jpg` 内容逐字节相同，还有一对跨事件的（Car_Accident ↔ Ice_Skating）。量级只有 2 张图，可忽略，但说明"只比文件名"的检查不够。

**一个有用的副产品**：本地每一张图都保留了官方来源前缀（`wider_train_` / `wider_val_`，无例外）。所以**官方 val 的 3,222 张图其实全都在本地**（2,565 张混在 train、325 张在 val、332 张在 test），按前缀就能重组成一份官方 val 评测集——这是建立对外可比指标的现成材料，做法见 [action_plan.md](action_plan.md) §1.3。

### 1.2 官方属性已丢失，无法用官方口径复核

WIDER 官方的 blur / occlusion / expression / pose / **invalid** 属性全部不在本地标签里（标签是纯 YOLO 5 列）。承载属性的 LabelMe JSON 源目录 `datasets/face/WIDER/` 已删除，官方 `wider_face_val_bbx_gt.txt` 全盘不存在。这导致两件事：

- **属性无法复核**，需要重新下载官方标注包才能核对 blur/occlusion 等字段。
- **本地把官方约定里会被忽略的框当成必须检出的 GT。** 官方约定对高度 ≤10px、低分辨率难以辨认的人脸打 `Ignore`/`invalid` 标记；本地 test 里这类框有 **3,423 个，占 15.0%**。

> **更正（2026-09-10，写 [action_plan.md](action_plan.md) 时核对）**：本节初稿写的"easy/medium/hard 无法重建"**不准确**。官方随标注包提供 `wider_easy_val.mat` / `wider_medium_val.mat` / `wider_hard_val.mat`，每个只含一个 `gt_list` 字段，给出每张图里属于该难度的人脸索引；官方公开的 val 评测脚本**只读这些难度列表来决定哪些脸参与考核**（不读属性、也不做尺寸过滤）。因此三档 AP 是**可复现**的，与属性是否丢失无关。同时"≤10px 被官方忽略"也要讲准：那是标注约定，而公开脚本里的 ignore 由难度列表驱动——跟官方 easy/medium 比时那些脸确实不考核，跟官方 hard 比时它们本来就参与考核，不应把它当成固定的 15% 折扣。详见 [action_plan.md](action_plan.md) §1.1。

---

## 2. 当前指标低的真实原因：72% 的 GT 小于一个 token

这是本次调研最重要的一组数字，也是此前所有讨论都缺的一块。

模型侧事实（全部来自代码核对）：

| 项 | 实际值 | 证据 |
|---|---|---|
| backbone patch size | 16（ViT-S/16） | `src/lightly_train/_models/dinov3/dinov3_src/models/vision_transformer.py:383`；`dinov3_vit_wrapper.py:137` 从模型读取 |
| 640 输入的 patch 网格 | **40×40** | `dinov3_vit_wrapper.py:191` |
| 检测特征层级 | stride 8 / 16 / 32，三层，**无 stride 4** | `task_model.py:161-173` `feat_strides=[8,16,32]`；`hybrid_encoder.py:210-240` |
| stride-8 特征怎么来的 | **stride-16 特征双线性上采样**（不是新信息） | `dinov3_vit_wrapper.py:202-219` |
| 被丢弃的高分辨率特征 | STA 分支算了 **1/4** 特征 `c1` 但 `return c2, c3, c4`，**1/4 直接扔掉** | `dinov3_vit_wrapper.py:53-117` |
| decoder 最小先验框边长 | `grid_size=0.05`、`wh = 0.05*2^lvl` → 归一化 0.05/0.10/0.20 = **32 / 64 / 128 px** | `rtdetrv2_decoder.py:601,616` |
| 输出 query 数 / top-k | **300 / 300** | `task_model.py:212,312` |
| 训练损失归一化 | **按整个 batch 的框总数** all-reduce 后除以 world_size | `rtdetrv2_criterion.py:197-204` |

也就是说：640 输入下模型只有 40×40 = 1,600 个 token 位置可用，最小先验框是 32px，而 stride-8 那层是把 stride-16 的图放大的。

数据侧对应的数字（口径见下方说明）：

| 设备像素短边 | train（原始） | combined train | test |
|---|---:|---:|---:|
| **< 8px（半个 token）** | 63,246（40.5%） | 73,468（24.6%） | **9,335（40.8%）** |
| 8–16px | 45,741（29.3%） | 95,079（31.9%） | 7,144（31.2%） |
| 16–32px | 28,226（18.1%） | 83,759（28.1%） | 4,088（17.9%） |
| ≥ 32px | 19,006（12.2%） | 46,258（15.5%） | 2,299（10.1%） |
| **< 16px 合计（不足一个 patch）** | **69.8%** | **56.5%** | **72.1%** |

> **口径声明（重要，对外引用务必带上）**：`设备像素` = `min(bw*640/W, bh*640/H)`。训练（`scale_jitter` 用 albumentations `Resize` 到 (S,S)，`scale_jitter.py:100-103`）与整图推理（`transforms_functional.resize(x, self.image_size)`，`task_model.py:685`）都是**方形拉伸**到 640，因此这个口径正是模型实际看到的框尺寸。另一种"保比缩放"口径（`min(bw,bh)*640/min(W,H)`）描述的是原图的真实比例，不是模型输入，两者差别很大（保比口径下 <8px 只有 24.0%），**不可混用**。

把分辨率下限、官方忽略规则、评测容量三者叠加，得到"原理上可检 GT"的比例（`ceiling_audit.json`）：

| 项 | train | val | **test** |
|---|---:|---:|---:|
| GT 总数 | 156,219 | 16,995 | 22,866 |
| 位于 >100 张脸的图片中的 GT | 56,839（36.4%） | 4,419（26.0%） | **9,090（39.8%）** |
| 超出 maxDets=100 容量（永远无法命中） | 33,339（21.34%） | 2,219（13.06%） | **5,290（23.13%）** |
| 超出 300 query 预算 | 10,664（6.83%） | 374（2.20%） | **1,415（6.19%）** |
| 小于一个 patch（<16 设备像素） | 69.8% | 66.8% | **72.07%** |
| 官方约定标为 ignore（原生高度 ≤10px） | 18.10% | 13.38% | **14.97%** |
| **≥1 patch 且非 ≤10px（可检 GT）** | 47,232（30.23%） | 5,639（33.18%） | **6,387（27.93%）** |
| 非 ≤10px 但不足一个 patch | 80,715（51.67%） | 9,082（53.44%） | **13,056（57.10%）** |

> 口径说明：这里的"≤10px"用的是**标注约定的 ignore 标记**（高度 ≤10px 视为难以辨认），不是官方难度列表。官方公开的 val 评测由难度列表决定哪些脸参与考核，两者不必一致；`band_eval.py` 输出的 `official_valid` 子集沿用了这个"排除 ≤10px"的定义。这个子集的意义是"原理上可达"——同时满足"人眼可辨认"和"至少占一个 token"。

其中 test 最难的格子里情况更极端：**38 张密集图（GT>100）贡献 9,090 个 GT（占 test 39.8%），而这 9,090 个里有 8,781 个（96.6%）不足一个 patch。** 也就是说，指标权重最大的一块 GT，恰好是分辨率上最不可达的一块。这两件事在 combined 里同时向坏的方向走：切片把密集图从训练里拆掉了（tile 子集 GT>100 的图片数为 0），而被拆掉的正是测试时权重最大的场景。

由此可以解释两件此前看着奇怪的事：

- **为什么绝对指标这么低（mAP 0.28 / AP50 0.54），却和公开榜单的 0.9 差距巨大？** 因为本地测的是一个严格更难的题（保留 ≤10px 的框、用 IoU 0.5–0.95 平均的 COCO 口径、冻结的 ViT-S/16、640 输入），公开数字用的是官方口径（忽略 ≤10px）+ 专用人脸检测器 + 1024–1650px 输入。**两边不可比，0.28 不代表数据坏了。**
- **为什么两个模型只差约 1 个百分点？** 因为 72% 的 GT 贴着分辨率地板，两组都在同一条墙前面。任何数据侧调整能影响的只有那 28% 的一部分，天花板很低。**在这个口径下继续做增强消融，信噪比极差。**

---

## 3. "数据太差了吗"——分性质回答

数据不是"太差"，但有四类问题，性质完全不同，处理方式也不同。

### 3.1 标注质量：能查的都过了，不是问题

| 检查 | 结果 |
|---|---|
| tile 几何重建（6,442 张，与生成器相同浮点顺序） | 标签差异 **0** |
| 越界框 | **0**（转换期已裁剪到边界） |
| 退化框（短边 <2px 原生像素） | train 160 / val 15 / test 20（占比 0.1%） |
| 同图内重复框（IoU>0.9） | train 14 对 / val 0 / test 6 对 |
| 非有限值、非正宽高 | **0**（全量 298,564 框） |
| val/test 与原始版本一致性 | 图片与标签**逐字节相同**（SHA-256） |
| manifest 来源合法性 | 全部 donor/recipient/tile 源图均属本地 train，0 跨划分引用 |

唯一的确定缺陷是切片边界（见 3.4），而它是**生成规则**问题，不是原始标注问题。

值得知道的背景：WIDER FACE 自身的标注也不是零噪声——BDC（IET Computer Vision 2022）明确指出 train 集存在框错位并发布了修正版标注；一篇改进版 RetinaFace 论文直述 WIDER FACE "there are also cases of missing labels, mislabeling"。但这些属于数据集固有噪声底，**不解释本次两组之间的差异**。

### 3.2 划分方式：这是最实质的问题，且性质是"不可比"

见 1.1。它不影响 A/B 的内部效度，但让绝对值失去参照，也让"泛化能力"这个指标名不副实。

### 3.3 评测设计：容量封顶，且口径与官方不一致

- `maxDets=[1,10,100]`（torchmetrics 默认，`src/lightly_train/_metrics/mean_average_precision.py:85-91`）导致 test **23.13% 的 GT 永远无法被命中**。
- 模型自身只输出 300 个框（`num_top_queries=300`），在 809 脸的图上召回上限就是 37%，在 >300 脸的 9 张图（4,115 个 GT，占 test 18.0%）上被 query 预算二次截断。
- **两条推理路径的分数阈值口径不同**：不传 `--sahi` 时 mAP 用 `threshold=0.0`（`det_infer.py:690,703`），传 `--sahi` 时同一份预测会先被 0.3 阈值砍掉再算 mAP（`det_infer.py:706`、`task_model.py:794-812`）。所以"combined 全图 0.27175"和"combined+SAHI 0.27406"严格说不是同一打分口径；只有"原模型 SAHI vs combined SAHI"这一组是同口径可比的。

### 3.4 派生数据（combined）：两处确定缺陷

| 缺陷 | 量化 | 性质 |
|---|---|---|
| 切片边界丢弃可见人脸标签 | 6,442 张 tile 中 2,405 张受影响（37.3%）；3,915 个可见框被 <50% 面积规则删除；其中 1,241 个是"保留 ≥25% 面积且可见区域 ≥8px"的重点候选；已人工确认存在可辨认半张脸被删标签的实例（`tile_006379`） | 监督信号错误，确定要修 |
| 极小脸与密集图曝光下降 | <8px 框占比 40.5% → 24.6%；每图采样的 <8px 框期望 4.91 → 3.53（**−28%**）；密集图（>100 脸）采样概率 1.82% → 1.13%（**−38%**）；密集 GT 占比 21.3% → 11.2% | 分布偏移，方向与目标相反 |

生成器规则在 `datasets/convert_datasets/convert_tools/face_wider_prepare.py:408`（`if ratio < 0.50: continue`）与 `:422`（接受条件）。

> **更正（2026-09-10，代码审计后）**：本节初稿把 `tiles_v1` 与 `combined_v1` 的 tile 框总数差 514（134,941 vs 135,455）归因为"生成代码在两次运行之间改动过"，**这个归因是错的**。真正原因是 `rng = random.Random(args.seed)` 只建一次（`face_wider_prepare.py:581`），并先传给 `generate_copy_paste`（`:589`）再传给 `generate_tiles`（`:605`）；`combined` 模式下 Copy-Paste 先消耗随机流，导致 tile 阶段拿到的是错位后的随机数，于是选中的源图集合不同（3,504 vs 3,454，交集 2,576）、框数不同。证据：两个数据集的 copy-paste 记录**逐字节相同**（pastes 均 2,931）、tile 记录数都是 6,442、几何重建失配为 0——若代码改过，三者不会同时成立。结论是 `--seed` 只在"同 mode"下才是复现键。

---

## 4. "针对小脸增强却让小脸比重下降，这是负优化吗"

**是。但准确的因果表述应该是：想法没错，执行把一个"重加权"问题做成了"整体平移"问题。**

### 4.1 三个操作的方向，逐个核对

| 操作 | 方向 | 机理 |
|---|---|---|
| 切片放大 | 训练人脸上移 | 生成器要求 `target_short >= 12` 且 `>= full_short * 1.25`（`face_wider_prepare.py:422`），即至少放大 1.25×；实际 tile 的 <8px 占比只有 7.55% |
| 关闭 RandomZoomOut | **删掉了唯一制造更小人脸的增强** | 已核实语义方向：该增强只做 padding、把画布放大 1–4 倍（`side_range=(1.0,4.0)`，`random_zoom_out.py:111-150`），随后 ScaleJitter 再缩到 480–800，**目标在最终输入里变小**。默认 `prob=0.5`；传 `None` 时不是 prob=0，而是**根本不构造该 transform**（`object_detection_transform.py:169-176`） |
| 保留在线 RandomIoUCrop | 双向、偏上移 | 裁剪 0.3–1.0 倍、要求裁剪中心含至少一个框（`random_iou_crop.py:120-154`），最多放大约 3.3×，同时**直接丢弃中心落在裁剪外的框** |
| Copy-Paste | 补充"看得清的小脸" | 目标尺度 12–28px 线性均匀采样（`face_wider_prepare.py`），1,466 张图只净增 2,931 张脸（占 combined 的 0.98%） |

四个操作里，三个把训练分布往"更大的脸"推，唯一往"更小的脸"推的那个被关掉了。方向判断没有歧义。

### 4.2 为什么"比重下降"本身不能单独定罪，但这次成立

"比重下降"不是自动等于负优化——如果放大让模型学到可迁移的外观特征，而极小脸靠原始样本仍然练够了，净效果可以是正的。判断要看三条：**（a）极小脸的绝对练习量是否还够；（b）训练分布是否仍然匹配测试分布；（c）放大带来的外观学习能否迁移回原始尺度。**

这次三条全不满足：(a) 每图采样的极小脸期望 −28%；(b) 测试分布一点没变，仍以极小脸为主（设备像素 <8px 占 40.8%、<16px 占 72.1%）；(c) 没有证据支持迁移——模型学到的是"15px 的脸长什么样"，而测试要它认的是 6px 的脸，**token 数量都不一样，特征表达不是同一件事**。

而且有个更根本的问题：**放大切片提升的是"训练时这张脸有多少 token"，不是"推理时这张脸有多少 token"。** 推理固定 640 输入、patch 固定 16，测试图的 6px 脸永远只有 0.4 个 token。所以这条路线在原理上就无法把收益送达目标。**这正是"负优化"最本质的表述：它优化了训练难度，没有优化推理能力。**

### 4.3 上一轮解释中需要更正的一条

上一轮说"一张图有 300 张脸时，小脸框很多但只集中在少数训练机会里，所以每张脸学得少"。**这条不成立。** 回归到代码：损失分母是 `sum(len(t["labels"]) for t in targets)`，即**整个 batch 的框总数**（`rtdetrv2_criterion.py:197-204`），不是单图框数、也不是 query 数。在 batch 内，无论一张图有 3 张脸还是 300 张脸，**每张脸分到的权重都是 1/框总数**。密集图不会被稀释。

密集图的真正代价在**容量**，不在梯度：

- 每张图最多输出 300 个框（query top-k），809 脸的图召回上限 37%；
- 评测每图最多计入 100 框，超出部分 5,290 个 GT（test 的 23.13%）永远无法命中；
- 而 >100 脸的图片承载了 test **39.8% 的 GT**——也就是说指标的一大块被这些结构性封顶的图主导。
- 训练侧则相反：combined 把密集图从训练里"拆掉了"——**tile 子集中 GT>100 的图片数为 0**，密集全图的采样概率降到 1.13%。模型在测试时最需要的那一类场景，训练时见得最少。

所以"数据集里那些有特别多小脸的图会不会影响结果"——**会，而且是一阶影响**，但机制是"输出容量 + 评测容量 + 训练曝光"三者，不是"梯度被摊薄"。

---

## 5. 由此得到的解法：按杠杆从高到低

### T0 先修测量（不做这步，后面所有实验都读不出信号）

现状问题：单一 mAP、单一 seed、两路径阈值口径不同、72% 的 GT 贴地板。改进：

1. **锁定一个尺寸口径**（建议"设备像素短边"，即模型输入中的框短边），在报告、脚本、评审口径中统一；不要再混用保比口径。
2. **分档报告召回**，而不是只看 AP_small：`<8 / 8–16 / 16–32 / ≥32 px` 设备像素，以及每个档的召回与 AP50。AP_small 在当前数据上几乎没有信息量——COCO small 的阈值是面积 <32² 设备像素，而本地 90% 的框都在这个区间里，等于把"墙内"和"墙外"混在一锅。
3. **增加可达子集指标**：排除 ≤10px 难以辨认框、且 ≥1 patch（≥16 设备像素）的 GT 上的 AP50/召回。参考值：**test 该子集只有 6,387 个框（27.93%）**。这是唯一能真实反映模型能力的指标。
4. **密集子集单独报告**：GT>100 的 38 张图，以及 maxDets=300 的诊断结果。当前 100 框上限砍掉了 23.13% 的 GT。
5. **SAHI 与全图路径统一到 threshold=0.0 计算 mAP**，部署阈值只用于可视化/JSON。否则两组数字不可比。
6. **多 seed**：两次实验差 0.94pp，而单 seed 的训练波动在这个尺度上通常同量级。**任何小于 2pp 的差异，先用 3 个 seed 确认再下结论。**
7. **恢复官方参照**：重新下载官方 WIDER 标注包（只为了 val 的属性与 `wider_face_val_bbx_gt.txt`），在官方 val 上按官方口径（忽略 ≤10px、AP@0.5）算一次，作为外部锚点。本地未使用官方 test，这一锚点不可替代。
8. **把本地 test 里的 1,283 张官方 train 图挪回 train，或至少单独报告**；理想做法是从官方 val 里重新划一份干净 val/test。

### T1 恢复训练分布（低成本，方向明确，先做）

9. **重新打开 RandomZoomOut，但不要照抄默认值。** 方向正确（它是唯一会把脸变小的增强），但它 `side_range` 上限 4.0 会把脸压到 1/4，本就贴地板的训练集会被进一步压到地板上。建议做 `side_range=(1.0,2.0) × prob∈{0.3,0.5}` 的小扫描，而不是二选一。
10. **对密集图与极小脸图做文件级过采样**（这是本轮最关键的改动）。不必改上游采样器：在数据准备阶段把"GT>100"和"含大量 <8px 框"的原图在 train 列表里复制若干份即可，效果等同 PyramidBox 的 data-anchor-sampling 思路（通过缩放+采样把尺度分布拉回目标分布），但实现成本极低。**验收标准：把每图采样的 <8px 框期望恢复到 ≥4.91（原始水平），密集图采样概率恢复到 ≥1.82%。**
11. **修正切片边界规则**并重新生成 v2：对"仍有明显可辨认人脸、且原框保留面积 ≥25%"的窗口（1,241 个重点候选）**重新选窗口**，而不是删除标签或提高面积阈值。生成器改动落在 `face_wider_prepare.py:408` 与 `:422`。
12. **降低切片混合比例，或只保留"含小脸"的切片。** 当前 tile 的 <8px 占比只有 7.55%，属于"把难样本变简单的样本"，混入 31% 与目标相反。首轮建议 10–15%，并把比例作为扫描变量。文献侧佐证（可信度中低）：人脸 copy-paste 的合成比例研究报 30–50% 最优、>70% 饱和甚至回落。
13. **Copy-Paste 的目标尺度改为对齐 test 分布**，而不是固定 12–28px；或者先整支关掉，把变量数降到最低。

### T2 分辨率：最大的单一杠杆（但有一个代码陷阱）

14. **必须在改 image_size 的同时改多尺度列表。** 已核实：训练时的实际分辨率来自 `scale_jitter.sizes` 这个**硬编码列表**（`transforms.py:71-93`，13 个尺寸，640 出现 3 次，**列表均值恰好是 640.0**），`image_size` 在训练侧只用于解析 `auto` 字段、构造模型和给 decoder 设 `eval_spatial_size`。**因此直接把 `IMAGE_SIZE=800` 只会改变评测分辨率与 anchor 缓冲，训练输入仍以 ≤640 为主（61.5% 的采样 ≤640）——那等于没做这个实验。** 正确做法是同时通过 `transform_args` 传入以目标分辨率为中心的 `scale_jitter.sizes`（例如 800 档用 640–1024）。
15. **阶梯验证 640 → 800 → 1024**，每档同时提升训练与评测。预期量级（文献）：人脸检测 640→1024 在 WIDER hard 上约 **+10 AP**（SCRFD 论文表格）；通用小目标 640→1280 约 **+8 AP**（多篇一致）。代价是显存/算力约 4×（8GB 卡上 batch 需从 32 降到约 8，配合梯度累积）。
16. **修正 SAHI 的实现，让切片真正放大。** 这是本次发现的一个具体缺陷：切片尺寸被硬编码为模型 `image_size`（`det_infer.py:714`、`task_model.py:751`），切片**不做任何上采样**，只回收了整图路径因方形拉伸丢掉的像素。实测短边放大倍数中位数只有 **1.18×**，四分位上限 1.60×，而 **12.8% 的测试图放大倍数正好是 1.00×**（`min(W,H)<640` 时整图先被放大到容纳一个切片）。这解释了为什么 SAHI 只带来 +0.23pp。标准 SAHI 的做法是切 512 再放大到 640（1.25×）或切 320 放大 2×。**代码里根本不存在 `slice_size` 参数**（全仓库 grep 无命中），与截图里的 `sahi_slice_size=null` 对应。**建议：新增 `slice_size < image_size` 并把每个切片 resize 到 `image_size`，先扫 320/448/512。** 这是低成本、可立刻验证的一条。

### T3 结构改动（直面根因，需要改 fork，成本高）

17. **把被丢弃的 1/4 特征接回去。** `dinov3_vit_wrapper.py:53-117` 里的 STA 分支已经算出 stride-4 特征 `c1`，然后 `return c2, c3, c4` 把它丢了。接入为第 4 层并配一个 16px 级先验（当前最小先验是 32px），是成本相对可控、针对"最小可检尺度"的直接改动。同类做法有 ViTDet 的 simple feature pyramid（从 stride-16 反卷积出 {1/32,1/16,1/8,1/4}，作者称无需 FPN 横向连接）。
18. **输出容量与 query 预算。** 300 query 在密集图上封顶召回。工程上有两条路：直接把 `num_queries`/`num_top_queries` 提到 900（代价是显存与推理时间），或参照 DQ-DETR 做密度自适应 query。**这是唯一能解开 test 那 39.8% 密集 GT 的改动。**
19. **标签分配与损失针对微小框。** 现状是归一化坐标上的 L1 代价 + GIoU，且 GIoU 权重 2.0、L1 权重 5.0，**没有任何尺度感知项**（`matcher.py:102-129`、`train_model.py:81-86`）。对 6px 的框，归一化 L1 天然趋近 0、IoU 对 1px 偏移极度敏感，匹配几乎由分类代价主导。可考虑 NWD（Normalized Wasserstein Distance，把框建模为高斯，论文报告 AI-TOD +6.7 AP）与 scale-aware loss 权重。注意 RFLA 的作者明确说它不适合多尺度混合场景，**NWD 是更稳妥的选择**。
20. **解冻 backbone。** DINOv3 官方动机之一就是解决长训练的 dense feature collapse，冻结特征在密集预测上本就是受支持但受限的方案；ViTDet 显示 plain ViT 能做强检测，但前提是 **fine-tune + 1024 分辨率**。冻结主干是当前系统的根本限制之一，代价也最大（显存、时间、需要重跑基线）。放在 T2/T3 之后。
21. **LR 单变量。** 冻结主干下头部 lr 只有 5e-5×√2 ≈ 7.07e-5。曲线已进平台期说明"步数不够"不成立，但"步长太小"没有被排除。这是最便宜的待测项之一。

### T4 目标定义（先想清楚要什么）

22. **明确业务上的最小可检尺度，并把低于它的框从 KPI 里剔除。** 官方 WIDER 也是这么做的（≤10px 标 ignore）。如果业务只关心"输入中 ≥16px 的脸"，那当前 test 的相关 GT 只有 6,387 个（27.93%），**优化目标应该建立在这 27.93% 上**，而不是被另外 72% 稀释。
23. **考虑级联推理（内容自适应的 SAHI）**：640 全图找候选 → 裁剪候选区域 → 高分辨率复检。它与"提高训练分辨率"互补，且不要求重训，可与 T2 并行验证。

---

## 6. 推荐的执行顺序与判据

**阶段一：修测量（1 周内可完成，无需重训）**
做 T0 的第 1–7 条。产出：一份分尺寸档、分密集度、含可达子集的评测报告，覆盖现有两个 checkpoint。**判据：能在现有权重上把"墙内"和"墙外"的表现在同一张表里分开呈现。** 这一步之后才知道 0.94pp 的差距落在哪个尺寸档。

**阶段二：2×2 数据/增强消融（各 20,000 step，seed 42，统一 batch 32）**
四组：A 原图＋ZoomOut 开（默认值）；B 原图＋ZoomOut 关（= 上一轮建议的第 1 条对照）；C combined v1＋ZoomOut 开；D combined v1＋ZoomOut 关。
**判据：用阶段一的指标读，看差异落在 <8px 档还是 ≥16px 档。若四组的可达子集指标都动不了，就停止在数据侧投入，直接进阶段三。**

**阶段三：分辨率阶梯（用阶段二胜出的数据配置）**
640（已知）→ 800（同时改 `scale_jitter.sizes` 到 640–1024）→ 1024。同步做 T2 第 16 条的 SAHI `slice_size` 扫描（320/448/512）。
**判据：可达子集（≥16 设备像素）的 AP50 是否随分辨率单调上升。预期这是全流程里涨幅最大的一步。**

**阶段四：数据 v2（修边界 + 过采样密集图 + 比例扫描）**
先只改切片边界规则生成 v2（唯一变量），再加过采样，最后扫混合比例。每步单独验证。

**阶段五：结构改动**
按 T3 的 17 → 18 → 19 → 20 顺序，每次一个变量。17 和 18 针对的是本报告定位的两个结构性上限（最小可检尺度、密集图容量），优先级高于 19/20。

**明确不要做的三件事**
- 不要在当前口径下继续调切片混合比例——72% 的 GT 贴地板，信噪比不够，会得到随机结论。
- 不要指望延长训练（80,000 step 已进平台期）或换更多数据来源（数据量不是瓶颈）。
- 不要在 SAHI 阈值为 0.3 的情况下比较 mAP（两条路径口径不同）。

---

## 7. 方法与限制

**实测部分（可复现）**：本地数据全量统计（`ceiling_audit.py`，含 GT 计数、原生/设备像素分档、容量上限）；模型与增强代码逐行核对（patch size、特征层级、先验尺寸、query 数、损失归一化、ZoomOut/RandomIoUCrop 语义、SAHI 切片逻辑、maxDets 与面积阈值）；划分来源来自 `split_manifest.csv` 与既有 `artifact.json`。

**文献外推部分（需按自身条件验证）**：分辨率收益量级（SCRFD / ESOD / YOLOX 数据）、NWD 与 scale-aware loss、P2 层、DQ-DETR 的 query 自适应、copy-paste 合成比例。这些是在别的任务/数据集上的数字，只用于排序优先级，不代表本地可复现的幅度。

**仍待回答**：
- 原始模型（旧服务器）的完整训练配置（数据版本、steps、lr、seed、增强开关）依然缺失，因此"原模型 vs combined"不是严格单变量对照。上一轮已用截图与脚本工作区改动推断为 vits16、80,000 step、batch 32、冻结主干，但**没有服务器侧日志或 checkpoint 可核**。
- 官方属性标注本地不可恢复，easy/medium/hard 无法复算，必须重新下载官方标注包。
- 本次未做训练复验；报告中的所有因果表述（分辨率、密集图容量、切片边界）都是"机制已确认、贡献待消融"。
- 3,915 个边界候选的语义可辨认程度只人工确认了 1 例，其余待抽查。
- 单 seed、两次实验相差 0.94pp，**这个差值本身还没有被证明超出训练噪声**。

---

## 8. 证据索引

**本地**
- 数据与清单：`datasets/face_detect/face_yolo_wider`、`face_yolo_wider_combined_v1`（`augmentation_manifest.jsonl`、`preparation_summary.json`）、`datasets/face_detect/face_yolo/split_manifest.csv`
- 生成器：`datasets/convert_datasets/convert_tools/face_wider_prepare.py:387-427`（`make_tile`）
- 训练与增强：`train_face_wider.py`；`src/lightly_train/_task_models/dinov3_ltdetr_object_detection/transforms.py`、`task_model.py`、`dinov3_vit_wrapper.py`；`src/lightly_train/_transforms/object_detection_transform.py`、`scale_jitter.py`、`random_zoom_out.py`、`random_iou_crop.py`
- 匹配与损失：`src/lightly_train/_task_models/object_detection_components/rtdetrv2_criterion.py`、`matcher.py`、`rtdetrv2_decoder.py`
- 推理与评测：`tool_lib/det_infer.py`、`src/lightly_train/_metrics/mean_average_precision.py`
- 本报告统计：`reports/face_wider_tiny_face_strategy/ceiling_audit.py` → `ceiling_audit.json`
- 分档评测工具（第 0 步用）：`reports/face_wider_tiny_face_strategy/band_eval.py`（`--self-test` 可离线校验匹配与分档逻辑）
- 前序报告：`reports/face_wider_combined_audit/report_source.md`

**外部**
- WIDER FACE 官方项目与标注约定（≤10px 标 Ignore）：[WIDER FACE](https://mmlab.ie.cuhk.edu.hk/projects/WIDERFace/)、[CUHK-CSE/wider_face 数据卡](https://huggingface.co/datasets/CUHK-CSE/wider_face)、[WIDER FACE, CVPR 2016](https://arxiv.org/abs/1511.06523)
- 官方评测实现（`thresh_num=1000` 是 PR 采样点数，不是 proposal 上限）：[WiderFace-Evaluation](https://github.com/wondervictor/WiderFace-Evaluation)
- 标注噪声：[BDC, IET Computer Vision 2022](https://ietresearch.onlinelibrary.wiley.com/doi/10.1049/cvi2.12122)
- data-anchor-sampling：[PyramidBox, ECCV 2018](https://arxiv.org/abs/1803.07737)
- 小目标过采样 + 多次粘贴：[Augmentation for Small Object Detection](https://arxiv.org/abs/1902.07296)
- Simple Copy-Paste：[Ghiasi et al., CVPR 2021](https://arxiv.org/abs/2012.07177)
- ZoomOut 出处与语义：[SSD](https://arxiv.org/abs/1512.02325)、[torchvision RandomZoomOut](https://docs.pytorch.org/vision/main/generated/torchvision.transforms.v2.RandomZoomOut.html)
- SAHI：[arXiv:2202.06934](https://arxiv.org/abs/2202.06934)
- 分辨率收益（人脸）：[SCRFD, ICLR 2022](https://arxiv.org/abs/2105.04714)
- 分辨率收益（通用小目标）：[ESOD, IEEE TIP 2025](https://arxiv.org/abs/2407.16424)
- 密集图 query 上限：[DQ-DETR, ECCV 2024](https://arxiv.org/abs/2404.03507)
- 微小目标度量：[NWD](https://arxiv.org/abs/2110.13389)、[RFLA, ECCV 2022](https://arxiv.org/abs/2208.08738)
- ViT 单尺度特征金字塔：[ViTDet](https://arxiv.org/abs/2203.16527)；[DINOv3](https://github.com/facebookresearch/dinov3)
- 小目标综述：[Small Object Detection: A Comprehensive Survey](https://arxiv.org/abs/2503.20516)
