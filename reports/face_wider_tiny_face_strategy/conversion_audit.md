# 数据集转换与划分审计

审计日期：2026-09-10。范围：WIDER FACE 人脸数据集从原始标注到 `face_yolo_wider` 的转换与划分，以及三个派生集（`face_yolo_wider_tiles_v1` / `_copypaste_v1` / `_combined_v1`）的生成。

**结论：标注转换层（LabelMe → YOLO）的做法是合理的；划分层与派生规则不合理，而后者恰好决定评测数字的可信度。**

审计方法说明（供读者判断证据强度）：标 **[亲验]** 的条目由本次审计直接读代码或用数据核对确认；标 **[代码审计]** 的条目来自对源码的定向审计并附 `文件:行号`，未逐条复跑，可按行号自查。

---

## 1. 转换链路

| 环节 | 实现 | 说明 |
|---|---|---|
| 入口 | `datasets/convert_datasets/convert.py` | `oneclick` 等命令的总入口，`doctor` 校验注册与 CLI 契约 |
| 划分 + 复制 | `convert_tools/sync_picture.py` | 真正洗牌与切分的位置 |
| 标注格式转换 | `convert_tools/LabelMeToYOLO.py` | LabelMe JSON → YOLO 5 列 |
| 路径/目录编排 | `convert_tools/one_click_convert.py` | 多来源合并、源划分保留、类别 schema 对齐 |
| 派生集生成 | `convert_tools/face_wider_prepare.py` | tile 与 copy-paste |
| 原子发布 | `convert_tools/dataset_transaction.py` | staging → 校验 → `os.replace` → 回滚 |

---

## 2. 合理的做法

| 项 | 证据 |
|---|---|
| 坐标转换：先 clip 到图像边界，再 min/max 归一化，退化框（宽或高 ≤0）**抛错并计数**而非静默丢弃 | `LabelMeToYOLO.py:349-354,374-385`（`bbox_to_yolo`、`_clip_bbox`）；计数处 `:1101-1107` **[代码审计]** |
| 跨来源类别映射按**名字**重映射 id，类别集合不一致直接报错，避免 class id 漂移 | `one_click_convert.py:31-66`（`resolve_yolo_schemas`，`:47` 抛 ValueError）**[代码审计]** |
| 多来源身份用 `(source_index, 相对路径)`，输出名冲突加 `__2` 并记 `rename_log` | `sync_picture.py:341-353,465` **[代码审计]** |
| 真原子发布：staging → 校验 → `os.replace`，失败回滚，另有输出级 `flock` 互斥 | `dataset_transaction.py:137-153,175-204` **[代码审计]** |
| `--dry-run` 真零写入（有测试断言） | `face_wider_prepare.py:577-579`；`tests/test_face_wider_prepare.py:47-48` 断言 `not output.exists()` **[代码审计]** |
| 输出与源**双向**嵌套检查 + 符号链接拒绝 + 非空输出默认拒绝 | `dataset_transaction.py:92-99,120-121,164-170` **[代码审计]** |
| manifest 记录 tile 的 `crop_xyxy` 与粘贴的 donor/recipient/patch 坐标，可做几何重建；实测 6,442 张 tile 重建失配为 0 | `reports/face_wider_combined_audit/verify_tiles.py` **[亲验]** |
| 标签数值精度无系统偏移 | 固定 6 位小数；全量 196,080 框实测仅 1 个值不在开区间，且为合理 clip 结果 **[亲验]** |

---

## 3. 问题清单

### 第一档：直接污染评测结论

**3.1 划分按图片随机，不按场景分组** **[亲验]**

`sync_picture.py:384-400`：

```python
def split_indices(n: int, ratio=(0.8, 0.1, 0.1), seed=None):
    indices = list(range(n))
    rng = random.Random(seed)
    rng.shuffle(indices)
    n_train = round(n * ratio[0]); n_val = round(n * ratio[1])
    train_idx = indices[:n_train]
    val_idx = indices[n_train:n_train + n_val]
    test_idx = indices[n_train + n_val:]
```

完全不看事件、序列、类别或来源。后果：61 个事件在 train/val/test 三份中 100% 重叠；test 的 1,615 张图全部与 train 共享连拍子场景。

**3.2 没有任何跨 split 重复检查** **[亲验]**

全流程无该检查。实测 2 对 train↔test **字节完全相同**的图片（`wider_train_41_Swimming_Swimming_41_184` ↔ `..._Swimmer_41_745`；`wider_train_5_Car_Accident_Accident_5_907` ↔ `wider_train_39_Ice_Skating_Ice_Skating_39_617`），另 1 对在 val 内部。

值得注意的是：`yolo_dataset_merger.py` **具备**这个能力（全量 SHA-256 + `--split-leakage-policy warn|error|drop`，`:1285-1310`），但这条流水线没有使用它，而且该策略在 `tests/` 里零覆盖 **[代码审计]**。

**3.3 唯一的整体性校验返回值被丢弃** **[代码审计]**

`LabelMeToYOLO.py:1179-1210` 的 `integrity_check` 只检查"同一 split 内图片↔标签配对"，且调用处 `:1390` 丢弃返回值，校验失败也不会中止流程。

### 第二档：改变监督语义

**3.4 切片按"框自身面积 <50%"丢框**（已知问题） **[亲验]**

`face_wider_prepare.py:408` `if ratio < 0.50: continue`。窗口图像保留、对应框被删。全量重建：2,405 张 tile 受影响（占 tile 的 37.3%），3,915 个候选，其中 1,241 个是"保留 ≥25% 面积且可见区域 ≥8 设备像素"的重点候选。

**3.5 Copy-Paste 的粘贴区域与标签框不一致** **[亲验]**

`face_wider_prepare.py:255-271`：

```python
feather = max(1, min(4, min(patch_w, patch_h) // 8))
mask = Image.new("L", adjusted.size, 0)
face_pad_x = (scaled_face[2] - scaled_face[0]) * 0.06
face_pad_y = (scaled_face[3] - scaled_face[1]) * 0.06
ImageDraw.Draw(mask).ellipse((... scaled_face ± 6% ...), fill=255)
mask = mask.filter(ImageFilter.GaussianBlur(max(0.8, feather / 2)))
recipient.paste(adjusted, (left, top), mask)
x1, y1, x2, y2 = face_xyxy        # 标签写的是矩形
```

实际只粘贴"内接于脸框 + 6% 外扩的**椭圆**"，标签写出的却是**矩形框**。三个后果：框四角（约框面积的 21%，即 1−π/4）没有粘贴像素；椭圆 6% 外扩使部分脸像素落在标签框外（漏标）；羽化环带是半透明混合像素。量级对 12–28px 的粘贴小脸不算可忽略。另有 `:250-254` 的亮度匹配（0.75–1.25 倍）改变 donor 外观，属域偏移，不改标签。

**3.6 密集裁剪被整块丢弃** **[亲验]**

`:422` 的接受条件含 `1 <= len(boxes) <= max_boxes`（`tile_max_boxes` 默认 100），超过则重试 30 次后返回 `None`。因此 **GT>100 的密集窗口系统性生成不出 tile**——这解释了为什么 tile 子集里"GT>100 的图片数"为 0，也是"密集图曝光下降"在生成阶段的第二个机制（第一个是采样概率被稀释）。

其他会丢框/丢 tile 的分支 **[代码审计]**：`:396-399` `side < 64` 整块丢；`:422` 强制目标脸放大 ≥1.25× 且 ≥12px 等效短边，不满足则整块丢；`:455` 每源图最多 3 张 tile；`:463` 同源图内裁剪 IoU>0.9 去重；`:453-454` 配额打满即停且不报错。保留的框会被 clip 到 tile 边界（`:410-419`），但**裁剪后没有最小尺寸/最小可见面积过滤**。

### 第三档：可复现性与参数化

**3.7 `--seed` 只在"同 mode"下是复现键** **[亲验]**

`face_wider_prepare.py:581` 只创建一个 RNG：

```python
rng = random.Random(args.seed)
...
generate_copy_paste(splits["train"], stage, manifest, rng, ...)   # :589
...
generate_tiles(splits["train"], stage, manifest, rng, ...)        # :605
```

`combined` 模式下 Copy-Paste 先消耗随机流，tile 阶段拿到错位后的随机数。产物证据：两数据集的 copy-paste 记录**逐字节相同**（pastes 均 2,931）、tile 记录数都是 6,442、几何重建失配 0，但 tile 框总数 134,941 vs 135,455、唯一源图 3,504 vs 3,454（交集 2,576）。

> 这条同时**推翻**了 `reports/face_wider_tiny_face_strategy/report.md` 初稿里"生成代码在两次运行之间改动过"的归因，该处已更正。

**3.8 划分按遍历位置而非文件名洗牌** **[亲验]**

`sync_picture.py:515-519` 以 `enumerate(paired_items)` 的下标为键建立 `split_map`，而 `paired_items` 的顺序来自目录遍历（`os.walk`，`:368` **[代码审计]**）。同 seed 在不同文件系统/机器上不保证同一划分。

**3.9 比例硬编码、无分组键参数** **[亲验]**

`splits_indices` 默认 `(0.8, 0.1, 0.1)` 为函数常量；`copy_files` 只透传 `seed`。`one_click_convert.py` 仅暴露 `--seed`（默认 `None`，即默认不可复现）与 `--preserve-splits auto|yes|no`（`:242-258`）。**没有** ratio 参数，**没有**分组键/正则参数。

**3.10 manifest 信息量不足** **[代码审计]**

tile 记录只有 `boxes`（整数计数）与 `crop_xyxy`，**不含任何一个框坐标**，也没有 seed、被丢弃框清单、被跳过的接收图；`preparation_summary.json` 无代码版本（git commit / 文件哈希）与输入摘要。结论：标签内容无法从 manifest 独立复核，只能回源重建——审计脚本 `verify_tiles.py` 正是这么做的。

**3.11 校验偏弱** **[代码审计]**

`validate_stage`（`:517-534`）只做：图片/标签 stem 集合相等、标签可解析、图片可解码。**没有** 与 `plan()` 的生成前后计数比对、没有哈希、没有 manifest↔磁盘核对。tile 数量不足（`:489-490`）与"接收图零产出"（`:352-353`）都只打印或跳过，不会让流程失败。

### 第四档：低 / 需知悉

| 项 | 说明 |
|---|---|
| **所有图片宽度都是 1024** **[亲验]** | 全量 16,094 张，宽度≠1024 的为 **0** 张。说明进仓库脚本之前有过一次统一缩放，**该步骤不在仓库里**。影响：所有以"原生像素"为口径的统计（含"高度 ≤10px"）都是在缩放后的像素上算的，与官方标注包的原始像素可能不是同一尺度 |
| **基础图是硬链接** **[亲验]** | `face_yolo_wider` 与三个派生集的图片共享同一 inode（link count 4，`find -samefile` 命中 4 个数据集）。路径分离但数据未隔离：就地改图会同时改掉四个数据集。标签是独立复制，改标签安全 |
| YOLO→YOLO 路径 `_clip01` 语义错误 **[亲验]** | `LabelMeToYOLO.py:586-587` 对 `cx,cy,w,h` **各自独立** clamp 到 [0,1]，不保证框在图中（如 `cx=0.99,w=0.9` → 右边界 1.44）；`w>1` 被静默改成 1.0；`w=0/h=0` 因 0 可通过 clip 而被保留；`nan` 经 min/max 静默变 0.0。**WIDER 走 LabelMe 路径，未经过它，当前数据不受影响** |
| 无 JSON 的图被整体丢弃 **[代码审计]** | `CREATE_EMPTY_TXT_FOR_IMAGE_WITHOUT_JSON = False`（`:109-111,988-999`）。检测器见不到"无目标图"。本数据集恰好没有这种图，未暴露 |
| 标签文件缺结尾换行 **[代码审计]** | `:1150` 用 `"\n".join(...)` 写出。Ultralytics 可正常解析，但 `wc -l` 会少计 1 行 |
| 原子替换只有 `--clean`【代码审计】 | `face_wider_prepare.py:660` 只有 `--clean`，无 `--overwrite`/`--force` 别名（AGENTS.md 与 README 列了三者）。语义一致，属文档宽容 |
| `dataset_transaction` 内有未被引用的并行实现 **[代码审计]** | `create_output_stage`/`publish_output_stage`（`:103,114`）全仓库无调用，属重复实现 |

---

## 4. 对既有结论的影响

| 既有结论 | 是否受影响 |
|---|---|
| 设备像素口径：test 有 72.07% 的 GT 小于一个 token；可达子集占 27.93% | **不受影响**。该口径按模型实际输入计算 |
| 极小脸曝光下降 28%、密集图采样概率下降 38% | **不受影响**，且现在有了生成阶段的补充机制（3.6） |
| 切片边界 3,915 个候选 | **不受影响**，现有代码级解释（3.4） |
| "原生高度 ≤10px 占 test 的 15.0%" | **需打折**。图片被统一缩放到宽 1024，与官方标注包原始像素不一定是同一尺度；要与官方 ignore 规则对齐，需拿官方原图复核（3.9 第 1 项） |
| `tiles_v1` 与 `combined_v1` 的 514 框差异 | **归因已更正**：不是代码改动，是共享 RNG（3.7） |
| "本地划分可用于 A/B，不可对外" | **仍然成立**；但 3.2 的两对图片级泄漏说明"两组共享同一偏差"这句话在个别样本上不严格 |

---

## 5. 修复面盘点：改成"按子场景分组划分"需要什么

**现成可用的骨架** **[代码审计]**：`rebalance_yolo_splits.py` 已经具备所需的大部分能力——汇总源三个 split → 按类别分层（`stratified_assign :311-399`）→ 原子发布到新目录；参数含 `--ratios T V T`（`:913`，默认 `0.8 0.1 0.1`）、`--seed`（`:921`）、`--deduplicate`、`--dry-run`、`--clean`；输出目录必须与源分离（`validate_output_location` 显式拒绝等于或嵌套于源根），且它**已在 `convert.py doctor` 的检查名单里**。

**缺什么**：分组键与组级分配。该脚本没有任何分组概念（`group`/`prefix`/`scene`/`GroupShuffleSplit` 全文件零命中）。

**分组键是否可得** **[亲验]**：

- 子场景键：`face_yolo_wider` 文件名去掉末尾序号即可得到（train/val/test 分别 168/166/162 个前缀；161 个三份全出现）。
- 事件键：文件名的 `N_EventName` 段给出 **61 个事件，61/61 三份全出现**。
- 官方事件目录（`0--Parade`）：只能从 `datasets/face_detect/face_yolo/split_manifest.csv` 的 `source_image` 字段取；该字段指向的源目录 `datasets/face/WIDER/` 已删除，**字符串本身仍可用作分组键，但无法回源核对**。
- `face_yolo_wider` 自身**没有逐图 manifest**，只有汇总计数与自由文本 `split_mapping`。

**全仓库无分组划分实现** **[代码审计]**：`tool_lib/`、`datasets/`、`src/`、`tests/` 中 `GroupShuffleSplit`/`GroupKFold`/`StratifiedGroupKFold`/sklearn 零命中；`scene`/`burst`/`连拍`/`序列` 在 `.py` 中零命中。现有划分类代码全是"按图随机"或"按类别分层"。

---

## 6. 出处

- 转换与划分：`datasets/convert_datasets/convert_tools/{one_click_convert,sync_picture,LabelMeToYOLO,face_wider_prepare,dataset_transaction}.py`
- 划分来源记录：`datasets/face_detect/face_yolo/split_manifest.csv`、`datasets/face_detect/face_yolo/audit.json`
- 相关既有报告：`reports/face_wider_combined_audit/report_source.md`（切片边界与曝光）、[report.md](report.md)（根因）、[action_plan.md](action_plan.md)（行动清单）
- 安全约定：`AGENTS.md` §"安全约定"、`datasets/convert_datasets/README.md`
- 测试：`datasets/convert_datasets/convert_tools/tests/{test_face_wider_prepare,test_conversion_safety,test_dataset_transaction}.py`、`tests/test_rebalance_yolo_splits.py`
