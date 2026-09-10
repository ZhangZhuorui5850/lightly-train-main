# face_yolo_wider 小目标检测与训练优化分析

> 数据集：`datasets/face_detect/face_yolo_wider`
> 任务：单类别人脸检测  
> 当前模型：DINOv3 ViT-S/16 系列 backbone + LTDETR  
> 本机环境：RTX 4060 Laptop，8GB 显存  
> 报告日期：2026-08-31

## 技术结论

1. **小目标总量已经充足。** train 中共有 156,219 个有效人脸框，其中 COCO small 框 127,183 个，占 81.4%；按短边统计，137,213 个框小于 32 像素，占 87.8%。
2. **当前瓶颈集中在有效像素、样本分布和密集场景容量。** 40.5% 的 train 人脸在 640 输入下短边小于 8 像素；51.9% 的训练图片完全没有短边小于 32 像素的人脸；超过 100 张脸的 1.83% 图片承载了 41.3% 的小脸。
3. **普通 GT 框复制的预期收益较低。** 继续复制现有极小、模糊的人脸主要增加重复像素和标签数量。定向 Copy-Paste 可以把清晰人脸缩小后分散到稀疏图片，提升“含小脸图片”的覆盖率。
4. **训练优化优先级为：减弱 Random Zoom Out、训练切片、定向 Copy-Paste、提高输入分辨率。** 模型结构更换适合放在数据与尺度实验之后。
5. **本机采用短周期筛选。** 四组方案各训练 5,000 step，胜出方案训练到 20,000 step。RTX 4060 使用 `batch_size=2`，LightlyTrain 自动梯度累积形成有效 batch 16。

## 1. 分析范围与指标定义

### 1.1 数据来源

- 数据配置：[data.yaml](../datasets/face_detect/face_yolo_wider/data.yaml)
- 图片目录：`datasets/face_detect/face_yolo_wider/images/{train,val,test}`
- 标签目录：`datasets/face_detect/face_yolo_wider/labels/{train,val,test}`
- LTDETR 训练增强：[transforms.py](../src/lightly_train/_task_models/dinov3_ltdetr_object_detection/transforms.py)
- LTDETR 查询数量：[task_model.py](../src/lightly_train/_task_models/dinov3_ltdetr_object_detection/task_model.py)
- LightlyTrain 梯度累积：[train_task.py](../src/lightly_train/_commands/train_task.py)
- 本机训练速度参考：`out/0428/0428-neu-vits16/train.log`

### 1.2 尺寸口径

验证阶段默认把图像 resize 到 640×640。YOLO 标签中的宽高为归一化坐标，因此 resize 后尺寸按以下方式计算：

```text
box_width_px  = normalized_width  × 640
box_height_px = normalized_height × 640
```

本报告使用两种尺寸口径：

- **COCO面积口径**
  - small：面积 `< 32²`
  - medium：`32² ≤ 面积 < 96²`
  - large：面积 `≥ 96²`
- **短边口径**
  - `<8px`
  - `8–16px`
  - `16–32px`
  - `≥32px`

COCO面积口径用于对应评估器的 `AP_small/medium/large`；短边口径用于分析 ViT `/16` patch 和小脸可见像素。

## 2. 数据质量与数据规模

### 2.1 基础质量检查

| 检查项 | 结果 |
|---|---:|
| 缺失标签文件 | 0 |
| 空标签文件 | 0 |
| 非法类别 | 0 |
| 中心坐标越界 | 0 |
| 宽高无效或越界 | 0 |
| 类别数量 | 1（face） |

数据目录结构和YOLO标签格式完整，可以支持训练与尺寸分析。

### 2.2 Split规模

| Split | 图片数 | 标注框数 | 平均每图框数 | 每图框数中位数 | 单图最大框数 |
|---|---:|---:|---:|---:|---:|
| train | 12,883 | 156,219 | 12.13 | 3 | 1,962 |
| val | 1,596 | 16,995 | 10.65 | 3 | 467 |
| test | 1,615 | 22,866 | 14.16 | 3 | 809 |

截图中的 test `GT=22,865`，本地有效YOLO框统计为 22,866，差异为 1 个框，占 0.0044%。该差异可能来自评估阶段的边界裁剪或有效框过滤，对尺寸分布结论影响极小。

### 2.3 数据来源构成

当前三个 split 都混合包含原始 `wider_train` 和 `wider_val` 图片：

| Split | wider_train来源 | wider_val来源 |
|---|---:|---:|
| train | 10,318 | 2,565 |
| val | 1,271 | 325 |
| test | 1,283 | 332 |

这种划分适合内部随机验证。若需要与 WIDER FACE 官方 easy/medium/hard 结果比较，需要恢复官方 split 和难度定义。

## 3. 小目标数量充足，有效尺度依然困难

### 3.1 COCO面积尺寸构成

| Split | Small | Medium | Large |
|---|---:|---:|---:|
| train | 81.4% | 15.4% | 3.2% |
| val | 79.0% | 17.3% | 3.7% |
| test | 84.0% | 13.4% | 2.6% |

test 的 small 占比最高，medium 和 large 占比最低，因此 test 总体指标会更受小脸性能影响。

### 3.2 640输入下的短边尺寸

train 中各短边区间如下：

| 短边区间 | 框数 | 占train全部框 |
|---|---:|---:|
| `<8px` | 63,246 | 40.5% |
| `8–16px` | 45,741 | 29.3% |
| `16–32px` | 28,226 | 18.1% |
| `≥32px` | 19,006 | 12.2% |

三个 split 的短边分位数：

| Split | P10 | P25 | P50 | P75 | P90 |
|---|---:|---:|---:|---:|---:|
| train | 3.75px | 5.62px | 9.37px | 18.75px | 36.25px |
| val | 4.38px | 6.25px | 10.63px | 20.63px | 39.90px |
| test | 4.38px | 5.62px | 9.37px | 17.50px | 32.50px |

ViT-S/16 的 patch size 为 16。train 中 69.8% 的人脸短边小于 16 像素，test 中该比例为 72.1%。一个目标覆盖的有效patch数量很少，分类置信度与边界定位都会受到影响。

## 4. 小脸集中在少量密集图片

按train图片的总框数分组：

| 每图框数 | 图片数 | 图片占比 | 组内小脸数 | 占全部小脸 |
|---|---:|---:|---:|---:|
| 0–5 | 8,476 | 65.79% | 4,857 | 3.54% |
| 6–20 | 2,871 | 22.29% | 25,743 | 18.76% |
| 21–100 | 1,301 | 10.10% | 49,950 | 36.40% |
| 101–300 | 185 | 1.44% | 31,015 | 22.60% |
| `>300` | 50 | 0.39% | 25,648 | 18.69% |

这里的“小脸”采用短边 `<32px` 口径。

关键现象：

- 8,476 张稀疏图片占train的65.8%，仅包含3.5%的小脸。
- 6,682 张图片完全没有短边小于32像素的人脸，占train的51.9%。
- 235 张超过100个框的图片包含56,663个小脸，占全部小脸的41.3%。
- 50张图片超过LTDETR的300 query容量，超过300的GT合计10,664个，占train全部框的6.83%。

这说明小目标的框数量很多，图片级覆盖率偏低。均匀按图片采样时，大量batch来自没有小脸或小脸很少的图片；密集batch又会承受匹配、显存和query容量压力。

## 5. 当前指标低的原因

用户提供的评估截图显示：

| Split | mAP@0.5:0.95 | mAP@0.5 | mAP@0.75 | AP Small | AP Medium | AP Large |
|---|---:|---:|---:|---:|---:|---:|
| test | 0.2835 | 0.5403 | 0.2645 | 0.1608 | 0.5901 | 0.7253 |
| val | 0.3290 | 0.6134 | 0.3191 | 0.1920 | 0.5965 | 0.7181 |

### 5.1 主要损失来自小目标

test 的 `AP_small=0.1608`，medium和large分别达到0.5901和0.7253。模型已经能处理较清晰的中大人脸，总体mAP主要受到占比84%的small目标影响。

### 5.2 AP50与AP75差距反映定位困难

test 的 `mAP@0.5=0.5403`，`mAP@0.75=0.2645`，相差0.2758。对于8×8或12×12的小框，预测边界偏移2–3像素即可造成显著IoU下降。

### 5.3 test比val更难

test相较val具有以下特征：

- COCO small占比：84.0% vs 79.0%。
- 短边 `<8px`：40.8% vs 34.8%。
- 平均每图框数：14.16 vs 10.65。
- 每图至少20个框：14.6% vs 11.9%。
- 每图至少50个框：5.6% vs 4.6%。

这些差异与test mAP低于val的方向一致。

### 5.4 两套AP50需要分开使用

截图中的汇总表给出test `mAP@0.5=0.5403`，类别表给出face `AP@0.5=0.5924`。单类别任务中两者依然不同，说明它们来自不同评估路径、预测过滤或阈值设置。

训练对照实验应固定使用同一份COCO汇总指标：

```text
mAP@0.5:0.95
mAP@0.5
mAP@0.75
AP_small / AP_medium / AP_large
AR_small / AR_medium / AR_large
```

## 6. GT Copy-Paste可行性

### 6.1 普通复制增加重复训练信号

train 已经拥有127,183个COCO small框和137,213个短边小于32像素的框。继续复制同一批极小人脸会增加以下训练信号：

- 相同模糊模式重复出现。
- 相同背景边缘重复出现。
- 极小框数量继续提高。
- 密集图片中的query竞争继续增强。

这类增强对“数量不足”有效，当前数据的主要问题集中在有效像素与图片级分布。

### 6.2 定向Copy-Paste具有明确目标

合理目标是把高质量、较清晰的人脸缩小后粘贴到稀疏图片，使更多训练图片包含12–28像素的人脸。

推荐参数：

| 项目 | 推荐值 |
|---|---|
| 供体尺寸 | 640等效短边24–96px |
| 目标尺寸 | 短边12–28px，采用对数均匀采样 |
| 接收图片 | 2–5框、无小脸、平均脸短边≤128px的多人图片 |
| 每张粘贴数 | 1–3 |
| 最终框数上限 | 20 |
| 与已有GT的IoU | ≤0.05 |
| 粘贴框之间IoU | ≤0.10 |
| 边缘处理 | 2–4px羽化 |
| 颜色处理 | 局部亮度、对比度和色温匹配 |
| 单供体复用上限 | 5次 |
| 首轮增强比例 | 符合条件接收图的70% |
| 数据使用范围 | train |

实际筛选出2,094张符合条件的多人接收图，选择其中70%生成1,466张增强副本；每张平均粘贴2张脸，共新增2,931个小脸框。保留原图并新增增强副本后，含小脸图片的采样占比从48.1%提高到约53.4%。这种设计主要改善图片级覆盖率，并将合成样本控制在训练图片的约10.2%。

### 6.3 供体质量比供体数量更重要

供体选择建议：

- 原始人脸短边至少24像素。
- 人脸框保留1.2–1.5倍上下文区域。
- 选择清晰度位于候选供体前70%的裁剪。
- 过滤贴近图像边缘、严重截断和宽高比异常的框。
- 使用较清晰供体向下缩放到12–28像素。

YOLO标签已经丢失WIDER原始的blur、occlusion、invalid等属性，因此供体质量需要通过尺寸、清晰度、边界位置和宽高比进行二次筛选。

### 6.4 Copy-Paste的主要风险

| 风险 | 可能结果 | 控制方法 |
|---|---|---|
| 矩形边缘明显 | 模型学习粘贴边缘 | 上下文裁剪、羽化、颜色匹配 |
| 人脸位置不自然 | 增加误检 | 限制粘贴区域并采样合理高度 |
| 供体重复过多 | 过拟合少数外观 | 供体复用≤5次 |
| 粘贴尺寸过小 | 重复低信息像素 | 目标短边以12–28px为主 |
| 接收图过密 | query竞争和显存增加 | 接收图≤5框，最终≤20框 |
| 标签遮挡关系错误 | 定位与分类噪声 | 控制IoU并保留完整框 |

## 7. 默认Random Zoom Out可能继续缩小小脸

DINOv3-LTDETR默认训练增强包含：

```python
class DINOv3LTDETRObjectDetectionRandomZoomOutArgs(RandomZoomOutArgs):
    prob = 0.5
    side_range = (1.0, 4.0)
```

该增强有50%概率把原图放到1–4倍画布中，再缩放到训练输入尺寸。原本16像素的人脸可能降到4–16像素，原本8像素的人脸可能降到2–8像素。

建议先做以下对照：

```python
# 方案1：关闭
transform_args={"random_zoom_out": None}

# 方案2：减弱
transform_args={
    "random_zoom_out": {
        "prob": 0.2,
        "side_range": (1.0, 1.5),
    }
}
```

当前 `launcher.py -> tool_lib/train_tools.py` 训练路径尚未传入 `transform_args`。该实验需要在launcher配置层增加参数传递，或者使用独立训练脚本。

## 8. 训练切片的预期收益更高

训练切片从高分辨率原图裁出围绕小脸或人群的局部区域，再把tile缩放到640输入。它具有三个直接作用：

1. 小脸获得1.5–4倍的有效放大。
2. 单tile内人脸数量下降，query容量压力减小。
3. 背景、遮挡、光照和人群结构保持自然。

推荐生成规则：

- tile尺寸：原图上的512–768像素窗口，保持方形或接近方形。
- overlap：20%–30%。
- 选片条件：至少包含1个原640等效短边 `<16px` 的人脸。
- 框保留条件：裁剪后保留面积≥原框70%。
- 每tile框数：1–100。
- 每张原图生成：1–3个有效tile。
- 数据混合：full image与tile样本按约1:1或2:1采样。
- val/test：保留全图评估，同时增加独立tile/SAHI诊断结果。

训练切片与SAHI形成一致的训练—推理尺度。当前SAHI缺少提升时，训练切片可以验证模型是否需要先学习局部放大视野。

## 9. 提高输入分辨率

输入分辨率提高后，小脸短边分布变化如下：

| Split | 输入 | 短边 `<8px` | 短边 `<16px` | COCO Small |
|---|---:|---:|---:|---:|
| train | 640 | 40.5% | 69.8% | 81.4% |
| train | 800 | 32.1% | 62.0% | 75.2% |
| train | 960 | 22.1% | 55.4% | 69.2% |
| test | 640 | 40.8% | 72.1% | 84.0% |
| test | 800 | 31.8% | 63.9% | 77.8% |
| test | 960 | 20.1% | 56.3% | 71.6% |

800输入可以把test中短边 `<8px` 的比例从40.8%降到31.8%。RTX 4060 8GB建议从 `batch_size=1` 开始，通过自动梯度累积维持有效batch。训练时间和显存压力会明显增加，适合作为第三阶段实验。

## 10. 方法优先级

| 方法 | 解决的问题 | 预期收益 | 风险/成本 | 优先级 |
|---|---|---|---|---:|
| 减弱/关闭Zoom Out | 防止小脸继续缩小 | 中到高 | 改动小，需要暴露transform_args | 1 |
| 训练切片 | 放大小脸、降低密度 | 高 | 需要生成派生数据集 | 2 |
| 定向Copy-Paste | 提高含小脸图片覆盖率 | 中 | 需要控制真实感 | 3 |
| 800输入 | 直接提高有效像素 | 中到高 | 显存与时间增加 | 4 |
| LTDETRv2 | 更换检测结构 | 待实验 | 需要新权重与一致对照 | 5 |
| 专用人脸检测器 | 匹配密集小脸任务 | 高价值基线 | 需要离线准备代码和权重 | 6 |

## 11. RTX 4060实验设计

### 11.1 第一阶段：5,000 step筛选

统一配置：

```text
model        = 当前相同的DINOv3-LTDETR模型
image_size   = 640
batch_size   = 2
steps        = 5000
seed         = 固定
train/val    = 固定
初始化权重    = 固定
评估阈值      = 固定
```

对照组：

| 实验 | 数据/增强 | 目的 |
|---|---|---|
| A | 当前原始方案 | 建立基线 |
| B | A + 关闭Random Zoom Out | 验证缩小增强的影响 |
| C | B + 1,466张定向Copy-Paste副本 | 验证图片级小脸覆盖率 |
| D | B + 训练切片 | 验证尺度与密度瓶颈 |

本机每组5,000 step预计约6–9小时，四组约24–36小时。所有实验使用独立输出目录。

### 11.2 第二阶段：胜出方案训练到20,000 step

20,000 step在有效batch 16下约等于：

```text
20000 × 16 ÷ 12883 ≈ 24.8个有效epoch
```

预计训练时间约26–38小时。20,000 step时曲线持续上升，可以续训到30,000 step；曲线进入平台后，计算预算转向下一种数据方案。

### 11.3 当前epoch换算风险

LightlyTrain在物理batch 2时自动梯度累积8次，有效batch为16。当前launcher的epoch换算按物理batch 2计算：

```text
ceil(12883 / 2) × 200 = 1,288,400 steps
```

按有效batch换算的200 epoch为：

```text
ceil(12883 / 16) × 200 = 161,200 steps
```

当前训练请选择step模式。200有效epoch在本机预计约9–13天，适合留给服务器执行。

## 12. 评估与验收标准

### 12.1 主指标

每次实验固定比较：

```text
mAP@0.5:0.95
mAP@0.5
mAP@0.75
AP_small
AR_small
Precision@0.5
Recall@0.5
```

### 12.2 尺寸分桶召回

建议在现有评估报告中增加：

```text
Recall_<8px
Recall_8_16px
Recall_16_32px
Recall_>=32px
```

该指标能区分Copy-Paste对极小脸、patch附近小脸和正常尺寸人脸的真实影响。

### 12.3 方案保留条件

5,000 step筛选阶段建议采用以下阈值：

- `AP_small`绝对提升≥0.02。
- `AR_small`绝对提升≥0.03。
- `AP_medium`下降≤0.02。
- `AP_large`下降≤0.02。
- `Precision@0.5`下降≤0.03。
- 可视化结果中的矩形粘贴边缘误检保持低水平。

这些阈值用于工程筛选，完整结论以20,000 step复验为准。

## 13. 离线服务器实施方案

服务器代码更新受限时，优先采用离线派生数据集：

```text
datasets/face_detect/face_yolo_wider
    原始数据集

datasets/face_detect/face_yolo_wider_copypaste_v1
    原始train + Copy-Paste增强副本
    原始val/test
    独立data.yaml
    augmentation_manifest.jsonl

datasets/face_detect/face_yolo_wider_tiles_v1
    原始train + 训练tiles
    原始val/test
    独立data.yaml
    augmentation_manifest.jsonl
```

`augmentation_manifest.json`建议记录：

- 增强图片路径。
- 接收图片路径。
- 供体图片和GT索引。
- 粘贴前后框坐标。
- 缩放比例。
- 羽化参数。
- 随机种子。

这种方式只需要把派生数据集复制到服务器，训练入口和`src/`保持现有版本。

## 14. 局限与待验证项

1. 当前mAP来自用户提供的服务器截图，本机缺少对应checkpoint和TensorBoard event，欠拟合、平台期和过拟合状态需要训练曲线确认。
2. Copy-Paste收益属于待验证假设。数据分布支持“图片级覆盖率可能改善”，最终因果结论需要固定配置的A/B实验。
3. YOLO转换后缺少WIDER原始blur、occlusion和invalid属性，供体质量只能通过图像启发式筛选。
4. 输入800/960的实际显存和step time需要在RTX 4060上用100–300 step预跑确认。
5. SAHI结果受到slice尺寸、overlap、置信度、NMS和全局/局部融合配置影响，训练切片实验需要同步保存完整SAHI参数。

## 15. 执行建议

1. 先完成A/B两组：当前基线与关闭Random Zoom Out，各训练5,000 step。
2. B组提升后，以B为基础生成Copy-Paste与训练切片两个派生数据集。
3. C/D各训练5,000 step，按`AP_small`和`AR_small`选择胜出方案。
4. 胜出方案训练到20,000 step，并在test上运行全图与SAHI两套评估。
5. 服务器最终训练沿用胜出方案和同一数据版本，保存训练曲线、配置快照与数据manifest。

当前最有信息量的第一步是关闭Random Zoom Out进行5,000 step对照；定向Copy-Paste作为第三组实验，验证其能否把小脸训练信号均匀分散到更多图片。

## 16. 已实现脚本与数据版本

已增加以下入口：

- `train_face_wider.py`：顶部集中配置数据版本、输出目录、step、输入尺寸、全局batch、模型和学习率；默认使用Tile数据、640输入、20,000 step、关闭Random Zoom Out；`BATCH_SIZE=0`时按可见GPU数量和总显存自动选择，本机RTX 4060选择batch 2；相同训练参数和输出目录支持断点续训。
- `tool_lib/training_run.py`：每次正式启动保存`run_config.json`参数指纹；续训参数发生变化时在加载优化器前终止；`--fresh`把旧实验完整移动到同级`_archive/`后创建新实验，隔离旧曲线、旧best权重和TensorBoard event。
- `datasets/convert_datasets/convert.py face-wider-prepare`：统一生成Copy-Paste、Tile和Combined派生数据集，支持`--dry-run`、独立staging、原子发布、固定随机种子、标签复核、manifest和增强预览。

实际生成结果：

| 数据版本 | train图片 | train框 | 新增内容 | 磁盘显示大小 |
|---|---:|---:|---|---:|
| `face_yolo_wider_copypaste_v1` | 14,349 | 163,109 | 1,466张CP，2,931个粘贴脸 | 2.2G |
| `face_yolo_wider_tiles_v1` | 19,325 | 291,160 | 6,442张真实tile | 2.9G |
| `face_yolo_wider_combined_v1` | 20,791 | 298,564 | 1,466张CP + 6,442张tile | 3.2G |

Tile平均包含20.9个框，中位数14，最大100。训练脚本默认选择`face_yolo_wider_tiles_v1`；Copy-Paste与Combined保留为独立对照实验。原始图片在同一文件系统内采用硬链接复用空间，标签采用独立复制，各派生目录本身保持完整的数据集结构，可独立归档上传。

生成后尺度复核显示：Copy-Paste新增框的640等效短边范围为12.01–27.99px，中位数19.80px；Tile内全部134,941个框的短边P10/P50/P90为8.63/17.63/39.09px，短边小于8px的比例为7.69%，相较原始train的40.5%显著下降。该结果确认Tile数据实现了预期的局部放大效果。真实裁剪IoU去重后，单张源图内裁剪IoU大于0.9的近重复Tile为0。

服务器默认运行命令：

```bash
conda run --no-capture-output -n lightlytrain python train_face_wider.py
```

运行前只读检查：

```bash
conda run --no-capture-output -n lightlytrain python train_face_wider.py --print-config
```

同一目录开启全新实验时使用`--fresh`。旧目录会自动归档，归档内容可直接恢复：

```bash
conda run --no-capture-output -n lightlytrain python train_face_wider.py --fresh
```

5,000 step筛选命令：

```bash
conda run --no-capture-output -n lightlytrain python train_face_wider.py --variant original --steps 5000 --out out/face_wider/B_original_zoomout_off
conda run --no-capture-output -n lightlytrain python train_face_wider.py --variant copy-paste --steps 5000 --out out/face_wider/C_copy_paste
conda run --no-capture-output -n lightlytrain python train_face_wider.py --variant tile --steps 5000 --out out/face_wider/D_tile
conda run --no-capture-output -n lightlytrain python train_face_wider.py --variant combined --steps 5000 --out out/face_wider/E_combined
```
