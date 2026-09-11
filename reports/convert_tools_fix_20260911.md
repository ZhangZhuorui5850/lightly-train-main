# 转换入口与工具检查（2026-09-11）

检查范围：`convert.py` 注册与命令分发、工具 CLI、CSV 字段、共享输出事务，以及已有转换测试覆盖的配对、类别映射、合并和发布流程。验证数据均位于测试临时目录。

## 已修复

1. 交互启动台在子工具参数错误或普通执行失败后回到菜单，支持修正参数继续运行；直接 CLI 调用保留子进程退出码。
2. `doctor` 扩展到全部 25 个注册命令，实际执行每个 `--help`；写入工具全部检查 `--dry-run`。LabelMe 检查工具标记为只读。
3. `labelme2yolo`、`seg2mvtec`、`sample-browse`、`make-sample` 补齐零写入预览；LabelMe 增加显式 `--clean`。
4. YOLO 转 MVTec 时，缺陷类别 `good` 自动分配独立目录，正常样本继续使用 `good`。物体版中重复缺陷名以及大小写、Unicode 规范化后重名的类别与物体分配独立目录，保留各自图像与掩码。
5. LabelMe 在开始时输出尚为空、转换中途其他进程创建输出的情况下，发布阶段再次检查覆盖权限，保护新出现的文件。

CSV 写入的字段与记录对应关系已检查；转换工具现有 CSV 路径及本次新增场景通过回归验证。

## 验证

先新增复现测试，观察到 6 项失败；修复后运行完整转换测试集：

```bash
conda run -n lightlytrain python -m pytest tests/test_convert_entry.py datasets/convert_datasets/convert_tools/tests -q
```

结果：225 项通过。包括真实 CLI 帮助检查、转换产物、不同类别掩码完整性、零写入预览和并发覆盖保护。服务器实际数据规模下的运行仍需更新后验证。

## 更新

同步 `datasets/convert_datasets/convert.py` 和完整 `datasets/convert_datasets/convert_tools/` 到服务器对应仓库，使用现有 `lightlytrain` 环境。更新后执行：

```bash
conda run -n lightlytrain python datasets/convert_datasets/convert.py doctor
```

正式转换前，给原命令加 `--dry-run` 检查计划，随后选择与源数据分离的输出目录执行。
