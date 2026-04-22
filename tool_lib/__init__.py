"""tool_lib 包入口。

这个目录放的是 launcher.py 依赖的功能模块：
- common.py: 公共配置、运行时依赖、通用工具函数
- interactive.py: 交互菜单和命令行参数入口
- dispatch.py: 根据用户选择分发到具体功能
- *_tools.py: 各任务的具体实现
- script_runner.py: 运行现有 train/test 脚本
"""

from .common import import_runtime_dependencies
