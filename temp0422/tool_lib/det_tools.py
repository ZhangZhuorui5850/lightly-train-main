"""检测任务兼容入口。

保留 det_tools 对外导入路径，内部实现已经拆分到 det_shared / det_analysis / det_export / det_infer。
"""

from __future__ import annotations

from .det_shared import *  # noqa: F401,F403
from .det_analysis import *  # noqa: F401,F403
from .det_export import *  # noqa: F401,F403
from .det_infer import *  # noqa: F401,F403

