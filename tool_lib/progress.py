"""统一的单行进度条工具(固定一行,\\r 原地刷新,不刷屏)。

设计目标(对齐用户诉求「固定在最下面一行,不要一直往上刷新占满页面」):
  * 交互终端(TTY): 用 tqdm 单行进度条,\\r 原地刷新,始终只占一行。
  * 输出重定向到日志(非 TTY): 自动降级为周期性打印,避免刷爆日志文件。
  * 未安装 tqdm: 退化为轻量自实现的 \\r 单行条,功能不受影响。

用法:
    from tool_lib.progress import track, plog
    for x in track(items, label="det/infer 推理", unit="img"):
        ...
        plog("某条需要保留的日志")   # 不会打断底部进度条
"""
from __future__ import annotations

import sys
import time
from typing import Iterable, Iterator, Optional, TypeVar

_T = TypeVar("_T")

try:
    from tqdm import tqdm as _tqdm
except ModuleNotFoundError:  # pragma: no cover
    _tqdm = None


def _isatty() -> bool:
    try:
        return bool(sys.stdout.isatty())
    except Exception:  # pragma: no cover
        return False


class _FallbackBar:
    """无 tqdm 时的极简单行进度条: [####....] cur/total。"""

    def __init__(self, label: str, total: Optional[int], width: int = 28) -> None:
        self.label = label
        self.total = total
        self.width = width
        self._last = 0.0
        self._tty = _isatty()

    def update(self, current: int) -> None:
        now = time.monotonic()
        done = self.total is not None and current >= self.total
        # TTY 上一律 \r 单行;非 TTY 上每 0.8s 提交一行,便于日志留痕。
        if not done and not self._tty and (now - self._last) < 0.8:
            return
        if self.total:
            filled = int(self.width * min(current, self.total) / self.total)
            bar = "#" * filled + "." * (self.width - filled)
            body = f"[{bar}] {min(current, self.total)}/{self.total}"
        else:
            body = f"{current}"
        end = "\n" if (done or not self._tty) else "\r"
        print(f"  {self.label} {body}", end=end, flush=True)
        self._last = now

    def close(self) -> None:
        if self._tty:
            print("", flush=True)  # 收尾换行,避免后续输出粘在进度条后


def track(
    iterable: Iterable[_T],
    *,
    label: str = "",
    total: Optional[int] = None,
    unit: str = "it",
    enable: bool = True,
) -> Iterator[_T]:
    """遍历 iterable 并显示固定单行进度条。

    total 未给时会尝试 len(iterable);对生成器请显式传 total。
    enable=False 可一键关闭(例如非交互批处理时)。
    """
    if not enable:
        yield from iterable
        return
    if total is None:
        try:
            total = len(iterable)  # type: ignore[arg-type]
        except (TypeError, AttributeError):
            total = None

    if _tqdm is not None:
        # tqdm 自身即 TTY 感知: 终端里 \r 单行, 重定向时按 mininterval 周期输出。
        yield from _tqdm(
            iterable, desc=label, total=total, unit=unit,
            dynamic_ncols=True, leave=True,
        )
        return

    bar = _FallbackBar(label, total)
    for i, item in enumerate(iterable, 1):
        yield item
        bar.update(i)
    bar.close()


def plog(message: str) -> None:
    """在进度条运行期间安全打印一条日志(不打断底部进度条)。"""
    if _tqdm is not None:
        _tqdm.write(message)
    else:  # pragma: no cover
        print(message, flush=True)
