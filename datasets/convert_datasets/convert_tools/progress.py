"""Progress policy shared by standalone dataset conversion tools.

Interactive terminals use tqdm.  Redirected output uses periodic plain-text
snapshots so long conversions still expose phase, count, rate and elapsed time.
"""

from __future__ import annotations

import os
import sys
import time
from typing import Iterable

try:
    from tqdm import tqdm as _tqdm
except ModuleNotFoundError:  # pragma: no cover - tqdm is a project dependency
    _tqdm = None


def _isatty(stream) -> bool:
    try:
        return bool(stream.isatty())
    except Exception:
        return False


def _snapshot_interval() -> float:
    raw = os.environ.get("LIGHTLY_PROGRESS_SNAPSHOT_INTERVAL", "5")
    try:
        return max(0.2, float(raw))
    except ValueError:
        return 5.0


class _SnapshotProgress:
    """Small tqdm-compatible progress object for logs and missing tqdm."""

    def __init__(
        self,
        iterable: Iterable | None = None,
        *,
        total: int | None = None,
        desc: str = "",
        unit: str = "it",
        file=None,
        enabled: bool = True,
        initial: int = 0,
        **_kwargs,
    ) -> None:
        self.iterable = iterable
        if total is None and iterable is not None:
            try:
                total = len(iterable)  # type: ignore[arg-type]
            except (TypeError, AttributeError):
                total = None
        self.total = max(0, int(total)) if total is not None else None
        self.desc = str(desc)
        self.unit = str(unit)
        self.file = file or sys.stderr
        self.n = max(0, int(initial))
        self._enabled = enabled
        self._postfix = ""
        self._started_at = time.monotonic()
        self._last_print_at = 0.0
        self._closed = False
        self._emit(force=True, status="running")

    def __iter__(self):
        if self.iterable is None:
            raise TypeError("manual progress object is not iterable")
        try:
            for item in self.iterable:
                yield item
                self.update()
        except BaseException:
            self.close(status="failed")
            raise
        else:
            self.close(status="complete")

    def _render(self, status: str) -> str:
        elapsed = max(time.monotonic() - self._started_at, 0.0)
        rate = self.n / elapsed if elapsed > 0 else 0.0
        if self.total is None:
            count = str(self.n)
        else:
            percent = 100.0 * self.n / self.total if self.total else 100.0
            count = f"{self.n}/{self.total} ({percent:5.1f}%)"
        prefix = f"{self.desc} " if self.desc else ""
        postfix = f" | {self._postfix}" if self._postfix else ""
        return (
            f"[progress] {prefix}{count} {self.unit} | "
            f"{rate:.2f} {self.unit}/s | elapsed {elapsed:.1f}s | {status}{postfix}"
        )

    def _emit(self, *, force: bool, status: str) -> None:
        if not self._enabled:
            return
        now = time.monotonic()
        if not force and now - self._last_print_at < _snapshot_interval():
            return
        print(self._render(status), file=self.file, flush=True)
        self._last_print_at = now

    def update(self, n: int = 1) -> None:
        self.n += max(0, int(n))
        self._emit(force=False, status="running")

    def set_postfix_str(self, value: str, refresh: bool = True) -> None:
        self._postfix = str(value)
        if refresh:
            self._emit(force=False, status="running")

    def set_description_str(self, value: str, refresh: bool = True) -> None:
        self.desc = str(value)
        if refresh:
            self._emit(force=False, status="running")

    def close(self, *, status: str | None = None) -> None:
        if self._closed:
            return
        if status is None:
            status = "complete" if self.total is not None and self.n >= self.total else "stopped"
        self._emit(force=True, status=status)
        self._closed = True

    def __enter__(self):
        return self

    def __exit__(self, exc_type, *_args) -> None:
        self.close(status="failed" if exc_type is not None else None)


def tqdm(iterable=None, **kwargs):
    """Return tqdm in a TTY and periodic snapshots in redirected output."""
    kwargs.setdefault("dynamic_ncols", True)
    disable = kwargs.pop("disable", None)
    progress_mode = os.environ.get("LIGHTLY_PROGRESS_MODE", "auto").strip().casefold()
    enabled = disable is not True and progress_mode not in {
        "off", "disable", "disabled", "none",
    }
    stream = kwargs.get("file") or sys.stderr
    if enabled and _tqdm is not None and _isatty(stream):
        return _tqdm(iterable, disable=False, **kwargs)
    return _SnapshotProgress(iterable, enabled=enabled, **kwargs)


def write(message: str) -> None:
    """Write a message above an active progress bar when tqdm is available."""
    writer = getattr(_tqdm, "write", None) if _tqdm is not None else None
    if writer is None:
        print(message)
    else:
        writer(message)
