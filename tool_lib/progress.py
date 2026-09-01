"""统一的单行进度条工具(固定一行,\\r 原地刷新,不刷屏)。

设计目标(对齐用户诉求「固定在最下面一行,不要一直往上刷新占满页面」):
  * 交互终端(TTY): 用 tqdm 单行进度条,\\r 原地刷新,始终只占一行。
  * 输出重定向到日志(非 TTY): 自动降级为周期性打印,避免刷爆日志文件。
  * 未安装 tqdm: 退化为轻量自实现的 \\r 单行条,功能不受影响。

多卡分片进度(gpu_parallel 启动的子进程):
  * 子进程终端不画条(LIGHTLY_PROGRESS_MODE=off)，改由 track() 把各自进度
    以协议行 `@@LT_PROGRESS@@{"shard":i,"phase":p,"done":n,"total":t}` 写回 stdout；
  * 父进程用 MultiShardProgress 聚合：一条总进度条 + postfix 里每个分片
    各自的图片进度(例如 s0=230/500 s1=✓)，满足「看到每张卡各跑到哪」。
  * 上报频率由 LIGHTLY_PROGRESS_EMIT_INTERVAL 控制(秒，默认 0.5)。

用法:
    from tool_lib.progress import track, plog
    for x in track(items, label="det/infer 推理", unit="img"):
        ...
        plog("某条需要保留的日志")   # 不会打断底部进度条
"""
from __future__ import annotations

import json
import itertools
import shutil
import sys
import threading
import time
import os
from typing import Any, Iterable, Iterator, Optional, TypeVar

_T = TypeVar("_T")

try:
    from tqdm import tqdm as _tqdm
except ModuleNotFoundError:  # pragma: no cover
    _tqdm = None


def _isatty(stream=None) -> bool:
    stream = sys.stdout if stream is None else stream
    try:
        return bool(stream.isatty())
    except Exception:  # pragma: no cover
        return False


# 子进程进度协议行前缀；父进程(gpu_parallel._forward_output)按它识别并路由。
PROGRESS_EVENT_PREFIX = "@@LT_PROGRESS@@"
_PHASE_IDS = itertools.count()


class _ShardEventEmitter:
    """分片子进程侧：把 track() 的进度以协议行写回父进程 stdout。

    仅当父进程设置了 LIGHTLY_PROGRESS_EVENTS=stdout 且带 LIGHTLY_SHARD_INDEX
    时生效；本地终端显示仍由 LIGHTLY_PROGRESS_MODE 控制(分片子进程为 off，
    不画条、只上报)。管道断裂时静默停发，不影响计算本身。
    """

    def __init__(self, total: Optional[int], *, label: str, aggregate: str) -> None:
        self.shard = os.environ.get("LIGHTLY_SHARD_INDEX", "").strip()
        self.total = total
        self.label = label
        self.aggregate = aggregate
        self.phase = next(_PHASE_IDS)
        try:
            self._interval = float(os.environ.get("LIGHTLY_PROGRESS_EMIT_INTERVAL", "0.5") or 0.5)
        except ValueError:
            self._interval = 0.5
        self._interval = max(0.05, self._interval)
        self._last_time = 0.0
        self._last_sent = -1
        self._closed = False

    def emit(self, done: int) -> None:
        if self._closed:
            return
        now = time.monotonic()
        # 节流：迭代收尾由 finish() 立即上报。
        if (now - self._last_time) < self._interval:
            return
        if done == self._last_sent:
            return
        self._send(done, status="running")

    def finish(self, count: int, *, status: str) -> None:
        """迭代结束时上报最终计数和完成状态。"""
        if self._closed:
            return
        self._send(count, status=status)
        self._closed = True

    def _send(self, done: int, *, status: str) -> None:
        payload = json.dumps(
            {
                "shard": self.shard,
                "phase": self.phase,
                "label": self.label,
                "aggregate": self.aggregate,
                "done": int(done),
                "total": self.total,
                "status": status,
            },
            separators=(",", ":"),
        )
        try:
            sys.stdout.write(f"{PROGRESS_EVENT_PREFIX}{payload}\n")
            sys.stdout.flush()
        except (OSError, ValueError):
            self._closed = True
        self._last_time = time.monotonic()
        self._last_sent = int(done)


def _make_shard_emitter(
    total: Optional[int], *, label: str, aggregate: str, shard_scope: str | None,
) -> Optional[_ShardEventEmitter]:
    mode = os.environ.get("LIGHTLY_PROGRESS_EVENTS", "").strip().casefold()
    if mode not in {"stdout", "1", "true"}:
        return None
    if not os.environ.get("LIGHTLY_SHARD_INDEX", "").strip():
        return None
    expected_scope = os.environ.get("LIGHTLY_PROGRESS_SCOPE", "").strip()
    if expected_scope and shard_scope != expected_scope:
        return None
    return _ShardEventEmitter(total, label=label, aggregate=aggregate)


class _FallbackBar:
    """无 tqdm 时的极简单行进度条: [####....] cur/total。"""

    def __init__(self, label: str, total: Optional[int], width: int = 28) -> None:
        self.label = label
        self.total = total
        self.width = width
        self._last = 0.0
        self._tty = _isatty()
        self._line_open = False
        self._current = 0
        self._last_printed_current = -1

    def update(self, current: int, *, force: bool = False) -> None:
        self._current = current
        now = time.monotonic()
        done = self.total is not None and current >= self.total
        # TTY 上一律 \r 单行;非 TTY 上每 0.8s 提交一行,便于日志留痕。
        if not force and not done and not self._tty and (now - self._last) < 0.8:
            return
        if self.total:
            filled = int(self.width * min(current, self.total) / self.total)
            bar = "#" * filled + "." * (self.width - filled)
            body = f"[{bar}] {min(current, self.total)}/{self.total}"
        else:
            body = f"{current}"
        end = "\n" if (done or not self._tty) else "\r"
        print(f"  {self.label} {body}", end=end, flush=True)
        self._line_open = end == "\r"
        self._last_printed_current = current
        self._last = now

    def close(self) -> None:
        if not self._tty and self._last_printed_current != self._current:
            self.update(self._current, force=True)
        if self._tty and self._line_open:
            print("", flush=True)  # 收尾换行,避免后续输出粘在进度条后
            self._line_open = False


def track(
    iterable: Iterable[_T],
    *,
    label: str = "",
    total: Optional[int] = None,
    unit: str = "it",
    enable: bool = True,
    aggregate: str = "primary",
    shard_scope: str | None = None,
) -> Iterator[_T]:
    """遍历 iterable 并显示固定单行进度条。

    total 未给时会尝试 len(iterable);对生成器请显式传 total。
    enable=False 可一键关闭(例如非交互批处理时)。

    分片子进程(LIGHTLY_PROGRESS_EVENTS=stdout)额外把进度以协议行上报给
    父进程：本地不画条，进度由父进程的 MultiShardProgress 聚合展示。
    shard_scope 用于隔离推理进度与目录扫描等其他 track 调用。
    """
    if total is None:
        try:
            total = len(iterable)  # type: ignore[arg-type]
        except (TypeError, AttributeError):
            total = None

    if aggregate not in {"primary", "extra"}:
        raise ValueError(f"aggregate must be 'primary' or 'extra', got {aggregate!r}")
    emitter = _make_shard_emitter(
        total, label=label, aggregate=aggregate, shard_scope=shard_scope,
    )
    progress_mode = os.environ.get("LIGHTLY_PROGRESS_MODE", "auto").strip().casefold()
    display_off = not enable or progress_mode in {"off", "disable", "disabled", "none"}

    if emitter is None:
        if display_off:
            yield from iterable
            return
        if _tqdm is not None and _isatty(sys.stderr):
            wrapped = _tqdm(
                iterable, desc=label, total=total, unit=unit,
                dynamic_ncols=True, leave=True, mininterval=0.2,
            )
            try:
                yield from wrapped
            finally:
                wrapped.close()
            return
        bar = _FallbackBar(label, total)
        try:
            for i, item in enumerate(iterable, 1):
                yield item
                bar.update(i)
        finally:
            bar.close()
        return

    # 分片上报路径：本地显示(通常关闭)与事件上报并行推进。
    tqdm_display = (not display_off) and _tqdm is not None and _isatty(sys.stderr)
    if tqdm_display:
        wrapped: Iterable[_T] = _tqdm(
            iterable, desc=label, total=total, unit=unit,
            dynamic_ncols=True, leave=True, mininterval=0.2,
        )
        fallback: Optional[_FallbackBar] = None
    elif not display_off:
        fallback = _FallbackBar(label, total)
        wrapped = iterable
    else:
        fallback = None
        wrapped = iterable
    count = 0
    outcome = "cancelled"
    try:
        for item in wrapped:
            yield item
            count += 1
            if fallback is not None:
                fallback.update(count)
            emitter.emit(count)
        outcome = "complete"
    except GeneratorExit:
        outcome = "cancelled"
        raise
    except BaseException:
        outcome = "failed"
        raise
    finally:
        if fallback is not None:
            fallback.close()
        emitter.finish(count, status=outcome)


class ProgressReporter:
    """A parent-owned manual counter used by multi-process supervisors."""

    def __init__(self, total: int, *, label: str, unit: str = "it") -> None:
        self.total = max(0, int(total))
        self.current = 0
        self._tqdm_bar = None
        self._fallback = None
        if os.environ.get("LIGHTLY_PROGRESS_MODE", "auto").strip().casefold() in {
            "off", "disable", "disabled", "none"
        }:
            return
        if _tqdm is not None and _isatty(sys.stderr):
            self._tqdm_bar = _tqdm(
                total=self.total, desc=label, unit=unit, dynamic_ncols=True,
                leave=True, mininterval=0.2,
            )
        else:
            self._fallback = _FallbackBar(label, self.total)

    def update(self, amount: int = 1) -> None:
        amount = max(0, int(amount))
        previous = self.current
        self.current = min(self.total, self.current + amount)
        applied = self.current - previous
        if self._tqdm_bar is not None:
            self._tqdm_bar.update(applied)
        elif self._fallback is not None:
            self._fallback.update(self.current)

    def close(self) -> None:
        if self._tqdm_bar is not None:
            self._tqdm_bar.close()
        elif self._fallback is not None:
            if self.current < self.total:
                self._fallback.update(self.current)
            self._fallback.close()


class MultiShardProgress:
    """多卡分片进度：一条合计进度条 + 每张卡一条固定进度条。

    由父进程持有；子进程经 stdout 协议行上报 phase/done/total/status。TTY 默认用
    ``position`` 固定多行，例如一条合计行加四条 GPU 行。显式设置
    ``LIGHTLY_PROGRESS_LAYOUT=compact`` 时使用单行合计 + 紧凑逐卡后缀。
    收不到任何子进程上报时(旧版子进程/秒级小任务)自动退化为按「分片完成数」
    计数，与旧版 ProgressReporter 行为一致。
    """

    def __init__(
        self,
        num_shards: int,
        *,
        label: str = "multi-GPU",
        unit: str = "shard",
        shard_totals: dict[int, int] | None = None,
        shard_labels: dict[int, str] | None = None,
    ) -> None:
        self.num_shards = max(0, int(num_shards))
        self._label = label
        self._unit = unit
        self._shard_labels = {
            int(shard): str(value) for shard, value in (shard_labels or {}).items()
        }
        self._states: dict[int, dict[str, object]] = {}
        for shard, total in (shard_totals or {}).items():
            if 0 <= int(shard) < self.num_shards:
                self._states[int(shard)] = self._new_state(total=max(0, int(total)))
        self._finished: dict[int, bool] = {}
        self._image_mode = self._all_totals_known()
        self._lock = threading.RLock()
        self._tqdm_bar = None
        self._shard_bars: dict[int, Any] = {}
        self._card_layout = False
        self._fallback_enabled = False
        self._fallback_last = 0.0
        if os.environ.get("LIGHTLY_PROGRESS_MODE", "auto").strip().casefold() in {
            "off", "disable", "disabled", "none"
        }:
            return
        if _tqdm is not None and _isatty(sys.stderr):
            layout = os.environ.get("LIGHTLY_PROGRESS_LAYOUT", "cards").strip().casefold()
            try:
                max_card_bars = max(
                    1, int(os.environ.get("LIGHTLY_PROGRESS_MAX_CARD_BARS", "8") or 8)
                )
            except ValueError:
                max_card_bars = 8
            self._card_layout = (
                layout not in {"compact", "single", "one-line"}
                and 1 < self.num_shards <= max_card_bars
            )
            initial_total = (
                sum(int(state["total"]) for state in self._states.values())
                if self._image_mode
                else self.num_shards
            )
            self._tqdm_bar = _tqdm(
                total=initial_total,
                desc=f"{label} 图片合计" if self._image_mode else label,
                unit="img" if self._image_mode else unit,
                dynamic_ncols=True,
                leave=True,
                mininterval=0.2,
                position=0,
            )
            if self._card_layout:
                for shard in range(self.num_shards):
                    state = self._states.get(shard, {})
                    total = int(state.get("total", 0))
                    self._shard_bars[shard] = _tqdm(
                        total=total or None,
                        desc=self._shard_description(shard),
                        unit="img",
                        dynamic_ncols=True,
                        leave=True,
                        mininterval=0.2,
                        position=shard + 1,
                    )
        else:
            self._fallback_enabled = True

    # ---- 子进程上报入口 ----

    def report_event(self, payload: str, *, shard: int) -> None:
        """解析一条协议行 payload(不含前缀)；解析失败静默忽略。"""
        try:
            data = json.loads(payload)
            done = max(0, int(data["done"]))
            raw_total = data.get("total")
            total = max(0, int(raw_total)) if raw_total is not None else None
            phase = int(data.get("phase", 0))
            label = str(data.get("label", ""))
            status = str(data.get("status", "running"))
            aggregate = str(data.get("aggregate", "primary"))
        except (ValueError, TypeError, KeyError):
            return
        self.update_shard(
            shard,
            done,
            total,
            phase=phase,
            label=label,
            status=status,
            aggregate=aggregate,
        )

    @staticmethod
    def _new_state(*, total: int = 0) -> dict[str, object]:
        return {
            "done": 0,
            "total": total,
            "offset": 0,
            "phase": None,
            "phase_done": 0,
            "phase_total": 0,
            "phase_status": "pending",
            "label": "",
            "phase_aggregate": "primary",
            "reported": False,
            "updated_at": time.monotonic(),
            "progress_mismatch": False,
        }

    def _all_totals_known(self) -> bool:
        return self.num_shards > 0 and all(
            int(self._states.get(shard, {}).get("total", 0)) > 0
            for shard in range(self.num_shards)
        )

    def update_shard(
        self,
        shard: int,
        done: int,
        total: Optional[int],
        *,
        phase: int = 0,
        label: str = "",
        status: str = "running",
        aggregate: str = "primary",
    ) -> None:
        with self._lock:
            if not 0 <= shard < self.num_shards or shard in self._finished:
                return
            state = self._states.setdefault(shard, self._new_state())
            previous_phase = state["phase"]
            if aggregate not in {"primary", "extra"}:
                aggregate = "primary"
            if previous_phase is None and aggregate == "extra" and total:
                state["total"] = int(state["total"]) + int(total)
            if previous_phase is not None and phase != previous_phase:
                previous_total = int(state["phase_total"])
                previous_done = int(state["phase_done"])
                state["offset"] = int(state["offset"]) + (
                    previous_total
                    if state["phase_status"] == "complete" and previous_total > 0
                    else previous_done
                )
                state["phase_done"] = 0
                state["phase_total"] = 0
                if aggregate == "extra" and total:
                    state["total"] = int(state["total"]) + int(total)
            state["phase"] = phase
            state["phase_done"] = done
            state["phase_total"] = total or 0
            state["phase_status"] = status
            state["label"] = label
            state["phase_aggregate"] = aggregate
            state["reported"] = True
            state["updated_at"] = time.monotonic()
            cumulative_done = int(state["offset"]) + done
            known_total = int(state["total"])
            if known_total <= 0:
                known_total = int(state["offset"]) + int(total or 0)
                state["total"] = known_total
            state["done"] = min(cumulative_done, known_total) if known_total else cumulative_done
            self._image_mode = self._all_totals_known()
            self._render()

    def finish_shard(self, shard: int, *, success: bool) -> None:
        """记录分片进程最终状态；成功分片补齐，失败分片保留实际进度。"""
        with self._lock:
            if not 0 <= shard < self.num_shards or shard in self._finished:
                return
            self._finished[shard] = bool(success)
            state = self._states.get(shard)
            if success and state is not None and int(state["total"]) > 0:
                if bool(state.get("reported")):
                    state["progress_mismatch"] = int(state["done"]) != int(state["total"])
                else:
                    state["done"] = state["total"]
            if state is not None:
                state["updated_at"] = time.monotonic()
            self._render()

    def note_activity(self, shard: int) -> None:
        """Record non-progress child output so startup and stall ages stay accurate."""
        with self._lock:
            state = self._states.get(shard)
            if state is not None and shard not in self._finished:
                state["updated_at"] = time.monotonic()

    def heartbeat(self) -> None:
        """Refresh elapsed startup/stall information while child processes are quiet."""
        with self._lock:
            self._render(force=True)

    def close(self) -> None:
        if self._tqdm_bar is not None:
            self._render(force=True)
            for shard in reversed(range(self.num_shards)):
                bar = self._shard_bars.get(shard)
                if bar is not None:
                    bar.close()
            self._tqdm_bar.close()
        elif self._fallback_enabled:
            self._fallback_print(final=True)

    # ---- 内部：聚合与渲染 ----

    def _aggregates(self) -> tuple[int, int]:
        done_sum = sum(int(state["done"]) for state in self._states.values())
        total_sum = sum(int(state["total"]) for state in self._states.values())
        return done_sum, total_sum

    def _shard_description(self, shard: int) -> str:
        gpu = self._shard_labels.get(shard)
        return f"GPU {gpu} [s{shard}]" if gpu is not None else f"shard {shard}"

    def _shard_status(self, shard: int, *, now: float) -> str:
        if shard in self._finished:
            state = self._states.get(shard, {})
            if not self._finished[shard]:
                return "失败"
            if bool(state.get("progress_mismatch")):
                return f"完成·计数异常 {int(state.get('done', 0))}/{int(state.get('total', 0))}"
            return "完成"
        state = self._states.get(shard)
        if state is None:
            return "等待启动"
        done = int(state.get("done", 0))
        total = int(state.get("total", 0))
        if (
            total > 0
            and done >= total
            and state.get("phase_status") == "complete"
        ):
            return "图片完成·收尾中"
        idle_seconds = max(0, int(now - float(state.get("updated_at", now))))
        if done == 0 and idle_seconds >= 5:
            return f"启动中 {idle_seconds}s"
        if idle_seconds >= 30:
            return f"等待输出 {idle_seconds}s"
        label = str(state.get("label", "")).strip().split("/")[-1]
        return label[-18:] if label else "准备中"

    def _postfix(self, *, max_width: int | None = None) -> str:
        parts: list[str] = []
        compact_parts: list[str] = []
        percent_parts: list[str] = []
        now = time.monotonic()
        for shard in range(self.num_shards):
            if shard in self._finished:
                state = self._states.get(shard, {})
                mismatch = bool(state.get("progress_mismatch"))
                if mismatch:
                    done = int(state.get("done", 0))
                    total = int(state.get("total", 0))
                    item = f"s{shard}=✓{done}/{total}"
                    parts.append(item)
                    compact_parts.append(item)
                    percent = round(100 * done / total) if total else 0
                    percent_parts.append(f"{shard}=!{percent}%")
                else:
                    mark = "✓" if self._finished[shard] else "✗"
                    parts.append(f"s{shard}={mark}")
                    compact_parts.append(f"s{shard}={mark}")
                    percent_parts.append(f"{shard}={mark}")
            elif shard in self._states:
                state = self._states[shard]
                done, total = int(state["done"]), int(state["total"])
                tailing = (
                    total > 0
                    and done >= total
                    and state.get("phase_status") == "complete"
                )
                phase_label = str(state.get("label", "")).strip().split("/")[-1]
                phase_label = phase_label[-12:]
                idle_seconds = max(0, int(now - float(state.get("updated_at", now))))
                if tailing:
                    phase_label = "图片完成·收尾中"
                elif done == 0 and idle_seconds >= 5:
                    phase_label = f"启动{idle_seconds}s"
                elif idle_seconds >= 30:
                    phase_label = f"等待{idle_seconds}s"
                suffix = f":{phase_label}" if phase_label else ""
                parts.append(f"s{shard}={done}/{total}{suffix}" if total else f"s{shard}={done}{suffix}")
                compact_tail = "·收尾" if tailing else ""
                compact_parts.append(
                    f"s{shard}={done}/{total}{compact_tail}" if total else f"s{shard}={done}"
                )
                if done == 0 and idle_seconds >= 5:
                    percent_parts.append(f"{shard}=启{idle_seconds}s")
                elif idle_seconds >= 30:
                    percent_parts.append(f"{shard}=等{idle_seconds}s")
                elif total:
                    suffix = (
                        "·收尾"
                        if tailing
                        else ""
                    )
                    percent_parts.append(f"{shard}={round(100 * done / total)}%{suffix}")
                else:
                    percent_parts.append(f"{shard}={done}")
            else:
                parts.append(f"s{shard}=…")
                compact_parts.append(f"s{shard}=…")
                percent_parts.append(f"{shard}=…")
        if max_width is None:
            return " ".join(parts)
        for candidate_parts in (parts, compact_parts, percent_parts):
            if len(" ".join(candidate_parts)) <= max_width:
                return " ".join(candidate_parts)
        parts = percent_parts
        rendered: list[str] = []
        used = 0
        for index, part in enumerate(parts):
            remaining = len(parts) - index
            reserve = len(f" +{remaining}") if remaining else 0
            if not rendered and len(part) + reserve > max_width:
                available = max(1, max_width - reserve)
                part = (
                    part[:available]
                    if len(part) <= available
                    else part[: max(1, available - 1)] + "…"
                )
            addition = len(part) + (1 if rendered else 0)
            if rendered and used + addition + reserve > max_width:
                rendered.append(f"+{remaining}")
                break
            rendered.append(part)
            used += addition
        return " ".join(rendered)

    def _render(self, *, force: bool = False) -> None:
        if self._tqdm_bar is not None:
            bar = self._tqdm_bar
            if self._image_mode:
                if bar.unit != "img":
                    bar.unit = "img"
                    bar.set_description_str(f"{self._label} 图片合计", refresh=False)
                done_sum, total_sum = self._aggregates()
                bar.total = total_sum
                bar.n = done_sum
            else:
                bar.n = sum(1 for success in self._finished.values() if success)
            if self._card_layout:
                completed = sum(1 for success in self._finished.values() if success)
                bar.set_postfix_str(
                    f"完成卡 {completed}/{self.num_shards}", refresh=False
                )
                now = time.monotonic()
                for shard, shard_bar in self._shard_bars.items():
                    state = self._states.get(shard, {})
                    total = int(state.get("total", 0))
                    done = int(state.get("done", 0))
                    shard_bar.total = total or None
                    shard_bar.n = done
                    shard_bar.set_postfix_str(
                        self._shard_status(shard, now=now), refresh=False
                    )
                    if force:
                        shard_bar.refresh()
                    else:
                        shard_bar.update(0)
            else:
                columns = shutil.get_terminal_size((100, 20)).columns
                bar.set_postfix_str(
                    self._postfix(max_width=max(24, columns - 44)),
                    refresh=False,
                )
            # update(0) 走 tqdm 自身的 mininterval 节流，避免高频重绘。
            if force:
                bar.refresh()
            else:
                bar.update(0)
        elif self._fallback_enabled:
            self._fallback_print(final=False)

    def _fallback_print(self, final: bool) -> None:
        now = time.monotonic()
        if not final and (now - self._fallback_last) < 5.0:
            return
        self._fallback_last = now
        if self._image_mode:
            done_sum, total_sum = self._aggregates()
            head = f"{self._label} 合计 {done_sum}/{total_sum} img"
        else:
            succeeded = sum(1 for success in self._finished.values() if success)
            head = f"{self._label} {succeeded}/{self.num_shards} {self._unit}"
        print(f"  {head} | {self._postfix()}", flush=True)


def plog(message: str) -> None:
    """在进度条运行期间安全打印一条日志(不打断底部进度条)。"""
    if _tqdm is not None:
        _tqdm.write(message)
    else:  # pragma: no cover
        print(message, flush=True)
