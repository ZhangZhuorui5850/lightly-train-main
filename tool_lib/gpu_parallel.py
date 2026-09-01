"""det / seg 共用的 GPU 调度内核：探测、负载过滤、选卡、分片、子进程编排。"""

from __future__ import annotations

import os
import shutil
import signal
import subprocess
import threading
import time
from pathlib import Path
from typing import Any, TextIO

from . import common as rt
from .progress import PROGRESS_EVENT_PREFIX, MultiShardProgress, plog

GPU_HIGH_MEMORY_RATIO_THRESHOLD = 0.75


def query_gpu_inventory() -> tuple[list[dict[str, Any]], str]:
    if rt.torch is None or not rt.torch.cuda.is_available():
        return [], "CUDA 当前不可用，进入单卡顺序模式。"

    nvidia_smi_path = shutil.which("nvidia-smi")
    if nvidia_smi_path is None:
        return [], "当前环境未找到 nvidia-smi，进入单卡顺序模式。"

    try:
        result = subprocess.run(
            [
                nvidia_smi_path,
                "--query-gpu=index,uuid,memory.used,memory.total,utilization.gpu",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            check=False,
            timeout=5,
        )
    except subprocess.TimeoutExpired:
        return [], "nvidia-smi 查询超过 5 秒，进入单卡顺序模式。"
    except OSError as exc:
        return [], f"nvidia-smi 无法执行: {exc}，进入单卡顺序模式。"
    if result.returncode != 0:
        message = result.stderr.strip() or result.stdout.strip() or "nvidia-smi 执行失败。"
        return [], f"{message} 进入单卡顺序模式。"

    gpus: list[dict[str, Any]] = []
    parse_errors: list[str] = []
    for raw_line in result.stdout.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        parts = [part.strip() for part in line.split(",")]
        if len(parts) < 5:
            parse_errors.append(line)
            continue
        try:
            index = int(parts[0])
            uuid = parts[1]
            memory_used = int(parts[2])
            memory_total = int(parts[3])
            utilization = int(parts[4])
        except ValueError:
            parse_errors.append(line)
            continue
        used_ratio = float(memory_used) / float(max(memory_total, 1))
        gpus.append(
            {
                "index": index,
                "uuid": uuid,
                "memory_used": memory_used,
                "memory_total": memory_total,
                "utilization": utilization,
                "used_ratio": used_ratio,
            }
        )

    gpus.sort(key=lambda item: (item["used_ratio"], item["memory_used"], item["utilization"], item["index"]))
    visible_devices_raw = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    if visible_devices_raw:
        visible_tokens = [token.strip() for token in visible_devices_raw.split(",") if token.strip()]
        rank = {token: index for index, token in enumerate(visible_tokens)}
        matched: list[dict[str, Any]] = []
        for gpu in gpus:
            physical = str(int(gpu["index"]))
            uuid = str(gpu.get("uuid", ""))
            token = next(
                (
                    item for item in visible_tokens
                    if item == physical or item == uuid or (item.startswith("GPU-") and uuid.startswith(item))
                ),
                None,
            )
            if token is not None:
                gpu["device_token"] = token
                matched.append(gpu)
        if not matched and any(token.startswith("MIG-") for token in visible_tokens):
            for logical_index, token in enumerate(visible_tokens):
                try:
                    free_bytes, total_bytes = rt.torch.cuda.mem_get_info(logical_index)
                except Exception:
                    continue
                total_mib = int(total_bytes // (1024 * 1024))
                free_mib = int(free_bytes // (1024 * 1024))
                matched.append(
                    {
                        "index": logical_index,
                        "uuid": token,
                        "device_token": token,
                        "memory_used": total_mib - free_mib,
                        "memory_total": total_mib,
                        "utilization": 0,
                        "used_ratio": float(total_mib - free_mib) / float(max(total_mib, 1)),
                    }
                )
        gpus = matched
        gpus.sort(key=lambda gpu: rank.get(str(gpu.get("device_token", "")), len(rank)))
        for logical_index, gpu in enumerate(gpus):
            gpu["logical_index"] = logical_index
    else:
        for gpu in gpus:
            gpu["logical_index"] = int(gpu["index"])
            gpu["device_token"] = str(int(gpu["index"]))
    if not gpus and parse_errors:
        return [], f"nvidia-smi 输出无法解析（{len(parse_errors)} 行），进入单卡顺序模式。"
    return gpus, ""


def filter_high_memory_gpus(gpus: list[dict[str, Any]]) -> list[dict[str, Any]]:
    min_free_mib = max(0, int(os.environ.get("LIGHTLY_GPU_MIN_FREE_MIB", "0") or 0))
    max_utilization = min(100, max(0, int(os.environ.get("LIGHTLY_GPU_MAX_UTILIZATION", "100") or 100)))
    return [
        gpu for gpu in gpus
        if float(gpu["used_ratio"]) < GPU_HIGH_MEMORY_RATIO_THRESHOLD
        and int(gpu["memory_total"]) - int(gpu["memory_used"]) >= min_free_mib
        and int(gpu["utilization"]) <= max_utilization
    ]


def format_gpu_summary(gpu: dict[str, float]) -> str:
    return (
        f"GPU {int(gpu['index'])}: "
        f"memory.used={int(gpu['memory_used'])}/{int(gpu['memory_total'])} MiB "
        f"({float(gpu['used_ratio']):.1%}), util={int(gpu['utilization'])}%"
    )


def select_fallback_single_gpu(
    gpus: list[dict[str, Any]], eligible_gpus: list[dict[str, Any]], *, component: str = "det/infer"
) -> tuple[str | None, str]:
    candidates = eligible_gpus if eligible_gpus else gpus
    if not candidates:
        return None, f"[{component}] 当前未检测到可用 GPU，进入默认顺序模式。"
    chosen = candidates[0]
    logical_index = int(chosen.get("logical_index", chosen["index"]))
    return f"cuda:{logical_index}", f"[{component}] 进入单卡顺序模式，使用 {format_gpu_summary(chosen)}"


def _forward_output(
    stream: TextIO | None,
    shard_index: int,
    progress: MultiShardProgress | None = None,
) -> None:
    if stream is None:
        return
    buffered_logs: list[str] = []

    def flush_logs() -> None:
        if buffered_logs:
            plog("\n".join(buffered_logs))
            buffered_logs.clear()

    for line in iter(stream.readline, ""):
        text = line.rstrip("\r\n")
        if not text:
            continue
        if progress is not None:
            progress.note_activity(shard_index)
        # 子进程进度协议行：路由给聚合进度条，不进入日志流。
        if progress is not None and text.startswith(PROGRESS_EVENT_PREFIX):
            flush_logs()
            progress.report_event(text[len(PROGRESS_EVENT_PREFIX):], shard=shard_index)
            continue
        buffered_logs.append(f"[shard {shard_index:02d}] {text}")
        if len(buffered_logs) >= 8:
            flush_logs()
    flush_logs()
    stream.close()


def _terminate_process(process: subprocess.Popen[str], *, force: bool = False) -> None:
    if process.poll() is not None:
        return
    try:
        if os.name != "nt" and getattr(process, "pid", None):
            os.killpg(os.getpgid(process.pid), signal.SIGKILL if force else signal.SIGTERM)
        elif force:
            process.kill()
        else:
            process.terminate()
    except (OSError, ProcessLookupError):
        pass


def run_sharded_subprocesses(
    jobs: list[tuple[int, int | str, list[str]]],
    *,
    cwd: str | Path,
    shard_totals: dict[int, int] | None = None,
) -> list[str]:
    """起一组子进程，每个 pin 到一张卡，等待全部完成，返回失败 shard 描述列表。

    jobs: (shard_index, gpu_index, command) 列表。
    子进程的终端进度被关闭(LIGHTLY_PROGRESS_MODE=off)，但通过
    LIGHTLY_PROGRESS_EVENTS=stdout 把各自图片进度上报回来，由
    MultiShardProgress 聚合成「总进度 + 每卡分片进度」展示。
    """
    processes: dict[int, subprocess.Popen[str]] = {}
    readers: dict[int, threading.Thread] = {}
    failed: list[str] = []
    progress = MultiShardProgress(
        len(jobs),
        label="multi-GPU",
        unit="shard",
        shard_totals=shard_totals,
        shard_labels={shard: str(gpu) for shard, gpu, _command in jobs},
    )
    try:
        for shard_index, gpu_index, command in jobs:
            child_env = os.environ.copy()
            child_env["CUDA_VISIBLE_DEVICES"] = str(gpu_index)
            child_env["LIGHTLY_PROGRESS_MODE"] = "off"
            child_env["LIGHTLY_PROGRESS_EVENTS"] = "stdout"
            child_env["LIGHTLY_PROGRESS_SCOPE"] = "inference"
            child_env["LIGHTLY_INTERNAL_SHARD"] = "1"
            child_env["PYTHONUNBUFFERED"] = "1"
            child_env["LIGHTLY_SHARD_INDEX"] = str(shard_index)
            child_env["LIGHTLY_NUM_SHARDS"] = str(len(jobs))
            popen_kwargs: dict[str, Any] = {
                "cwd": str(cwd), "env": child_env, "stdout": subprocess.PIPE,
                "stderr": subprocess.STDOUT, "text": True, "bufsize": 1,
            }
            if os.name == "nt":
                popen_kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
            else:
                popen_kwargs["start_new_session"] = True
            try:
                process = subprocess.Popen(command, **popen_kwargs)
            except TypeError:
                # Compatibility for lightweight test doubles and third-party wrappers.
                process = subprocess.Popen(command, cwd=str(cwd), env=child_env)
            processes[shard_index] = process
            if getattr(process, "stdout", None) is not None:
                reader = threading.Thread(
                    target=_forward_output, args=(process.stdout, shard_index, progress), daemon=True
                )
                reader.start()
                readers[shard_index] = reader

        pending = set(processes)
        abort_deadline: float | None = None
        last_heartbeat = 0.0
        while pending:
            progressed = False
            for shard_index in list(pending):
                process = processes[shard_index]
                if hasattr(process, "poll"):
                    return_code = process.poll()
                else:
                    return_code = process.wait()
                if return_code is None:
                    continue
                pending.remove(shard_index)
                reader = readers.get(shard_index)
                if reader is not None:
                    reader.join(timeout=2.0)
                progress.finish_shard(shard_index, success=return_code == 0)
                progressed = True
                if return_code != 0:
                    failed.append(f"shard_{shard_index:02d}(exit={return_code})")
                    for sibling_index in pending:
                        _terminate_process(processes[sibling_index])
                    abort_deadline = time.monotonic() + 5.0
            if failed and pending and abort_deadline is not None and time.monotonic() >= abort_deadline:
                for shard_index in pending:
                    _terminate_process(processes[shard_index], force=True)
                abort_deadline = time.monotonic() + 1.0
            now = time.monotonic()
            if now - last_heartbeat >= 1.0:
                progress.heartbeat()
                last_heartbeat = now
            if not progressed:
                time.sleep(0.05)
        return failed
    except BaseException:
        for process in processes.values():
            _terminate_process(process)
        deadline = time.monotonic() + 2.0
        while time.monotonic() < deadline and any(p.poll() is None for p in processes.values()):
            time.sleep(0.05)
        for process in processes.values():
            _terminate_process(process, force=True)
        raise
    finally:
        for reader in readers.values():
            reader.join(timeout=1.0)
        progress.close()


def filter_indices_for_shard(
    count: int, *, shard_index: int | None, num_shards: int,
    weights: list[float] | None = None,
) -> list[int]:
    num_shards = int(num_shards or 1)
    if num_shards < 1:
        raise ValueError(f"num_shards 必须 >= 1，实际为 {num_shards}")
    if shard_index is None:
        return list(range(count))
    shard_index = int(shard_index)
    if not 0 <= shard_index < num_shards:
        raise ValueError(
            f"shard_index 必须满足 0 <= shard_index < num_shards，实际为 {shard_index}/{num_shards}"
        )
    if num_shards == 1:
        return list(range(count))
    if weights is not None:
        if len(weights) != count:
            raise ValueError("weights 长度必须与 count 一致")
        assignments: list[list[int]] = [[] for _ in range(num_shards)]
        loads = [0.0] * num_shards
        for index in sorted(range(count), key=lambda item: (-float(weights[item]), item)):
            target = min(range(num_shards), key=lambda shard: (loads[shard], shard))
            assignments[target].append(index)
            loads[target] += max(0.0, float(weights[index]))
        return sorted(assignments[shard_index])
    return [i for i in range(count) if i % num_shards == shard_index]


def shard_item_totals(
    count: int,
    *,
    num_shards: int,
    weights: list[float] | None = None,
) -> dict[int, int]:
    """Return the exact item count assigned to every shard."""
    return {
        shard: len(
            filter_indices_for_shard(
                count,
                shard_index=shard,
                num_shards=num_shards,
                weights=weights,
            )
        )
        for shard in range(num_shards)
    }
