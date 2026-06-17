"""det / seg 共用的 GPU 调度内核：探测、负载过滤、选卡、分片、子进程编排。"""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path
from typing import Any

from . import common as rt

GPU_HIGH_MEMORY_RATIO_THRESHOLD = 0.75


def query_gpu_inventory() -> tuple[list[dict[str, float]], str]:
    if rt.torch is None or not rt.torch.cuda.is_available():
        return [], "CUDA 当前不可用，进入单卡顺序模式。"

    nvidia_smi_path = shutil.which("nvidia-smi")
    if nvidia_smi_path is None:
        return [], "当前环境未找到 nvidia-smi，进入单卡顺序模式。"

    result = subprocess.run(
        [
            nvidia_smi_path,
            "--query-gpu=index,memory.used,memory.total,utilization.gpu",
            "--format=csv,noheader,nounits",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        message = result.stderr.strip() or result.stdout.strip() or "nvidia-smi 执行失败。"
        return [], f"{message} 进入单卡顺序模式。"

    gpus: list[dict[str, float]] = []
    for raw_line in result.stdout.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        parts = [part.strip() for part in line.split(",")]
        if len(parts) < 4:
            continue
        try:
            index = int(parts[0])
            memory_used = int(parts[1])
            memory_total = int(parts[2])
            utilization = int(parts[3])
        except ValueError:
            continue
        used_ratio = float(memory_used) / float(max(memory_total, 1))
        gpus.append(
            {
                "index": index,
                "memory_used": memory_used,
                "memory_total": memory_total,
                "utilization": utilization,
                "used_ratio": used_ratio,
            }
        )

    gpus.sort(key=lambda item: (item["used_ratio"], item["memory_used"], item["utilization"], item["index"]))
    visible_devices_raw = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    if visible_devices_raw:
        visible_indices: set[int] = set()
        for token in visible_devices_raw.split(","):
            token = token.strip()
            if not token:
                continue
            try:
                visible_indices.add(int(token))
            except ValueError:
                continue
        if visible_indices:
            gpus = [gpu for gpu in gpus if int(gpu["index"]) in visible_indices]
    return gpus, ""


def filter_high_memory_gpus(gpus: list[dict[str, float]]) -> list[dict[str, float]]:
    return [gpu for gpu in gpus if float(gpu["used_ratio"]) < GPU_HIGH_MEMORY_RATIO_THRESHOLD]


def format_gpu_summary(gpu: dict[str, float]) -> str:
    return (
        f"GPU {int(gpu['index'])}: "
        f"memory.used={int(gpu['memory_used'])}/{int(gpu['memory_total'])} MiB "
        f"({float(gpu['used_ratio']):.1%}), util={int(gpu['utilization'])}%"
    )


def select_fallback_single_gpu(gpus: list[dict[str, float]], eligible_gpus: list[dict[str, float]]) -> tuple[str | None, str]:
    candidates = eligible_gpus if eligible_gpus else gpus
    if not candidates:
        return None, "[det/infer] 当前未检测到可用 GPU，进入默认顺序模式。"
    chosen = candidates[0]
    return f"cuda:{int(chosen['index'])}", f"[det/infer] 进入单卡顺序模式，使用 {format_gpu_summary(chosen)}"


def run_sharded_subprocesses(
    jobs: list[tuple[int, int, list[str]]],
    *,
    cwd: str | Path,
) -> list[str]:
    """起一组子进程，每个 pin 到一张卡，等待全部完成，返回失败 shard 描述列表。

    jobs: (shard_index, gpu_index, command) 列表。
    """
    processes: list[tuple[int, subprocess.Popen[str]]] = []
    for shard_index, gpu_index, command in jobs:
        child_env = os.environ.copy()
        child_env["CUDA_VISIBLE_DEVICES"] = str(int(gpu_index))
        process = subprocess.Popen(command, cwd=str(cwd), env=child_env)
        processes.append((shard_index, process))

    failed: list[str] = []
    for shard_index, process in processes:
        return_code = process.wait()
        if return_code != 0:
            failed.append(f"shard_{shard_index:02d}(exit={return_code})")
    return failed


def filter_indices_for_shard(count: int, *, shard_index: int | None, num_shards: int) -> list[int]:
    num_shards = int(num_shards or 1)
    if shard_index is None or num_shards <= 1:
        return list(range(count))
    shard_index = int(shard_index)
    return [i for i in range(count) if i % num_shards == shard_index]
