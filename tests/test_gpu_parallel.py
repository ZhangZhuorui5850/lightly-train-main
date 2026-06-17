from __future__ import annotations

from tool_lib import gpu_parallel


def _gpu(index, used_ratio, memory_used=0, memory_total=10000, utilization=0):
    return {
        "index": index,
        "memory_used": memory_used,
        "memory_total": memory_total,
        "utilization": utilization,
        "used_ratio": used_ratio,
    }


def test_filter_high_memory_gpus_drops_busy_cards():
    gpus = [_gpu(0, 0.10), _gpu(1, 0.80), _gpu(2, 0.50)]
    eligible = gpu_parallel.filter_high_memory_gpus(gpus)
    assert [g["index"] for g in eligible] == [0, 2]


def test_select_fallback_single_gpu_picks_first_eligible():
    gpus = [_gpu(0, 0.10), _gpu(1, 0.05)]
    eligible = [gpus[1]]
    device, message = gpu_parallel.select_fallback_single_gpu(gpus, eligible)
    assert device == "cuda:1"
    assert "单卡顺序模式" in message


def test_select_fallback_single_gpu_handles_empty():
    device, message = gpu_parallel.select_fallback_single_gpu([], [])
    assert device is None
    assert "未检测到可用 GPU" in message
