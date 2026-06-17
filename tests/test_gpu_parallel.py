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


def test_filter_indices_for_shard_round_robin():
    assert gpu_parallel.filter_indices_for_shard(7, shard_index=0, num_shards=3) == [0, 3, 6]
    assert gpu_parallel.filter_indices_for_shard(7, shard_index=1, num_shards=3) == [1, 4]
    assert gpu_parallel.filter_indices_for_shard(7, shard_index=2, num_shards=3) == [2, 5]


def test_filter_indices_for_shard_single_shard_returns_all():
    assert gpu_parallel.filter_indices_for_shard(4, shard_index=None, num_shards=1) == [0, 1, 2, 3]


def test_run_sharded_subprocesses_reports_failures(monkeypatch):
    calls = []

    class _FakeProc:
        def __init__(self, returncode):
            self._rc = returncode
        def wait(self):
            return self._rc

    def _fake_popen(command, cwd, env):
        calls.append((command, env.get("CUDA_VISIBLE_DEVICES")))
        rc = 1 if "shard-1" in command else 0
        return _FakeProc(rc)

    monkeypatch.setattr(gpu_parallel.subprocess, "Popen", _fake_popen)
    jobs = [
        (0, 3, ["python", "x", "shard-0"]),
        (1, 5, ["python", "x", "shard-1"]),
    ]
    failed = gpu_parallel.run_sharded_subprocesses(jobs, cwd="/tmp")
    assert failed == ["shard_01(exit=1)"]
    assert calls[0][1] == "3" and calls[1][1] == "5"
