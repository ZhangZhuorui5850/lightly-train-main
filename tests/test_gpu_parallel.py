from __future__ import annotations

import json
import sys
import time
from types import SimpleNamespace

import pytest

from tool_lib import gpu_parallel, progress


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


def test_visible_physical_gpu_maps_to_parent_logical_zero(monkeypatch):
    fake_torch = SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: True))
    monkeypatch.setattr(gpu_parallel.rt, "torch", fake_torch)
    monkeypatch.setattr(gpu_parallel.shutil, "which", lambda _name: "/usr/bin/nvidia-smi")
    monkeypatch.setattr(
        gpu_parallel.subprocess,
        "run",
        lambda *_args, **_kwargs: SimpleNamespace(
            returncode=0, stdout="3, GPU-abcdef, 100, 10000, 0\n", stderr=""
        ),
    )
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "3")
    gpus, message = gpu_parallel.query_gpu_inventory()
    assert message == ""
    assert gpus[0]["device_token"] == "3"
    device, _ = gpu_parallel.select_fallback_single_gpu(gpus, gpus)
    assert device == "cuda:0"


def test_filter_indices_for_shard_round_robin():
    assert gpu_parallel.filter_indices_for_shard(7, shard_index=0, num_shards=3) == [0, 3, 6]
    assert gpu_parallel.filter_indices_for_shard(7, shard_index=1, num_shards=3) == [1, 4]
    assert gpu_parallel.filter_indices_for_shard(7, shard_index=2, num_shards=3) == [2, 5]


def test_filter_indices_for_shard_single_shard_returns_all():
    assert gpu_parallel.filter_indices_for_shard(4, shard_index=None, num_shards=1) == [0, 1, 2, 3]


def test_filter_indices_for_shard_rejects_invalid_index():
    with pytest.raises(ValueError, match="shard_index"):
        gpu_parallel.filter_indices_for_shard(4, shard_index=2, num_shards=2)


def test_weighted_shards_balance_large_items():
    assignments = [
        gpu_parallel.filter_indices_for_shard(
            4, shard_index=index, num_shards=2, weights=[10, 9, 1, 1]
        )
        for index in range(2)
    ]
    loads = [sum([10, 9, 1, 1][item] for item in assignment) for assignment in assignments]
    assert sorted(item for assignment in assignments for item in assignment) == [0, 1, 2, 3]
    assert max(loads) - min(loads) <= 1


def test_shard_item_totals_match_weighted_assignments():
    totals = gpu_parallel.shard_item_totals(
        7, num_shards=3, weights=[10, 9, 8, 2, 2, 1, 1]
    )
    assert sum(totals.values()) == 7
    assert set(totals) == {0, 1, 2}


def test_multi_shard_progress_accumulates_multiple_phases(monkeypatch):
    monkeypatch.setenv("LIGHTLY_PROGRESS_MODE", "off")
    reporter = progress.MultiShardProgress(1, shard_totals={0: 5})
    reporter.update_shard(0, 2, 2, phase=10, label="val", status="complete")
    reporter.update_shard(0, 1, 3, phase=11, label="test", status="running")

    assert reporter._aggregates() == (3, 5)
    assert reporter._postfix() == "s0=3/5:test"


def test_multi_shard_progress_preserves_failed_count(monkeypatch):
    monkeypatch.setenv("LIGHTLY_PROGRESS_MODE", "off")
    reporter = progress.MultiShardProgress(1, shard_totals={0: 5})
    reporter.update_shard(0, 2, 5, phase=1, status="failed")
    reporter.finish_shard(0, success=False)

    assert reporter._aggregates() == (2, 5)
    assert reporter._postfix() == "s0=✗"


def test_multi_shard_progress_exposes_successful_shard_count_mismatch(monkeypatch):
    monkeypatch.setenv("LIGHTLY_PROGRESS_MODE", "off")
    reporter = progress.MultiShardProgress(1, shard_totals={0: 5})
    reporter.update_shard(0, 4, 5, phase=1, status="complete")
    reporter.finish_shard(0, success=True)

    assert reporter._aggregates() == (4, 5)
    assert reporter._postfix() == "s0=✓4/5"


def test_multi_shard_progress_extends_total_for_extra_phase(monkeypatch):
    monkeypatch.setenv("LIGHTLY_PROGRESS_MODE", "off")
    reporter = progress.MultiShardProgress(1, shard_totals={0: 5})
    reporter.update_shard(0, 5, 5, phase=1, label="eval", status="complete")
    reporter.update_shard(
        0, 1, 2, phase=2, label="visualize", status="running", aggregate="extra"
    )

    assert reporter._aggregates() == (6, 7)
    assert reporter._postfix() == "s0=6/7:visualize"


def test_multi_shard_progress_extends_total_when_extra_is_first_event(monkeypatch):
    monkeypatch.setenv("LIGHTLY_PROGRESS_MODE", "off")
    reporter = progress.MultiShardProgress(1, shard_totals={0: 5})

    reporter.update_shard(
        0, 1, 2, phase=2, label="visualize", status="running", aggregate="extra"
    )

    assert reporter._aggregates() == (1, 7)


def test_multi_shard_progress_compacts_long_postfix(monkeypatch):
    monkeypatch.setenv("LIGHTLY_PROGRESS_MODE", "off")
    reporter = progress.MultiShardProgress(8, shard_totals={index: 100 for index in range(8)})
    for index in range(8):
        reporter.update_shard(index, index * 10, 100, phase=1, label="long-phase-name")

    rendered = reporter._postfix(max_width=32)

    assert len(rendered) <= 32
    assert "+" in rendered


def test_multi_shard_progress_shows_startup_and_quiet_ages(monkeypatch):
    monkeypatch.setenv("LIGHTLY_PROGRESS_MODE", "off")
    clock = [100.0]
    monkeypatch.setattr(progress.time, "monotonic", lambda: clock[0])
    reporter = progress.MultiShardProgress(1, shard_totals={0: 10})

    clock[0] = 106.0
    assert "启动6s" in reporter._postfix()

    reporter.update_shard(0, 1, 10, phase=1, label="infer")
    clock[0] = 140.0
    assert "等待34s" in reporter._postfix()


def test_multi_shard_progress_keeps_every_card_visible_in_compact_mode(monkeypatch):
    monkeypatch.setenv("LIGHTLY_PROGRESS_MODE", "off")
    reporter = progress.MultiShardProgress(
        7, shard_totals={index: 2500 for index in range(7)}
    )
    for index in range(7):
        reporter.update_shard(index, 100 * (index + 1), 2500, phase=1, label="det/infer 推理")

    rendered = reporter._postfix(max_width=60)

    assert "+" not in rendered
    assert all(f"{index}=" in rendered for index in range(7))


def test_multi_shard_progress_creates_one_fixed_bar_per_gpu(monkeypatch):
    created = []

    class _FakeBar:
        def __init__(self, **kwargs):
            self.total = kwargs["total"]
            self.desc = kwargs["desc"]
            self.unit = kwargs["unit"]
            self.position = kwargs["position"]
            self.n = 0
            self.postfix = ""
            self.closed = False
            created.append(self)

        def set_description_str(self, value, refresh=False):
            self.desc = value

        def set_postfix_str(self, value, refresh=False):
            self.postfix = value

        def update(self, _amount=0):
            pass

        def refresh(self):
            pass

        def close(self):
            self.closed = True

    monkeypatch.setenv("LIGHTLY_PROGRESS_MODE", "auto")
    monkeypatch.setenv("LIGHTLY_PROGRESS_LAYOUT", "cards")
    monkeypatch.setattr(progress, "_isatty", lambda _stream=None: True)
    monkeypatch.setattr(progress, "_tqdm", _FakeBar)
    reporter = progress.MultiShardProgress(
        4,
        shard_totals={index: 100 for index in range(4)},
        shard_labels={0: "2", 1: "3", 2: "6", 3: "7"},
    )

    reporter.update_shard(2, 37, 100, phase=1, label="det/infer 推理")

    assert len(created) == 5
    assert [bar.position for bar in created] == [0, 1, 2, 3, 4]
    assert [bar.desc for bar in created[1:]] == [
        "GPU 2 [s0]",
        "GPU 3 [s1]",
        "GPU 6 [s2]",
        "GPU 7 [s3]",
    ]
    assert created[3].n == 37
    assert created[3].postfix == "infer 推理"
    assert created[0].desc == "multi-GPU 图片合计"


def test_multi_shard_progress_distinguishes_image_and_process_completion(monkeypatch):
    monkeypatch.setenv("LIGHTLY_PROGRESS_MODE", "off")
    reporter = progress.MultiShardProgress(1, shard_totals={0: 2})

    reporter.update_shard(0, 2, 2, phase=1, label="det/eval 推理", status="complete")

    assert reporter._shard_status(0, now=time.monotonic()) == "图片完成·收尾中"
    assert "收尾" in reporter._postfix()
    assert "收尾" in reporter._postfix(max_width=20)

    reporter.finish_shard(0, success=True)
    assert reporter._shard_status(0, now=time.monotonic()) == "完成"


def test_track_emits_complete_status_for_shard(monkeypatch, capsys):
    monkeypatch.setenv("LIGHTLY_PROGRESS_EVENTS", "stdout")
    monkeypatch.setenv("LIGHTLY_PROGRESS_MODE", "off")
    monkeypatch.setenv("LIGHTLY_SHARD_INDEX", "2")

    assert list(progress.track(["a", "b"], label="infer", aggregate="primary")) == ["a", "b"]

    events = [
        json.loads(line.removeprefix(progress.PROGRESS_EVENT_PREFIX))
        for line in capsys.readouterr().out.splitlines()
        if line.startswith(progress.PROGRESS_EVENT_PREFIX)
    ]
    assert events[-1]["status"] == "complete"
    assert events[-1]["done"] == 2
    assert events[-1]["label"] == "infer"


def test_track_emits_failed_status_for_shard(monkeypatch, capsys):
    monkeypatch.setenv("LIGHTLY_PROGRESS_EVENTS", "stdout")
    monkeypatch.setenv("LIGHTLY_PROGRESS_MODE", "off")
    monkeypatch.setenv("LIGHTLY_SHARD_INDEX", "1")

    def broken_items():
        yield 1
        raise RuntimeError("boom")

    with pytest.raises(RuntimeError, match="boom"):
        list(progress.track(broken_items(), label="eval"))

    events = [
        json.loads(line.removeprefix(progress.PROGRESS_EVENT_PREFIX))
        for line in capsys.readouterr().out.splitlines()
        if line.startswith(progress.PROGRESS_EVENT_PREFIX)
    ]
    assert events[-1]["status"] == "failed"
    assert events[-1]["done"] == 1


def test_track_filters_unrelated_shard_scopes(monkeypatch, capsys):
    monkeypatch.setenv("LIGHTLY_PROGRESS_EVENTS", "stdout")
    monkeypatch.setenv("LIGHTLY_PROGRESS_MODE", "off")
    monkeypatch.setenv("LIGHTLY_SHARD_INDEX", "1")
    monkeypatch.setenv("LIGHTLY_PROGRESS_SCOPE", "inference")

    assert list(progress.track([1, 2], label="目录扫描")) == [1, 2]
    assert list(
        progress.track([1, 2], label="推理", shard_scope="inference")
    ) == [1, 2]

    events = [
        json.loads(line.removeprefix(progress.PROGRESS_EVENT_PREFIX))
        for line in capsys.readouterr().out.splitlines()
        if line.startswith(progress.PROGRESS_EVENT_PREFIX)
    ]
    assert [event["label"] for event in events] == ["推理", "推理"]


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


def test_run_sharded_subprocesses_fails_fast_and_stops_siblings(tmp_path):
    start = time.monotonic()
    failed = gpu_parallel.run_sharded_subprocesses(
        [
            (0, 0, [sys.executable, "-c", "import time; time.sleep(10)"]),
            (1, 1, [sys.executable, "-c", "raise SystemExit(7)"]),
        ],
        cwd=tmp_path,
    )
    assert time.monotonic() - start < 4
    assert any(item.startswith("shard_01(exit=7)") for item in failed)


def test_run_sharded_subprocesses_routes_child_progress_events(monkeypatch, tmp_path):
    captured = SimpleNamespace(events=[], finished=[], heartbeats=0, closed=False)

    class _CaptureProgress:
        def __init__(self, *_args, **_kwargs):
            pass

        def note_activity(self, shard):
            pass

        def report_event(self, payload, *, shard):
            captured.events.append((shard, json.loads(payload)))

        def finish_shard(self, shard, *, success):
            captured.finished.append((shard, success))

        def heartbeat(self):
            captured.heartbeats += 1

        def close(self):
            captured.closed = True

    monkeypatch.setattr(gpu_parallel, "MultiShardProgress", _CaptureProgress)
    payload = json.dumps(
        {
            "shard": "0",
            "phase": 1,
            "label": "infer",
            "aggregate": "primary",
            "done": 2,
            "total": 2,
            "status": "complete",
        },
        separators=(",", ":"),
    )

    failed = gpu_parallel.run_sharded_subprocesses(
        [
            (
                0,
                0,
                [
                    sys.executable,
                    "-c",
                    f"print({(progress.PROGRESS_EVENT_PREFIX + payload)!r}, flush=True)",
                ],
            )
        ],
        cwd=tmp_path,
        shard_totals={0: 2},
    )

    assert failed == []
    assert captured.events[0][0] == 0
    assert captured.events[0][1]["done"] == 2
    assert captured.finished == [(0, True)]
    assert captured.closed is True
