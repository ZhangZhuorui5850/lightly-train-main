from __future__ import annotations

from types import SimpleNamespace

import pytest

from tool_lib import common


class FakeEventAccumulator:
    def __init__(self, scalar_values: dict[str, list[tuple[int, float, float]]]) -> None:
        self.scalar_values = scalar_values

    def Tags(self) -> dict[str, list[str]]:
        return {"scalars": list(self.scalar_values)}

    def Scalars(self, tag: str) -> list[SimpleNamespace]:
        return [
            SimpleNamespace(step=step, value=value, wall_time=wall_time)
            for step, value, wall_time in self.scalar_values[tag]
        ]


def test_load_merged_scalar_series_keeps_history_across_resume(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    first_event = tmp_path / "events.out.tfevents.first"
    resumed_event = tmp_path / "events.out.tfevents.resumed"
    accumulators = {
        first_event: FakeEventAccumulator(
            {
                "train_loss": [(0, 1.0, 10.0), (100, 0.8, 20.0)],
                "val_metric/map": [(100, 0.3, 20.0)],
            }
        ),
        resumed_event: FakeEventAccumulator(
            {
                "train_loss": [(100, 0.75, 30.0), (200, 0.6, 40.0)],
                "val_metric/map": [(200, 0.5, 40.0)],
            }
        ),
    }
    monkeypatch.setattr(
        common,
        "list_event_files",
        lambda _experiment_dir: [resumed_event, first_event],
    )
    monkeypatch.setattr(
        common,
        "load_event_accumulator",
        lambda event_file: accumulators[event_file],
    )

    result = common.load_merged_scalar_series(
        tmp_path,
        ("train_loss", "val_metric/map", "val_loss"),
    )

    assert result["train_loss"] == [(0, 1.0), (100, 0.75), (200, 0.6)]
    assert result["val_metric/map"] == [(100, 0.3), (200, 0.5)]
    assert result["val_loss"] == []
