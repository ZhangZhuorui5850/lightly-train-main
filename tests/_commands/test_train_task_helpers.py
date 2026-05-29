#
# Copyright (c) Lightly AG and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
from __future__ import annotations

from pathlib import Path

import torch
from lightning_fabric import Fabric
from pytest import LogCaptureFixture
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from lightly_train._commands.train_task_helpers import (
    BestAggregatedMetricValues,
    get_best_metrics,
    resume_from_checkpoint,
)
from lightly_train._metrics.task_metric import AggregatedMetricValues, TaskMetricArgs


class _TinyModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(2, 1)


def _make_state(with_best: bool) -> dict:
    """Build a minimal TrainTaskState-like dict for checkpoint round-trip tests."""
    model = _TinyModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1)
    dataset = TensorDataset(torch.zeros(2, 2), torch.zeros(2, 1))
    dataloader = DataLoader(dataset, batch_size=2)
    state: dict = {
        "train_model": model,
        "optimizer": optimizer,
        "scheduler": scheduler,
        "train_dataloader": dataloader,
        "step": 5,
        "model_class_path": "tests._commands.test_train_task_helpers._TinyModel",
        "model_init_args": {},
    }
    if with_best:
        state["best_agg_metric_values"] = BestAggregatedMetricValues(
            agg_metric_values=AggregatedMetricValues(
                metric_values={"val_metric/acc": 0.42},
                watch_metric="val_metric/acc",
                watch_metric_value=0.42,
                watch_metric_mode="max",
                best_head_name=None,
                best_head_metric_values=None,
            ),
            step=7,
        )
    return state


def test_resume_from_checkpoint__restores_best_agg_metric_values(
    tmp_path: Path,
) -> None:
    """best_agg_metric_values must survive a save/resume round-trip.

    Regression test: previously the best metric was a local variable that was reset to
    None on resume, so the first validation after resume overwrote the best checkpoint
    with a worse model.
    """
    fabric = Fabric(accelerator="cpu", devices=1)
    ckpt_path = tmp_path / "last.ckpt"
    fabric.save(ckpt_path, _make_state(with_best=True))  # type: ignore[arg-type]

    # Fresh state is constructed WITHOUT best_agg_metric_values, mirroring the real
    # training entrypoint (the key is not part of the fabric `state` passed to load).
    fresh_state = _make_state(with_best=False)
    resume_from_checkpoint(
        fabric=fabric, state=fresh_state, checkpoint_path=ckpt_path  # type: ignore[arg-type]
    )

    assert fresh_state["step"] == 5
    best = fresh_state["best_agg_metric_values"]
    assert best is not None
    assert best.step == 7
    assert best.agg_metric_values.watch_metric_value == 0.42


def test_resume_from_checkpoint__missing_best_is_backward_compatible(
    tmp_path: Path,
) -> None:
    """Resuming a checkpoint saved before best metric persistence must not error."""
    fabric = Fabric(accelerator="cpu", devices=1)
    ckpt_path = tmp_path / "last.ckpt"
    # Simulate an old checkpoint: saved without best_agg_metric_values.
    fabric.save(ckpt_path, _make_state(with_best=False))  # type: ignore[arg-type]

    fresh_state = _make_state(with_best=False)
    resume_from_checkpoint(
        fabric=fabric, state=fresh_state, checkpoint_path=ckpt_path  # type: ignore[arg-type]
    )

    assert fresh_state["step"] == 5
    # No best metric in the checkpoint -> key is simply not restored (defaults handled by
    # the training loop via state.get(...)).
    assert "best_agg_metric_values" not in fresh_state


def test_get_best_metrics__no_previous_best() -> None:
    last = AggregatedMetricValues(
        metric_values={"val_metric/acc": 0.8},
        watch_metric="val_metric/acc",
        watch_metric_value=0.8,
        watch_metric_mode="max",
        best_head_name=None,
        best_head_metric_values=None,
    )
    result = get_best_metrics(
        best_agg_metric_values=None,
        last_agg_metric_values=last,
        step=0,
        metric_args=TaskMetricArgs(watch_metric="val_metric/acc"),
    )
    assert result.agg_metric_values is last
    assert result.step == 0


def test_get_best_metrics__max_mode_improvement() -> None:
    prev = AggregatedMetricValues(
        metric_values={"val_metric/acc": 0.5},
        watch_metric="val_metric/acc",
        watch_metric_value=0.5,
        watch_metric_mode="max",
        best_head_name=None,
        best_head_metric_values=None,
    )
    best = BestAggregatedMetricValues(agg_metric_values=prev, step=0)
    last = AggregatedMetricValues(
        metric_values={"val_metric/acc": 0.8},
        watch_metric="val_metric/acc",
        watch_metric_value=0.8,
        watch_metric_mode="max",
        best_head_name=None,
        best_head_metric_values=None,
    )
    result = get_best_metrics(
        best_agg_metric_values=best,
        last_agg_metric_values=last,
        step=1,
        metric_args=TaskMetricArgs(watch_metric="val_metric/acc"),
    )
    assert result.agg_metric_values is last
    assert result.step == 1


def test_get_best_metrics__max_mode_no_improvement() -> None:
    prev = AggregatedMetricValues(
        metric_values={"val_metric/acc": 0.9},
        watch_metric="val_metric/acc",
        watch_metric_value=0.9,
        watch_metric_mode="max",
        best_head_name=None,
        best_head_metric_values=None,
    )
    best = BestAggregatedMetricValues(agg_metric_values=prev, step=0)
    last = AggregatedMetricValues(
        metric_values={"val_metric/acc": 0.7},
        watch_metric="val_metric/acc",
        watch_metric_value=0.7,
        watch_metric_mode="max",
        best_head_name=None,
        best_head_metric_values=None,
    )
    result = get_best_metrics(
        best_agg_metric_values=best,
        last_agg_metric_values=last,
        step=1,
        metric_args=TaskMetricArgs(watch_metric="val_metric/acc"),
    )
    assert result is best


def test_get_best_metrics__min_mode_improvement() -> None:
    prev = AggregatedMetricValues(
        metric_values={"val_loss": 0.8},
        watch_metric="val_loss",
        watch_metric_value=0.8,
        watch_metric_mode="min",
        best_head_name=None,
        best_head_metric_values=None,
    )
    best = BestAggregatedMetricValues(agg_metric_values=prev, step=0)
    last = AggregatedMetricValues(
        metric_values={"val_loss": 0.3},
        watch_metric="val_loss",
        watch_metric_value=0.3,
        watch_metric_mode="min",
        best_head_name=None,
        best_head_metric_values=None,
    )
    result = get_best_metrics(
        best_agg_metric_values=best,
        last_agg_metric_values=last,
        step=2,
        metric_args=TaskMetricArgs(watch_metric="val_loss"),
    )
    assert result.agg_metric_values is last
    assert result.step == 2


def test_get_best_metrics__min_mode_no_improvement() -> None:
    prev = AggregatedMetricValues(
        metric_values={"val_loss": 0.3},
        watch_metric="val_loss",
        watch_metric_value=0.3,
        watch_metric_mode="min",
        best_head_name=None,
        best_head_metric_values=None,
    )
    best = BestAggregatedMetricValues(agg_metric_values=prev, step=0)
    last = AggregatedMetricValues(
        metric_values={"val_loss": 0.9},
        watch_metric="val_loss",
        watch_metric_value=0.9,
        watch_metric_mode="min",
        best_head_name=None,
        best_head_metric_values=None,
    )
    result = get_best_metrics(
        best_agg_metric_values=best,
        last_agg_metric_values=last,
        step=2,
        metric_args=TaskMetricArgs(watch_metric="val_loss"),
    )
    assert result is best


def test_get_best_metrics__missing_watch_metric(caplog: LogCaptureFixture) -> None:
    # watch_metric configured but not present in computed metrics
    # last is returned as best since no valid best exists.
    prev = AggregatedMetricValues(
        metric_values={"val_metric/acc": 0.9},
        watch_metric=None,
        watch_metric_value=None,
        watch_metric_mode=None,
        best_head_name=None,
        best_head_metric_values=None,
    )
    best = BestAggregatedMetricValues(agg_metric_values=prev, step=0)
    last = AggregatedMetricValues(
        metric_values={"val_metric/acc": 0.95},
        watch_metric=None,
        watch_metric_value=None,
        watch_metric_mode=None,
        best_head_name=None,
        best_head_metric_values=None,
    )
    with caplog.at_level("WARNING"):
        result = get_best_metrics(
            best_agg_metric_values=best,
            last_agg_metric_values=last,
            step=1,
            metric_args=TaskMetricArgs(watch_metric="val_metric/nonexistent"),
        )
    assert "Unknown watch metric" in caplog.text
    assert result.agg_metric_values is last
    assert result.step == 1
