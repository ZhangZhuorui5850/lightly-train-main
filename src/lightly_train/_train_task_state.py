#
# Copyright (c) Lightly AG and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
from __future__ import annotations

from typing import TYPE_CHECKING, Any, TypedDict

from torch.nn import Module
from torch.optim.lr_scheduler import LRScheduler
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader
from typing_extensions import NotRequired

from lightly_train.types import TaskBatch

if TYPE_CHECKING:
    from lightly_train._commands.train_task_helpers import BestAggregatedMetricValues


class TrainTaskState(TypedDict):
    train_model: Module
    optimizer: Optimizer
    scheduler: LRScheduler
    train_dataloader: DataLoader[TaskBatch]
    step: int
    # Model class path and initialization arguments for serialization.
    # Used to reconstruct the model after training.
    model_class_path: str
    model_init_args: dict[str, Any]
    license_info: NotRequired[str]
    # Best validation metric seen so far. Persisted in the checkpoint so that resuming a
    # run does not overwrite the best checkpoint with a worse model. This is not a
    # stateful object (no state_dict) and may be absent in older checkpoints, so it is
    # restored manually from the loaded checkpoint rather than via fabric's `state`.
    best_agg_metric_values: NotRequired["BestAggregatedMetricValues | None"]


class CheckpointDict(TypedDict):
    train_model_state_dict: dict[str, Any]
    # Model class path and initialization arguments for serialization.
    # Used to reconstruct the model after training.
    model_class_path: str
    model_init_args: dict[str, Any]
    license_info: NotRequired[str]
