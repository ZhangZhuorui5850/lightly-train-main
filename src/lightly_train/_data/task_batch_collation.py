#
# Copyright (c) Lightly AG and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
from lightly_train._transforms.image_classification_transform import (
    ImageClassificationCollateFunction,
)
from lightly_train._transforms.instance_segmentation_transform import (
    InstanceSegmentationCollateFunction,
)
from lightly_train._transforms.object_detection_transform import (
    ObjectDetectionCollateFunction,
)
from lightly_train._transforms.oriented_object_detection_transform import (
    OrientedObjectDetectionCollateFunction,
)
from lightly_train._transforms.panoptic_segmentation_transform import (
    MaskPanopticSegmentationCollateFunction,
)
from lightly_train._transforms.semantic_segmentation_transform import (
    SemanticSegmentationCollateFunction,
)
from lightly_train._transforms.task_transform import TaskCollateFunction

