# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Dict

import torch
import torch.nn as nn


class SocialEyeModel(nn.Module):
    """
    A PyTorch module for the SocialEye model.
    This module combines a backbone and a head module to form the complete model.
    Args:
        backbone (nn.Module): The backbone module for feature extraction.
        head (nn.Module): The head module for prediction.
    Attributes:
        backbone (nn.Module): The backbone module for feature extraction.
        head (nn.Module): The head module for prediction.
    """

    def __init__(self, backbone, head):
        super(SocialEyeModel, self).__init__()
        self.backbone = backbone
        self.head = head

    def forward(self, x, class_index: int = -1) -> Dict[str, torch.Tensor]:
        x = self.backbone(x)
        x = self.head(x)
        return x

    def __compute_loss(self, preds, targets):
        if not isinstance(targets, torch.Tensor):
            targets = targets["gaze_target"]
        if not isinstance(preds, torch.Tensor):
            preds = preds["gaze_target"]
        loss = self.loss_module(preds, targets)
        return loss
