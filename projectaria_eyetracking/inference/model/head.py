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

import torch.nn as nn

from .backbone import SplitAndConcat
from .model_utils import init_weights


class SocialEyePredictionBoundHead(nn.Module):
    """
    A PyTorch module for predicting gaze direction in the SocialEye model.
    This module takes an input tensor, applies an average pooling operation to reduce its spatial dimensions,
    splits the tensor along the batch dimension (left and right eye) and concatenates it along the channel dimension.
    It then reshapes the tensor and applies three separate fully connected layers to predict the main gaze, upper,
    and lower bounds.
    Args:
        in_channels (int): The number of input channels.
        out_channels (int): The number of output channels.
        final_height_width (tuple, optional): The target height and width for the average pooling operation.
            Defaults to (1, 1).
    Attributes:
        in_channels (int): The number of input channels.
        out_channels (int): The number of output channels.
        final_height_width (tuple): The target height and width for the average pooling operation.
        avgpool (nn.AdaptiveAvgPool2d): The average pooling layer.
        splitconcat (SplitAndConcat): The split and concatenate operation.
        output_feature_dim (int): The dimensionality of the output feature vector.
        fc_main (nn.Linear): The main gaze fully connected layer.
        fc_upper (nn.Linear): The upper bound gaze fully connected layer.
        fc_lower (nn.Linear): The lower bound gaze fully connected layer.
    """

    def __init__(self, in_channels, out_channels, final_height_width: tuple = (1, 1)):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.final_height_width = final_height_width
        self.avgpool = nn.AdaptiveAvgPool2d(self.final_height_width)
        # split into 2 chunks along batch dimension and concat along channel
        # dimension: 2N x C x H x W -> N x 2C x H x W
        self.splitconcat = SplitAndConcat(split_dim=0, concat_dim=1)
        self.output_feature_dim = (
            self.final_height_width[0] * self.final_height_width[1]
        )
        self.fc_main = nn.Linear(
            2 * in_channels * self.output_feature_dim, out_channels
        )
        self.fc_upper = nn.Linear(
            2 * in_channels * self.output_feature_dim, out_channels
        )
        self.fc_lower = nn.Linear(
            2 * in_channels * self.output_feature_dim, out_channels
        )
        init_weights(self.modules)

    def forward(self, x):
        x = self.avgpool(x)
        x = self.splitconcat(x)
        x = x.view(x.shape[0], -1)
        x_main = self.fc_main(x)
        x_upper = self.fc_upper(x)
        x_lower = self.fc_lower(x)
        return {"main": x_main, "lower": x_lower, "upper": x_upper}

    def extra_repr(self):
        return f"in_channels={self.in_channels}, out_channels={self.out_channels}"
