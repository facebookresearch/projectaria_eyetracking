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

from collections import OrderedDict
from typing import Dict, List

import torch
import torch.nn as nn

from .model_archs import MODEL_ARCH

from .model_utils import Bottleneck, init_weights, NONLINEARITY


class SplitAndConcat(nn.Module):
    """
    A PyTorch module that splits an input tensor along a specified dimension and then concatenates the chunks along another specified dimension.
    This module is useful for rearranging the dimensions of a tensor in a specific way, which can be useful in certain types of neural network architectures.
    Args:
        split_dim (int, optional): The dimension along which to split the input tensor. Defaults to 1.
        concat_dim (int, optional): The dimension along which to concatenate the split chunks. Defaults to 0.
        chunk (int, optional): The number of chunks to split the input tensor into. Defaults to 2.
    Attributes:
        split_dim (int): The dimension along which to split the input tensor.
        concat_dim (int): The dimension along which to concatenate the split chunks.
        chunk (int): The number of chunks to split the input tensor into.
    """

    def __init__(self, split_dim: int = 1, concat_dim: int = 0, chunk: int = 2):
        super(SplitAndConcat, self).__init__()
        self.split_dim = split_dim
        self.concat_dim = concat_dim
        self.chunk = chunk

    def forward(self, x):

        block_size = int(x.size(dim=self.split_dim) / self.chunk)
        out = torch.narrow(x, self.split_dim, 0, block_size)
        for chunk_id in range(1, self.chunk):
            slice = torch.narrow(x, self.split_dim, chunk_id * block_size, block_size)
            out = torch.cat((out, slice), dim=self.concat_dim)

        return out

    def extra_repr(self):
        return (
            f"split_dim={self.split_dim}, concat_dim={self.concat_dim}, "
            f"chunk={self.chunk}"
        )


class SocialEye(nn.Module):
    """
    A PyTorch module for the SocialEye model. This module constructs a deep learning model based on the provided stage information
    Args:
        stage_info (Dict[str, List]): A dictionary containing information about the stages of the model.
        in_channels (int, optional): The number of input channels. Defaults to 1.
        nonlinearity (str, optional): The type of nonlinearity to use. Defaults to "relu".
        basic_block (nn.Module, optional): The type of basic block to use in the model. Defaults to Bottleneck.
    Attributes:
        description (str): A string description of the model.
        nonlinearity (str): The type of nonlinearity used in the model.
        split_concat (SplitAndConcat): The split and concatenate operation.
        conv1 (nn.Conv2d): The first convolutional layer.
        bn1 (nn.BatchNorm2d): The first batch normalization layer.
        relu (nn.Module): The nonlinearity.
        run_pool (bool): Whether to run the max pooling operation.
        maxpool (nn.MaxPool2d): The max pooling operation.
        inplanes (int): The number of input planes for the layers.
        feature_extractor (nn.Sequential): The sequence of layers that extract features from the input.
        out_channels (int): The number of output channels.
    """

    def __init__(
        self,
        stage_info: Dict[str, List],
        in_channels: int = 1,
        nonlinearity: str = "relu",
        basic_block=Bottleneck,
    ):
        super(SocialEye, self).__init__()
        self.description = (
            f"stage_info={stage_info}, "
            f"in_channels={in_channels}, in_channels={in_channels}, "
            f"nonlinearity={nonlinearity}"
        )
        self.nonlinearity = nonlinearity
        # split tensors into 2Nx1xHxW
        self.split_concat = SplitAndConcat(split_dim=1, concat_dim=0, chunk=2)
        first_out_ch = stage_info["first"][0]
        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels=first_out_ch,
            kernel_size=stage_info["first"][1],
            stride=stage_info["first"][2],
            padding=stage_info["first"][1] // 2,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(first_out_ch)
        self.relu = NONLINEARITY[nonlinearity]
        self.run_pool = False
        if stage_info["first"][3] >= 2:
            self.maxpool = nn.MaxPool2d(
                kernel_size=3, stride=stage_info["first"][3], padding=1
            )
            self.run_pool = True

        stages = stage_info["stages"]
        assert isinstance(stages[0], list), (
            "stage_info['stages'] type error! Should be list of "
            "[channel, layer repeat, stride]!"
        )
        self.inplanes = first_out_ch
        layers = OrderedDict()
        for i, stage in enumerate(stages):
            layers[f"layer{i+1}"] = self._make_layer(
                basic_block, planes=stage[0], blocks=stage[1], stride=stage[2]
            )

        self.feature_extractor = nn.Sequential(layers)
        self.out_channels = stages[-1][0] * basic_block.expansion

        init_weights(self.modules)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(
                    self.inplanes,
                    planes * block.expansion,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(
            block(
                self.inplanes,
                planes,
                stride,
                downsample,
                nonlinearity=self.nonlinearity,
            )
        )
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(
                block(
                    self.inplanes,
                    planes,
                    nonlinearity=self.nonlinearity,
                )
            )

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.split_concat(x)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        if self.run_pool:
            x = self.maxpool(x)

        x = self.feature_extractor(x)

        return x

    def extra_repr(self):
        return self.description


def build_social_eye(config):
    stage_info = MODEL_ARCH[config["MODEL"]["arch"]]
    kwargs = {k: v for k, v in config["MODEL"]["BACKBONE"].items() if k != "type"}
    kwargs["in_channels"] = 1
    return SocialEye(stage_info=stage_info, **kwargs)
