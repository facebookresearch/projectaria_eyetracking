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

import math
from typing import OrderedDict

import torch

import torch.nn as nn


NONLINEARITY = {
    "relu": nn.ReLU(inplace=True),
    "relu6": nn.ReLU6(inplace=True),
    "selu": nn.SELU(inplace=True),
    "softplus": nn.Softplus(),
}


def init_weights(modules: nn.Module):
    # Note that modules is a function that returns an iterator over all modules
    # in the network
    for m in modules():
        if isinstance(m, nn.Conv2d):
            # hard code nonlinearity to "relu" since kaiming_normal_ only accepts
            # tanh, relu, leaky relu, and sigmoid
            nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
        elif isinstance(m, nn.Linear):
            m.weight.data.normal_(0, math.sqrt(2.0 / m.in_features))
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1.0)
        if getattr(m, "bias", None) is not None:
            nn.init.constant_(m.bias, 0)


def conv3x3(
    in_planes: int,
    out_planes: int,
    stride: int = 1,
    groups: int = 1,
    padding: int = 1,
    dilation: int = 1,
) -> nn.Module:
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=padding,
        groups=groups,
        bias=False,
        dilation=dilation,
    )


def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Module:
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class Bottleneck(nn.Module):
    """
    A PyTorch module that implements a bottleneck block for ResNet architectures.
    This module includes three convolutional layers: a 1x1 convolution, a 3x3 convolution, and another 1x1 convolution.
    Each convolution is followed by a batch normalization layer (if provided) and a nonlinearity.
    If a downsample module is provided, it is applied to the input before it is added to the output of the bottleneck.
    Args:
        inplanes (int): The number of input planes.
        planes (int): The number of output planes for the first two convolutions. The final convolution will have `planes * expansion` output planes.
        stride (int, optional): The stride to use for the convolution. Defaults to 1.
        downsample (nn.Module, optional): A module to downsample the input. Defaults to None.
        groups (int, optional): The number of groups to use for the 3x3 convolution. Defaults to 1.
        base_width (int, optional): The base width for the 3x3 convolution. Defaults to 64.
        dilation (int, optional): The dilation to use for the 3x3 convolution. Defaults to 1.
        norm_layer (nn.Module, optional): The normalization layer to use after each convolution. Defaults to nn.BatchNorm2d.
        nonlinearity (str, optional): The type of nonlinearity to use. Defaults to "relu".
    Attributes:
        description (str): A string description of the module.
        conv1 (nn.Module): The first 1x1 convolution.
        bn1 (nn.Module): The batch normalization after the first convolution.
        conv2 (nn.Module): The 3x3 convolution.
        bn2 (nn.Module): The batch normalization after the second convolution.
        conv3 (nn.Module): The second 1x1 convolution.
        bn3 (nn.Module): The batch normalization after the third convolution.
        relu (nn.Module): The nonlinearity.
        downsample (nn.Module): The downsampling module.
        stride (int): The stride for the convolution.
    """

    expansion = 4

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: nn.Module = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer=nn.BatchNorm2d,
        nonlinearity: str = "relu",
    ):
        super().__init__()
        self.description = (
            f"inplanes={inplanes}, planes={planes}, "
            f"stride={stride}, downsample={downsample}, "
            f"groups={groups}, base_width={base_width}, "
            f"dilation={dilation}, norm_layer={norm_layer}, "
            f"nonlinearity={nonlinearity}"
        )
        width = int(planes * (base_width / 64.0)) * groups
        # Both self.conv2 and self.downsample layers downsample
        # the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width) if norm_layer is not None else None
        self.conv2 = conv3x3(width, width, stride, groups, dilation, dilation)
        self.bn2 = norm_layer(width) if norm_layer is not None else None
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = (
            norm_layer(planes * self.expansion) if norm_layer is not None else None
        )
        self.relu = NONLINEARITY[nonlinearity]
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        if self.bn1 is not None:
            out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        if self.bn2 is not None:
            out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        if self.bn3 is not None:
            out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

    def extra_repr(self):
        return self.description


def change_dataparallel_to_original(input_state, affected_module=""):
    """[change the names/states of dataparallel module to original]

    Args:
        input ([List[str], OrderedDict[str,torch.tensor]]) : [all state/module names]
        affected_module (str): [the module name to change for i.e. backbone]

    Returns:
        [List[str],OrderedDict[str,torch.tensor]]: [Changed back to the original names]
    """
    dataparallel_affected_module = (
        affected_module + ".module" if affected_module else "module."
    )
    if isinstance(input_state, OrderedDict):
        output = OrderedDict(
            [
                (
                    (
                        name.replace(dataparallel_affected_module, affected_module)
                        if dataparallel_affected_module in name
                        else name
                    ),
                    buffer,
                )
                for name, buffer in input_state.items()
            ]
        )
    else:
        raise NotImplementedError
    return output


def load_checkpoint(
    model,
    chkpt_path: str,
):
    map_location = "cpu"

    model_buffer = torch.load(chkpt_path, map_location=map_location)

    state_dict = model_buffer["model"]
    any_key = next(iter(state_dict.keys()))
    if "module" in any_key:
        print("Renaming DataParallel modules")
        state_dict = change_dataparallel_to_original(state_dict)

    model.load_state_dict(state_dict, strict=True)

    return model_buffer
