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

import torch
from torchvision import transforms


def preprocess_image(et_image, size=(240, 320)):

    h, w = et_image.shape
    pred_image = torch.zeros((1, 2, h, w // 2))

    pred_image[0, 0, :, :] = resize_and_normalize(et_image[:, : w // 2], size, False)
    pred_image[0, 1, :, :] = resize_and_normalize(et_image[:, w // 2 :], size, True)

    return pred_image


def resize_and_normalize(image, size=(240, 320), should_flip=False):
    image = image.float()
    normalized_image = (image - torch.min(image)) / (
        torch.max(image) - torch.min(image)
    ) - 0.5
    # Flip the image
    if should_flip:
        normalized_image = torch.fliplr(normalized_image)
    transform = transforms.Compose(
        [
            transforms.Resize(size),  # replace with desired size
        ]
    )
    # Resize the image
    final_image = transform(normalized_image)
    # Convert back to tensor and assign to pred_image
    return final_image
