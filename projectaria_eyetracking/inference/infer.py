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
import yaml

from .data.data import preprocess_image

from .model import backbone
from .model.head import SocialEyePredictionBoundHead
from .model.model import SocialEyeModel
from .model.model_utils import load_checkpoint


class EyeGazeInference:
    """
    A class for performing eye gaze inference using a trained SocialEye model.
    This class loads a trained model from a checkpoint file and uses it to perform eye gaze inference on input images.
    Args:
        model_checkpoint_path (str): The path to the model checkpoint file.
        model_config_path (str): The path to the model configuration file.
        device (str, optional): The device to run the model on. Defaults to "cpu".
    Attributes:
        model_checkpoint_path (str): The path to the model checkpoint file.
        model_config_path (str): The path to the model configuration file.
        device (str): The device to run the model on. e.g. 'cpu', 'cuda:0'
        config (dict): The loaded model configuration.
        model (nn.Module): The loaded model.
    """

    def __init__(self, model_checkpoint_path, model_config_path, device="cpu"):
        self.model_checkpoint_path = model_checkpoint_path
        self.model_config_path = model_config_path
        self.device = device
        self.load_config()
        self.model = self.set_model()

    def load_config(self):
        with open(self.model_config_path, "r") as file:
            self.config = yaml.safe_load(file)

    def set_model(self):
        model_backbone = backbone.build_social_eye(
            self.config,
        )
        head_in_channel = model_backbone.out_channels
        head_out_channel = 2
        final_height_width = tuple(self.config["MODEL"]["HEAD"]["final_height_width"])

        head = SocialEyePredictionBoundHead(
            head_in_channel, head_out_channel, final_height_width
        )
        model = SocialEyeModel(backbone=model_backbone, head=head)

        load_checkpoint(
            model,
            self.model_checkpoint_path,
        )

        print(f"Initialized network weights from:\n{self.model_checkpoint_path}")
        print(" ******************MODEL LOADED AND INIT*******************")
        # Make Data parallel and put on GPU
        if torch.cuda.is_available() and self.device != "cpu":
            model = model.cuda(self.device)
            torch.backends.cudnn.benchmark = True

        model.eval()
        return model

    def post_process(self, preds):
        stats = self.config["STATS"]

        processed_preds = preds * torch.tensor(
            stats["sA"], device=self.device
        ) + torch.tensor(stats["mA"], device=self.device)

        return processed_preds

    def predict(self, image):
        processed_image = preprocess_image(image, self.config["RESIZED_IMAGE_SHAPE"])

        processed_image = (
            processed_image.clone().detach().to(torch.float).to(self.device)
        )
        preds = self.model.forward(processed_image)

        processed_preds = self.post_process(preds["main"])
        return processed_preds, preds["lower"], preds["upper"]
