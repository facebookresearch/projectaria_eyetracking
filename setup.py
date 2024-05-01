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

from setuptools import find_namespace_packages, find_packages, setup

setup(
    name="projectaria_eyetracking",
    version="1.0",
    author="Meta Reality Labs Research",
    license="Apache-2.0",
    python_requires=">=3.8",
    install_requires=[
        "easydict",
        "pyyaml",
        "torch",
        "torchvision",
        "projectaria_tools",
    ],
    packages=find_namespace_packages(
        where="./",
        include=[
            "projectaria_eyetracking",
            "projectaria_eyetracking.inference",
            "projectaria_eyetracking.inference.data",
            "projectaria_eyetracking.inference.model",
            "projectaria_eyetracking.inference.model.pretrained_weights.social_eyes_uncertainty_v1",
        ],
    ),
    package_dir={"": "./"},
    include_package_data=True,
    entry_points={
        "console_scripts": [
            "eyetracking_demo = projectaria_eyetracking.model_inference_demo:main",
        ]
    },
)
