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

import pickle

import pytest
import torch
from easydict import EasyDict
from projectaria_eyetracking.inference.model.model_utils import load_checkpoint


class UntrustedMetadata:
    pass


def test_load_checkpoint_with_easydict_metadata(tmp_path):
    source = torch.nn.Linear(2, 1)
    checkpoint = tmp_path / "weights.pth"
    torch.save(
        {"model": source.state_dict(), "config": EasyDict(device="cpu")}, checkpoint
    )
    destination = torch.nn.Linear(2, 1)
    safe_globals = torch.serialization.get_safe_globals()

    result = load_checkpoint(destination, str(checkpoint))

    torch.testing.assert_close(destination.weight, source.weight)
    torch.testing.assert_close(destination.bias, source.bias)
    assert result["config"].device == "cpu"
    assert torch.serialization.get_safe_globals() == safe_globals


def test_load_checkpoint_rejects_untrusted_metadata(tmp_path):
    model = torch.nn.Linear(2, 1)
    checkpoint = tmp_path / "weights.pth"
    torch.save(
        {"model": model.state_dict(), "metadata": UntrustedMetadata()}, checkpoint
    )
    safe_globals = torch.serialization.get_safe_globals()

    with pytest.raises(pickle.UnpicklingError, match="UntrustedMetadata"):
        load_checkpoint(model, str(checkpoint))

    assert torch.serialization.get_safe_globals() == safe_globals
