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

import argparse
import csv
from unittest.mock import Mock, patch

import numpy as np
import pytest
import rerun as rr
import torch
from projectaria_eyetracking import model_inference_demo as demo


@pytest.mark.parametrize("console_only", [False, True])
def test_inference_exports_gaze_and_logs_with_current_rerun(tmp_path, console_only):
    output_file = tmp_path / "eye_gaze.csv"
    args = argparse.Namespace(
        vrs="sample.vrs",
        model_checkpoint_path="weights.pth",
        model_config_path="config.yaml",
        device="cpu",
        output_file=str(output_file),
        console_only=console_only,
    )
    eye_stream = demo.StreamId("211-1")
    rgb_stream = demo.StreamId("214-1")
    sensor_data = []
    for stream_id in (eye_stream, rgb_stream):
        frame = Mock()
        frame.get_time_ns.return_value = 123456000
        frame.sensor_data_type.return_value = demo.SensorDataType.IMAGE
        frame.stream_id.return_value = stream_id
        image = Mock()
        image.to_numpy_array.return_value = np.zeros((2, 2), dtype=np.uint8)
        frame.image_data_and_record.return_value = (image, Mock())
        sensor_data.append(frame)

    provider = Mock()
    provider.get_label_from_stream_id.side_effect = ["camera-et", "camera-rgb"]
    provider.get_num_data.return_value = 1
    provider.deliver_queued_sensor_data.return_value = sensor_data
    calibration = provider.get_device_calibration.return_value
    calibration.get_transform_device_cpf.return_value = np.eye(3)
    predictions = torch.tensor([[0.1, 0.2]])
    lower = torch.tensor([[0.01, 0.02]])
    upper = torch.tensor([[0.3, 0.4]])

    with (
        patch.object(demo, "parse_args", return_value=args),
        patch.object(demo.infer, "EyeGazeInference") as inference,
        patch.object(
            demo.data_provider, "create_vrs_data_provider", return_value=provider
        ),
        patch.object(demo, "AriaGlassesOutline", return_value=[[0, 0, 0], [1, 0, 0]]),
        patch.object(
            demo, "get_gaze_vector_reprojection", return_value=np.array([1.0, 2.0])
        ) as reproject,
        patch.object(demo.rr, "init", autospec=True) as init,
        patch.object(demo.rr, "log", autospec=True) as log,
        patch.object(demo.rr, "set_time", autospec=True),
    ):
        inference.return_value.predict.return_value = (predictions, lower, upper)
        demo.main()

    with output_file.open(newline="") as csvfile:
        rows = list(csv.DictReader(csvfile))
    assert len(rows) == 1
    assert rows[0]["tracking_timestamp_us"] == "123456"
    assert float(rows[0]["yaw_rads_cpf"]) == pytest.approx(0.1)
    assert float(rows[0]["pitch_rads_cpf"]) == pytest.approx(0.2)
    assert rows[0]["depth_m"] == ""

    if console_only:
        init.assert_not_called()
        log.assert_not_called()
        reproject.assert_not_called()
    else:
        init.assert_called_once_with("MPS Data Viewer", spawn=True)
        static_logs = [call for call in log.call_args_list if call.kwargs.get("static")]
        assert len(static_logs) == 8
        assert all("timeless" not in call.kwargs for call in log.call_args_list)
        assert (
            sum(isinstance(call.args[1], rr.SeriesLines) for call in static_logs) == 6
        )
        scalars = [
            call.args[1]
            for call in log.call_args_list
            if isinstance(call.args[1], rr.Scalars)
        ]
        assert len(scalars) == 6
        gaze = reproject.call_args.args[0]
        assert isinstance(gaze, demo.EyeGaze)
        assert gaze.yaw == pytest.approx(0.1)
        assert gaze.pitch == pytest.approx(0.2)
