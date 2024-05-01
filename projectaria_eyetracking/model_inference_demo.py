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
import os

import rerun as rr
import torch

try:
    from inference import infer  # Try local imports first
except ImportError:
    from projectaria_eyetracking.inference import infer

from projectaria_tools.core import data_provider
from projectaria_tools.core.mps import EyeGaze, get_eyegaze_point_at_depth
from projectaria_tools.core.mps.utils import get_gaze_vector_reprojection
from projectaria_tools.core.sensor_data import SensorDataType, TimeDomain
from projectaria_tools.core.stream_id import StreamId
from projectaria_tools.utils.rerun_helpers import AriaGlassesOutline

from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--vrs",
        type=str,
        required=True,
        help="path to VRS file",
    )
    parser.add_argument(
        "--model_checkpoint_path",
        type=str,
        default=f"{os.path.dirname(__file__)}/inference/model/pretrained_weights/social_eyes_uncertainty_v1/weights.pth",
        help="location of the model weights",
    )
    parser.add_argument(
        "--model_config_path",
        type=str,
        default=f"{os.path.dirname(__file__)}/inference/model/pretrained_weights/social_eyes_uncertainty_v1/config.yaml",
        help="location of the model config",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default="general_eye_gaze.csv",
        help="Output file to save inference results (Compatible with MPS eye gaze format)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="device to run inference on",
    )
    parser.add_argument(
        "-c", "--console_only", action="store_true", help="Do not show window"
    )
    return parser.parse_args()


def main():
    args = parse_args()

    inference_model = infer.EyeGazeInference(
        args.model_checkpoint_path, args.model_config_path, args.device
    )
    print(
        f"""
    Trying to load the following list of files:
    - vrs: {args.vrs}
    """
    )

    if args.console_only is False:
        # Initializing Rerun viewer
        rr.init("MPS Data Viewer", spawn=True)

    #
    # Go over EyeGaze data
    # - Run Inference
    # - Log data to plot, a 3D vector and image reprojection on a depth proxy
    #

    eye_gaze_inference_results = []
    eye_gaze_inference_results.append(
        [
            "tracking_timestamp_us",
            "yaw_rads_cpf",
            "pitch_rads_cpf",
            "depth_m",
            "yaw_low_rads_cpf",
            "pitch_low_rads_cpf",
            "yaw_high_rads_cpf",
            "pitch_high_rads_cpf",
        ]
    )

    provider = data_provider.create_vrs_data_provider(args.vrs)
    eye_stream_id = StreamId("211-1")
    rgb_stream_id = StreamId("214-1")
    eye_stream_label = provider.get_label_from_stream_id(eye_stream_id)
    rgb_stream_label = provider.get_label_from_stream_id(rgb_stream_id)
    device_calibration = provider.get_device_calibration()
    T_device_CPF = device_calibration.get_transform_device_cpf()
    rgb_camera_calibration = device_calibration.get_camera_calib(rgb_stream_label)
    # eye_camera_calibration = device_calibration.get_aria_et_camera_calib()

    # Configure the loop for data replay
    deliver_option = provider.get_default_deliver_queued_options()
    deliver_option.deactivate_stream_all()
    deliver_option.activate_stream(eye_stream_id)
    deliver_option.activate_stream(rgb_stream_id)
    eye_frame_count = provider.get_num_data(eye_stream_id)
    rgb_frame_count = provider.get_num_data(rgb_stream_id)

    if args.console_only is False:
        # Aria coordinate system sets X down, Z in front, Y Left
        rr.log("device", rr.ViewCoordinates.RIGHT_HAND_X_DOWN, timeless=True)
        rr.log(
            "device/glasses_outline",
            rr.LineStrips3D(AriaGlassesOutline(device_calibration)),
            timeless=True,
        )

        # Define Scalar color attributes
        color_mapping = {
            "yaw": [102, 255, 102],
            "pitch": [102, 178, 255],
            "yaw_lower": [102, 255, 102],
            "pitch_lower": [102, 102, 255],
            "yaw_upper": [102, 255, 178],
            "pitch_upper": [178, 102, 255],
        }
        for name, color in color_mapping.items():
            rr.log(
                f"eye_gaze_inference/{name}",
                rr.SeriesLine(color=color, name=name),
                timeless=True,
            )

    depth_m = 1  # 1 m

    # Iterate over the data and LOG data as we see fit
    progress_bar = tqdm(total=eye_frame_count + rgb_frame_count)
    value_mapping = {}
    for data in provider.deliver_queued_sensor_data(deliver_option):
        device_time_ns = data.get_time_ns(TimeDomain.DEVICE_TIME)

        if args.console_only is False:
            rr.set_time_nanos("device_time", device_time_ns)
            rr.set_time_sequence("timestamp", device_time_ns)
        progress_bar.update(1)

        # If image data is available, log it
        if data.sensor_data_type() == SensorDataType.IMAGE:
            img = torch.tensor(
                data.image_data_and_record()[0].to_numpy_array(), device=args.device
            )

            if data.stream_id() == eye_stream_id:
                preds, lower, upper = inference_model.predict(img)
                preds = preds.detach().cpu().numpy()
                lower = lower.detach().cpu().numpy()
                upper = upper.detach().cpu().numpy()

                value_mapping = {
                    "yaw": preds[0][0],
                    "pitch": preds[0][1],
                    "yaw_lower": lower[0][0],
                    "pitch_lower": lower[0][1],
                    "yaw_upper": upper[0][0],
                    "pitch_upper": upper[0][1],
                }

                # Save data to a list to enable to log to a CSV file later
                depth_m_str = ""
                eye_gaze_inference_result = [
                    int(device_time_ns / 1000),  # Convert ns to us
                    value_mapping["yaw"],
                    value_mapping["pitch"],
                    depth_m_str,
                    value_mapping["yaw_lower"],
                    value_mapping["pitch_lower"],
                    value_mapping["yaw_upper"],
                    value_mapping["pitch_upper"],
                ]
                eye_gaze_inference_results.append(eye_gaze_inference_result)

                if args.console_only is False:
                    rr.log(
                        f"{eye_stream_label}",
                        rr.Image(img),
                    )

                    for name, value in value_mapping.items():
                        rr.log(
                            f"eye_gaze_inference/{name}",
                            rr.Scalar(value),
                        )

            elif data.stream_id() == rgb_stream_id and args.console_only is False:
                rr.log(
                    f"{rgb_stream_label}",
                    rr.Image(img),
                )

                if len(value_mapping) > 0:  # If any value previously computed, use it
                    eye_gaze = EyeGaze
                    eye_gaze.yaw = value_mapping["yaw"]
                    eye_gaze.pitch = value_mapping["pitch"]
                    # Compute eye_gaze vector at depth_m reprojection in the image
                    gaze_projection = get_gaze_vector_reprojection(
                        eye_gaze,
                        rgb_stream_label,
                        device_calibration,
                        rgb_camera_calibration,
                        depth_m,
                    )
                    if gaze_projection is not None:
                        rr.log(
                            f"{rgb_stream_label}/eye-gaze_projection",
                            rr.Points2D(
                                gaze_projection, radii=20, colors=[[0, 255, 0]]
                            ),
                        )

            if len(value_mapping) > 0:

                if args.console_only is False:
                    gaze_vector_in_cpf = get_eyegaze_point_at_depth(
                        value_mapping["yaw"], value_mapping["pitch"], depth_m
                    )
                    # Move EyeGaze vector to CPF coordinate system for visualization
                    rr.log(
                        "device/eye-gaze",
                        rr.Arrows3D(
                            origins=[T_device_CPF @ [0, 0, 0]],
                            vectors=[T_device_CPF @ gaze_vector_in_cpf],
                            colors=[[255, 0, 255]],
                        ),
                    )

    # Export to CSV
    if args.output_file:
        csv_filename = args.output_file
        # Writing to the CSV file
        with open(csv_filename, "w", newline="") as csvfile:
            csvwriter = csv.writer(csvfile)
            for row in eye_gaze_inference_results:
                csvwriter.writerow(row)


if __name__ == "__main__":
    main()
