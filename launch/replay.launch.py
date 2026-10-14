# Copyright 2026 SenseAI Institute
# SPDX-License-Identifier: Apache-2.0
"""Replay a dataset into ROS 2 topics.

ros2 launch launch/replay.launch.py dataset:=dsec path:=/data/dsec/zurich_city_04_a speed:=1.0
"""

from launch.actions import DeclareLaunchArgument, ExecuteProcess
from launch.substitutions import LaunchConfiguration

from launch import LaunchDescription


def generate_launch_description() -> LaunchDescription:
    args = {
        "dataset": "dsec",
        "path": "",
        "tick_ms": "10",
        "speed": "1.0",
        "namespace": "/senseai",
        "python": "python3",
    }
    decls = [DeclareLaunchArgument(k, default_value=v) for k, v in args.items()]
    node = ExecuteProcess(
        cmd=[
            LaunchConfiguration("python"), "-m", "senseai_ingest.ros2.node",
            "--dataset", LaunchConfiguration("dataset"),
            "--path", LaunchConfiguration("path"),
            "--tick-ms", LaunchConfiguration("tick_ms"),
            "--speed", LaunchConfiguration("speed"),
            "--namespace", LaunchConfiguration("namespace"),
        ],
        output="screen",
    )  # fmt: skip
    return LaunchDescription([*decls, node])
