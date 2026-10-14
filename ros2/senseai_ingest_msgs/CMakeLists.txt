# Copyright 2026 SenseAI Institute
# SPDX-License-Identifier: Apache-2.0
cmake_minimum_required(VERSION 3.8)
project(senseai_ingest_msgs)

find_package(ament_cmake REQUIRED)
find_package(rosidl_default_generators REQUIRED)

rosidl_generate_interfaces(${PROJECT_NAME}
  "msg/Tensor.msg"
  "msg/Sample.msg"
  "msg/SampleList.msg"
  "msg/AlignedTick.msg"
)

ament_export_dependencies(rosidl_default_runtime)
ament_package()
