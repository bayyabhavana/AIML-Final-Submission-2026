# CARLA Simulator Integration

## Status
The CARLA client script is prepared in this folder.
The CARLA simulator and Python API are not installed in the current environment.

## CARLA Client
File: carla_client.py

The client is designed to:
- Connect to a CARLA server at localhost:2000
- Read the active simulation map
- Count simulated vehicles
- Count pedestrians

## ROS2 Perception Integration
The perception pipeline was tested using a ROS2 image publisher and perception node.

Topics:
- /camera/image_raw
- /camera/image_processed

YOLO object detection was demonstrated on the incoming road image.

## Limitation
A live CARLA simulator connection has not yet been tested.
CARLA installation and live simulation testing remain future work.

## Run when CARLA is installed
Start the CARLA simulator first, then run:

python simulator/carla/carla_client.py
