# Autonomous Driving Perception Stack

## Project Overview

The **Autonomous Driving Perception Stack** is an AI/ML-based project that explores computer vision techniques used in autonomous driving systems. It combines object detection, lane detection, semantic segmentation, monocular depth estimation, object tracking, and image-processing techniques to understand road scenes.

The project uses YOLOv8 for object detection, OpenCV for lane detection, MiDaS for relative depth estimation, SegFormer for semantic segmentation experiments, and ROS 2 for image-message communication.

It also includes KITTI dataset label analysis, ONNX model export and inference, and CPU performance benchmarking.

## Project Objectives

- Explore the KITTI dataset and analyze object-class distribution.
- Perform road-scene object detection using YOLOv8.
- Experiment with object tracking using ByteTrack.
- Detect road lanes using OpenCV image-processing techniques.
- Perform semantic segmentation using a pretrained SegFormer model.
- Estimate relative depth from images using MiDaS.
- Build a basic perception pipeline combining multiple computer vision modules.
- Experiment with ROS 2 image publishing and processing.
- Export YOLOv8 to ONNX and compare inference performance.
- Document implementation results, limitations, and future improvements.

## Technologies Used

| Category | Technologies |
|---|---|
| Programming Language | Python |
| Object Detection | YOLOv8, Ultralytics |
| Object Tracking | ByteTrack |
| Image Processing | OpenCV |
| Semantic Segmentation | SegFormer, Cityscapes pretrained model |
| Depth Estimation | MiDaS |
| Dataset | KITTI |
| Deep Learning | PyTorch |
| Model Optimization | ONNX |
| Robotics Middleware | ROS 2 |
| Data Analysis | Pandas, Matplotlib |
| Testing | Pytest |
| Version Control | Git, GitHub |

## Project Structure

```text
Autonomous-Driving-Perception/
│
├── datasets/
│   ├── bus.jpg
│   ├── road.jpg
│   └── kitti/
│       └── training/
│           └── label_2/
│
├── docs/
│   └──
│
├── models/
│   └── yolov8n.onnx
│
├── outputs/
│   ├── benchmark/
│   │   └── benchmark_results.csv
│   │
│   ├── dataset_analysis/
│   │   ├── kitti_class_distribution.csv
│   │   └── kitti_class_distribution.png
│   │
│   ├── lane_segmentation/
│   └── object_tracking/
│
├── simulator/
│   ├── carla/
│   │   ├── carla_client.py
│   │   └── README.md
│   │
│   ├── outputs/
│   │
│   ├── recordings/
│   │
│   ├── src/
│   │   ├── perception_pipeline.py
│   │   └── video_simulator.py
│   │
│   └── ros2/
│       └── perception_ws/
│
├── src/
│   ├── dataset_analysis/
│   │   └── kitti_analysis.py
│   │
│   ├── lane_detection/
│   │   └── lane.py
│   │
│   ├── lane_segmentation/
│   │   └── segmentation.py
│   │
│   ├── object_tracking/
│   │   └── track.py
│   │
│   └── optimization/
│       └── wandb_tracking.py
│
├── tests/
│   └── test_pipeline.py
│
├── requirements.txt
├── README.md
├── yolov8n.pt
└── yolov8n.onnx
```

*Note: This is the documented project layout. Keep only folders and files that actually exist in your repository. The KITTI dataset files are local and are not included in the GitHub repository.*

## 1. KITTI Dataset Analysis

The KITTI dataset is used to explore object labels commonly found in autonomous driving environments.

A Python script was developed to read the KITTI label files and calculate the distribution of object classes.

**Script:**
`src/dataset_analysis/kitti_analysis.py`

**Generated outputs:**
- `outputs/dataset_analysis/kitti_class_distribution.csv`
- `outputs/dataset_analysis/kitti_class_distribution.png`

### Dataset Analysis Results

- Label files analyzed: **1,600**
- Total annotations: **11,061**

| Object Class | Annotation Count |
|---|---:|
| Car | 5,991 |
| DontCare | 2,385 |
| Pedestrian | 1,111 |
| Van | 607 |
| Cyclist | 392 |
| Truck | 236 |
| Misc | 186 |
| Tram | 108 |
| Person_sitting | 45 |
| **Total** | **11,061** |

The analysis provides an overview of object-class frequency in the available KITTI training labels. The `DontCare` category is included in the label count.

### KITTI Class Distribution

![KITTI Class Distribution](outputs/dataset_analysis/kitti_class_distribution.png)

## 2. Object Detection Using YOLOv8

YOLOv8 is used to detect objects in road-scene images.

The pretrained YOLOv8n model was used to identify objects such as cars and people.

**Implementation:**
- Load the YOLOv8n model.
- Read the input road image.
- Perform object detection.
- Display detected objects and confidence scores.
- Save the resulting detection image.

**Model:** YOLOv8n

**Framework:** Ultralytics

## 3. Object Tracking Using ByteTrack

Object tracking was explored using the YOLOv8 tracking functionality with the ByteTrack tracker.

The tracking script uses video or webcam frames to detect and track objects across frames.

**Implementation:**
- Capture video frames.
- Run YOLO object detection.
- Apply ByteTrack tracking.
- Display tracking results.
- Save the output video.

**Script:** `src/object_tracking/track.py`

**Output:** `outputs/object_tracking/`

This is a basic tracking experiment. Quantitative tracking accuracy has not yet been evaluated.

## 4. Lane Detection Using OpenCV

OpenCV techniques were used to identify possible lane markings in road images.

The lane detection prototype includes:

- Image loading and preprocessing.
- Grayscale conversion.
- Gaussian blur.
- Canny edge detection.
- Region-of-interest masking.
- Hough Line Transform.
- Lane-line visualization.

**Script:** `src/lane_detection/lane.py`

The implementation demonstrates traditional computer vision techniques for road-lane detection. Detection quality can vary depending on image conditions.

## 5. Semantic Segmentation Using SegFormer

A pretrained SegFormer model was used for semantic segmentation experiments.

The model was loaded using the Hugging Face Transformers library and tested on road-scene images.

**Model:** SegFormer-B0

**Dataset:** Cityscapes pretrained model

**Implementation:**
- Load the pretrained segmentation model.
- Preprocess the input image.
- Perform semantic segmentation.
- Identify road-related classes.
- Generate and save the segmentation output.

**Script:** `src/lane_segmentation/segmentation.py`

This implementation is a pretrained inference experiment, not a custom fine-tuned segmentation model.

## 6. Monocular Depth Estimation Using MiDaS

MiDaS was used to estimate relative depth information from a single road image.

The depth estimation module generates a depth map that represents relative distance patterns within the image.

**Implementation:**
- Load the input image.
- Preprocess the image.
- Run the MiDaS model.
- Generate the relative depth map.
- Save the depth visualization.

**Output:** `simulator/outputs/depth_result.jpg`

The generated depth map represents relative depth and should not be interpreted as calibrated distance measurements in metres.

## 7. Integrated Perception Pipeline

A basic perception pipeline was developed to experiment with multiple computer vision operations on road-scene images.

The pipeline combines:

1. YOLOv8 object detection.
2. OpenCV-based lane detection.
3. MiDaS relative depth estimation.
4. Image-processing operations and result visualization.

**Main script:** `simulator/src/perception_pipeline.py`

The pipeline demonstrates how different perception modules can contribute to road-scene understanding. It is an image-based prototype and is not a complete autonomous vehicle control system.

## 8. ROS 2 Integration

ROS 2 was explored for communication between camera-image publishing and perception-processing nodes.

The perception node subscribes to the raw camera image topic, processes incoming frames, and publishes processed image messages.

**ROS 2 topics used:**

- `/camera/image_raw` – Input camera image topic.
- `/camera/image_processed` – Processed image output topic.

The ROS 2 image communication and processing workflow was tested independently.

**Workspace:** `simulator/ros2/perception_ws/`

Live CARLA camera integration has not yet been completed.

## 9. ONNX Model Export and Inference

The YOLOv8n model was exported to the ONNX format to experiment with model portability and inference.

**Model files:**
- `models/yolov8n.onnx`
- `yolov8n.onnx`

ONNX inference was tested using an input image, and the model successfully produced object detections.

## 10. Performance Benchmarking

A basic CPU benchmarking experiment was conducted to compare YOLOv8n inference using PyTorch and ONNX.

### Benchmark Results

| Model Format | Average Latency | FPS |
|---|---:|---:|
| PyTorch YOLOv8n | 109.37 ms | 9.14 |
| ONNX YOLOv8n | 243.51 ms | 4.11 |

**Benchmark output:** `outputs/benchmark/benchmark_results.csv`

### Observations

- PyTorch achieved lower average inference latency in this experiment.
- ONNX inference was slower in the tested CPU environment.
- The results depend on the hardware, software configuration, and inference settings.
- ONNX export and inference were validated, but performance improvement was not achieved in this benchmark.

TensorRT optimization and benchmarking have not yet been completed.

## 11. Testing

Basic automated tests were created using Pytest.

**Test file:** `tests/test_pipeline.py`

The tests check:

- Availability of the sample bus image.
- Availability of the sample road image.
- Successful loading of the bus image.
- Successful loading of the road image.

**Test command:**

```bash
python -m pytest tests -v
```

**Recorded result:** 4 tests passed.

These tests validate basic dataset-file availability and image loading. They do not represent complete integration, model accuracy, or autonomous-driving system validation.

## 12. CARLA Simulator Status

A CARLA client script and documentation were prepared under:

`simulator/carla/`

The client is intended to connect to a CARLA simulator running locally.

However, CARLA was not installed in the development environment, so a live simulator connection, virtual camera integration, and CARLA-to-ROS 2 streaming have not been completed.

## 13. Implementation Status and Limitations

### Implemented and Tested

- YOLOv8 object detection on sample images.
- Basic OpenCV lane detection prototype.
- MiDaS relative depth estimation experiment.
- SegFormer semantic segmentation inference experiment.
- KITTI training-label class distribution analysis.
- Basic YOLOv8 and ByteTrack tracking experiment.
- ROS 2 raw and processed image-topic workflow.
- YOLOv8 ONNX export and inference validation.
- CPU inference benchmarking.
- Four basic automated tests passed.

### Pending or Not Yet Validated

- Custom YOLO fine-tuning on the KITTI dataset.
- Quantitative object-tracking accuracy evaluation.
- Quantitative comparison of lane detection and semantic segmentation.
- Live CARLA simulator integration.
- CARLA virtual camera streaming through ROS 2.
- TensorRT optimization and benchmarking.
- Complete obstacle-reaction demonstration using CARLA.
- Comprehensive end-to-end perception accuracy evaluation.
- Verified Weights & Biases experiment logging.

### Hardware Limitations

The project was developed and tested in a CPU-based environment with limited system resources. The benchmark results are specific to the tested hardware and software configuration.

## 14. Installation and Setup

### Prerequisites

- Python
- Git
- Visual Studio Code (recommended)
- Required Python packages
- ROS 2 for ROS-related experiments

### Clone the Repository

```bash
git clone https://github.com/bayyabhavana/Autonomous-Driving-Perception.git
cd Autonomous-Driving-Perception
```

### Create a Virtual Environment

```bash
python -m venv .venv
```

### Activate the Environment

**Windows:**

```bash
.venv\Scripts\activate
```

**Linux / WSL:**

```bash
source .venv/bin/activate
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

Some experiments may require additional packages or model downloads depending on the environment.

### Run the Perception Pipeline

```bash
python simulator/src/perception_pipeline.py
```

### Run KITTI Dataset Analysis

```bash
python src/dataset_analysis/kitti_analysis.py
```

### Run Object Tracking

```bash
python src/object_tracking/track.py
```

### Run Lane Detection

```bash
python src/lane_detection/lane.py
```

### Run Semantic Segmentation

```bash
python src/lane_segmentation/segmentation.py
```

### Run Tests

```bash
python -m pytest tests -v
```

Run each script from the repository root. Some modules may require their own model weights, dataset files, or environment-specific dependencies.

## 15. Future Enhancements

- Fine-tune YOLOv8 using the KITTI or BDD100K dataset.
- Improve lane detection under different road and lighting conditions.
- Evaluate segmentation and lane-detection performance quantitatively.
- Integrate CARLA with ROS 2 for simulated camera input.
- Implement TensorRT optimization on compatible NVIDIA hardware.
- Evaluate object tracking using standard tracking metrics.
- Improve real-time perception performance.
- Build a complete simulation-based obstacle detection and reaction demonstration.
- Expand automated testing and system-level validation.

## 16. Conclusion

This project explores important computer vision techniques used in autonomous driving perception systems.

It includes object detection, object tracking experiments, lane detection, semantic segmentation, monocular depth estimation, KITTI label analysis, ROS 2 image communication, and ONNX model experimentation.

The project provided practical experience in developing and testing AI/ML perception modules, working with computer vision libraries, analyzing dataset labels, and benchmarking model inference.

Some advanced components, including live CARLA integration and TensorRT optimization, remain future work.

## Author

**Bayya Bhavana Goud**

GitHub: [bayyabhavana](https://github.com/bayyabhavana)
## Project Demonstration Videos

### Demo Video 1
[Watch Project Demonstration](https://drive.google.com/file/d/1XRTYFvYKVoFNitT1E1j3qjAsf_d-vNrY/view?usp=sharing)

### Demo Video 2
[Watch Additional Demonstration](https://drive.google.com/file/d/1Ch2MM67zELBBCsJw-IKdKh2SYPockflf/view?usp=sharing)

Project Repository: [Autonomous-Driving-Perception](https://github.com/bayyabhavana/Autonomous-Driving-Perception)
