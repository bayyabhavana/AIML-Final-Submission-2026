
import streamlit as st
import csv
from pathlib import Path

# Project paths
ROOT = Path(__file__).resolve().parent.parent
OUTPUTS = ROOT / "simulator" / "outputs"

# Page configuration
st.set_page_config(
    page_title="Autonomous Driving Perception",
    page_icon="🚗",
    layout="wide"
)

# Main heading
st.title("🚗 Autonomous Driving Perception")
st.subheader("AI-Powered Road Scene Analysis")

st.write(
    "An AI and computer vision project demonstrating "
    "object detection, lane detection, segmentation "
    "and monocular relative depth estimation."
)

st.divider()

# Perception results
st.header("🔍 Perception Results")

images = [
    ("Object Detection and Lane Detection", "perception_result.jpg"),
    ("MiDaS Relative Depth Estimation", "depth_result.jpg"),
    ("Lane Detection", "lane_result.jpg"),
    ("Edge Detection", "edge_result.jpg")
]

for title, filename in images:
    st.subheader(title)
    image_path = OUTPUTS / filename

    if image_path.exists():
        st.image(str(image_path), use_container_width=True)
    else:
        st.warning(f"Result image not found: {filename}")

st.divider()

# Benchmark results
st.header("⚡ ONNX Performance Benchmark")

benchmark_path = (
    ROOT / "outputs" / "benchmark" / "benchmark_results.csv"
)

if benchmark_path.exists():
    with benchmark_path.open(
        "r", encoding="utf-8-sig", newline=""
    ) as file:
        rows = list(csv.DictReader(file))

    if rows:
        st.dataframe(rows, use_container_width=True)
    else:
        st.info("No benchmark records available.")
else:
    st.info("Benchmark CSV is not available.")

st.divider()

# KITTI dataset analysis
st.header("📊 KITTI Dataset Class Distribution")

chart_path = (
    ROOT / "outputs" / "dataset_analysis"
    / "kitti_class_distribution.png"
)

if chart_path.exists():
    st.image(str(chart_path), use_container_width=True)
else:
    st.info("KITTI class distribution chart is not available.")

csv_path = (
    ROOT / "outputs" / "dataset_analysis"
    / "kitti_class_distribution.csv"
)

if csv_path.exists():
    with csv_path.open(
        "r", encoding="utf-8-sig", newline=""
    ) as file:
        rows = list(csv.DictReader(file))

    if rows:
        st.dataframe(rows, use_container_width=True)

st.divider()

# Technologies
st.header("🛠️ Technologies Used")

st.markdown("""
- YOLOv8 – Object Detection
- OpenCV – Lane Detection
- MiDaS – Relative Depth Estimation
- SegFormer – Semantic Segmentation Experiment
- ROS 2 – Image Communication
- ONNX – Model Export and Inference
- PyTorch – Deep Learning
- Streamlit – Web Demonstration
""")

st.divider()

# Project status
st.header("📌 Project Information")

st.write("""
This project demonstrates individual autonomous-driving
perception components and their integration experiments.

The displayed results are generated from the project's
local inference pipeline. The website is a results
showcase and does not control a real vehicle.
""")

st.caption(
    "AIML-04 | Autonomous Driving Perception Stack"
)
