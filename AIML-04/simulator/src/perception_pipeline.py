
import cv2
import torch
import numpy as np

from pathlib import Path
from ultralytics import YOLO


# ==========================================
# 1. PROJECT CONFIGURATION
# ==========================================

BASE_DIR = Path(__file__).resolve().parents[2]

IMAGE_PATH = BASE_DIR / "datasets" / "road.jpg"

YOLO_MODEL_PATH = BASE_DIR / "yolov8n.pt"

OUTPUT_DIR = BASE_DIR / "simulator" / "outputs"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = torch.device("cpu")


# ==========================================
# 2. MAIN PIPELINE
# ==========================================

def main():

    print("\n====================================")
    print(" AUTONOMOUS DRIVING PERCEPTION")
    print("====================================")

    # ======================================
    # LOAD ROAD IMAGE
    # ======================================

    frame = cv2.imread(str(IMAGE_PATH))

    if frame is None:
        raise FileNotFoundError(
            f"Road image not found: {IMAGE_PATH}"
        )

    print("\n[SUCCESS] Road image loaded")

    # ======================================
    # 3. YOLO OBJECT DETECTION
    # ======================================

    print("\n[1/4] Loading YOLO model...")

    model = YOLO(str(YOLO_MODEL_PATH))

    print("[INFO] Detecting and tracking objects...")

    results = model.track(
        frame,
        persist=True,
        verbose=False
    )

    annotated_frame = results[0].plot()

    detections = results[0].boxes

    print("\nDetected Objects:")

    if detections is not None and len(detections) > 0:

        for box in detections:

            class_id = int(box.cls[0])

            confidence = float(box.conf[0])

            class_name = model.names[class_id]

            print(
                f"{class_name}: {confidence:.2f}"
            )

    else:
        print("No objects detected.")

    print("[SUCCESS] YOLO detection completed")

    # ======================================
    # 4. OPENCV LANE DETECTION
    # ======================================

    print("\n[2/4] Detecting road lanes...")

    gray = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2GRAY
    )

    blur = cv2.GaussianBlur(
        gray,
        (5, 5),
        0
    )

    edges = cv2.Canny(
        blur,
        50,
        150
    )

    height, width = edges.shape

    mask = np.zeros_like(edges)

    polygon = np.array([[
        (0, height),
        (width // 2, int(height * 0.55)),
        (width, height)
    ]], dtype=np.int32)

    cv2.fillPoly(
        mask,
        polygon,
        255
    )

    roi = cv2.bitwise_and(
        edges,
        mask
    )

    lines = cv2.HoughLinesP(
        roi,
        1,
        np.pi / 180,
        40,
        minLineLength=30,
        maxLineGap=100
    )

    lane_overlay = annotated_frame.copy()

    lane_count = 0

    if lines is not None:

        # Correct handling of OpenCV output shape
        lines = np.asarray(lines).reshape(-1, 4)

        for line in lines:

            x1, y1, x2, y2 = map(int, line)

            cv2.line(
                lane_overlay,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                3
            )

            lane_count += 1

    annotated_frame = cv2.addWeighted(
        lane_overlay,
        0.8,
        annotated_frame,
        0.2,
        0
    )

    print(
        f"[INFO] Detected lane segments: {lane_count}"
    )

    print("[SUCCESS] Lane detection completed")

    # ======================================
    # 5. MIDAS DEPTH ESTIMATION
    # ======================================

    print("\n[3/4] Loading MiDaS model...")

    midas = torch.hub.load(
        "intel-isl/MiDaS",
        "MiDaS_small",
        trust_repo=True
    )

    midas.to(DEVICE)

    midas.eval()

    midas_transforms = torch.hub.load(
        "intel-isl/MiDaS",
        "transforms",
        trust_repo=True
    )

    transform = midas_transforms.small_transform

    rgb_image = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB
    )

    input_batch = transform(
        rgb_image
    ).to(DEVICE)

    print("[INFO] Estimating relative depth...")

    with torch.no_grad():

        prediction = midas(input_batch)

        prediction = torch.nn.functional.interpolate(
            prediction.unsqueeze(1),
            size=frame.shape[:2],
            mode="bicubic",
            align_corners=False
        ).squeeze()

    depth_map = prediction.cpu().numpy()

    depth_normalized = cv2.normalize(
        depth_map,
        None,
        0,
        255,
        cv2.NORM_MINMAX
    ).astype(np.uint8)

    depth_colormap = cv2.applyColorMap(
        depth_normalized,
        cv2.COLORMAP_MAGMA
    )

    print("[SUCCESS] MiDaS depth estimation completed")

    # ======================================
    # 6. SAVE RESULTS
    # ======================================

    print("\n[4/4] Saving results...")

    cv2.imwrite(
        str(OUTPUT_DIR / "perception_result.jpg"),
        annotated_frame
    )

    cv2.imwrite(
        str(OUTPUT_DIR / "depth_result.jpg"),
        depth_colormap
    )

    cv2.imwrite(
        str(OUTPUT_DIR / "lane_result.jpg"),
        roi
    )

    cv2.imwrite(
        str(OUTPUT_DIR / "edge_result.jpg"),
        edges
    )

    # ======================================
    # 7. FINAL SUMMARY
    # ======================================

    print("\n====================================")
    print(" PIPELINE COMPLETED SUCCESSFULLY!")
    print("====================================")

    print("Object Detection : Completed")
    print("Tracking API     : Invoked")
    print("Lane Detection   : Completed")
    print("Depth Estimation : Completed")

    print("\nOutput directory:")
    print(OUTPUT_DIR)

    print("\nGenerated files:")

    print("1. perception_result.jpg")
    print("2. depth_result.jpg")
    print("3. lane_result.jpg")
    print("4. edge_result.jpg")

    print("\n====================================")


# ==========================================
# 8. EXECUTION
# ==========================================

if __name__ == "__main__":

    try:
        main()

    except Exception as error:

        print("\n[ERROR] Pipeline execution failed!")

        print("Reason:", error)

        raise
