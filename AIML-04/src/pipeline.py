
import cv2
import torch
import numpy as np
import time
from pathlib import Path
from ultralytics import YOLO

# Project paths
ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "yolov8n.pt"
OUTPUT_DIR = ROOT / "outputs" / "integrated_pipeline"


def detect_lanes(frame):
    """Detect lane markings using OpenCV."""

    height, width = frame.shape[:2]

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 50, 150)

    mask = np.zeros_like(edges)

    polygon = np.array([[
        (int(width * 0.08), height),
        (int(width * 0.43), int(height * 0.58)),
        (int(width * 0.57), int(height * 0.58)),
        (int(width * 0.95), height)
    ]], dtype=np.int32)

    cv2.fillPoly(mask, polygon, 255)
    roi = cv2.bitwise_and(edges, mask)

    lines = cv2.HoughLinesP(
        roi,
        1,
        np.pi / 180,
        30,
        minLineLength=30,
        maxLineGap=100
    )

    output = frame.copy()

    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = map(
                int, np.asarray(line).reshape(-1)[:4]
            )

            cv2.line(
                output,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                3
            )

    return output


def main():

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading YOLO model...")

    device = "cpu"
    yolo = YOLO(str(MODEL_PATH))

    print("Loading MiDaS model...")

    midas = torch.hub.load(
        "intel-isl/MiDaS",
        "MiDaS_small",
        trust_repo=True
    )

    midas.to(device)
    midas.eval()

    transforms = torch.hub.load(
        "intel-isl/MiDaS",
        "transforms",
        trust_repo=True
    )

    transform = transforms.small_transform

    print("Models loaded successfully!")

    # Start webcam
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Camera unavailable. Please check your webcam.")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    # Output video: original perception + depth map
    output_video = OUTPUT_DIR / "integrated_output.mp4"

    writer = cv2.VideoWriter(
        str(output_video),
        cv2.VideoWriter_fourcc(*"mp4v"),
        10,
        (1280, 480)
    )

    if not writer.isOpened():
        print("Unable to create output video.")
        cap.release()
        return

    frame_count = 0
    cached_depth = np.zeros((480, 640, 3), dtype=np.uint8)

    print("\nIntegrated perception started!")
    print("Press Q to stop the camera.")

    try:
        while True:

            success, frame = cap.read()

            if not success:
                print("Unable to read camera frame.")
                break

            frame = cv2.resize(frame, (640, 480))
            frame_count += 1

            # 1. YOLO + ByteTrack
            results = yolo.track(
                frame,
                persist=True,
                tracker="bytetrack.yaml",
                imgsz=416,
                device=device,
                verbose=False
            )

            annotated = results[0].plot()

            # 2. Lane detection
            annotated = detect_lanes(annotated)

            # 3. MiDaS depth estimation
            # Run every fifth frame to reduce CPU load
            if frame_count % 5 == 0:

                rgb = cv2.cvtColor(
                    frame,
                    cv2.COLOR_BGR2RGB
                )

                batch = transform(rgb).to(device)

                with torch.no_grad():
                    prediction = midas(batch)

                    prediction = torch.nn.functional.interpolate(
                        prediction.unsqueeze(1),
                        size=(480, 640),
                        mode="bicubic",
                        align_corners=False
                    ).squeeze()

                depth = prediction.cpu().numpy()

                depth = cv2.normalize(
                    depth,
                    None,
                    0,
                    255,
                    cv2.NORM_MINMAX
                ).astype(np.uint8)

                cached_depth = cv2.applyColorMap(
                    depth,
                    cv2.COLORMAP_MAGMA
                )

            # 4. Display labels
            cv2.putText(
                annotated,
                "YOLO + ByteTrack + Lane Detection",
                (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 255),
                2
            )

            cv2.putText(
                cached_depth,
                "MiDaS Relative Depth",
                (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2
            )

            # Combine both views
            combined = np.hstack((annotated, cached_depth))

            writer.write(combined)

            cv2.imshow(
                "Autonomous Driving Perception Stack",
                combined
            )

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    finally:
        cap.release()
        writer.release()
        cv2.destroyAllWindows()

    print("\nIntegrated pipeline stopped.")
    print("Frames processed:", frame_count)
    print("Video saved:", output_video)


if __name__ == "__main__":
    main()
