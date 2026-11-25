
import cv2
import numpy as np
from pathlib import Path
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = ROOT / "yolov8n.pt"
OUTPUT_DIR = ROOT / "simulator" / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def detect_lanes(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 50, 150)

    height, width = edges.shape
    mask = np.zeros_like(edges)

    polygon = np.array([[
        (0, height),
        (width // 2, int(height * 0.55)),
        (width, height)
    ]], dtype=np.int32)

    cv2.fillPoly(mask, polygon, 255)
    roi = cv2.bitwise_and(edges, mask)

    lines = cv2.HoughLinesP(
        roi, 1, np.pi / 180, 40,
        minLineLength=30, maxLineGap=100
    )

    if lines is not None:
        lines = np.asarray(lines).reshape(-1, 4)
        for x1, y1, x2, y2 in lines:
            cv2.line(
                frame,
                (int(x1), int(y1)),
                (int(x2), int(y2)),
                (0, 255, 0), 2
            )

    return frame

def main():
    print("Loading YOLO model...")
    model = YOLO(str(MODEL_PATH))

    camera = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    if not camera.isOpened():
        print("ERROR: Webcam not available!")
        return

    print("Live perception started. Press Q to quit.")

    frame_count = 0
    video_path = OUTPUT_DIR / "live_perception.avi"

    writer = cv2.VideoWriter(
        str(video_path),
        cv2.VideoWriter_fourcc(*"XVID"),
        10,
        (640, 480)
    )

    try:
        while True:
            success, frame = camera.read()

            if not success:
                print("Unable to read webcam frame.")
                break

            frame = cv2.resize(frame, (640, 480))

            results = model.track(
                source=frame,
                persist=True,
                tracker="bytetrack.yaml",
                device="cpu",
                imgsz=416,
                conf=0.35,
                verbose=False
            )

            annotated = results[0].plot()
            annotated = detect_lanes(annotated)

            count = (
                len(results[0].boxes)
                if results[0].boxes is not None
                else 0
            )

            cv2.putText(
                annotated,
                f"Objects detected: {count}",
                (15, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7, (0, 255, 255), 2
            )

            cv2.putText(
                annotated,
                "LIVE AUTONOMOUS PERCEPTION",
                (15, 460),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55, (255, 255, 255), 2
            )

            writer.write(annotated)
            cv2.imshow("Autonomous Driving Perception", annotated)

            frame_count += 1

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    finally:
        camera.release()
        writer.release()
        cv2.destroyAllWindows()

    print("Frames processed:", frame_count)
    print("Video saved:", video_path)
    print("Live perception test completed!")

if __name__ == "__main__":
    main()
