
from pathlib import Path
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[2]

OUTPUT_PATH = ROOT / "outputs"

def main():

    print("Loading YOLO model...")

    model = YOLO("yolov8n.pt")

    print("Starting ByteTrack object tracking...")
    print("Using laptop webcam.")
    print("Press Q to stop tracking.")

    results = model.track(
        source=0,
        tracker="bytetrack.yaml",
        persist=True,
        device="cpu",
        classes=[0, 2, 5, 7],
        conf=0.25,
        show=True,
        save=True,
        project=str(OUTPUT_PATH),
        name="object_tracking",
        exist_ok=True
    )

    print("Tracking session finished.")

if __name__ == "__main__":
    main()
