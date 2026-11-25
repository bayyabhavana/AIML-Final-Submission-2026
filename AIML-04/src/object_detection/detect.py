
from pathlib import Path
from ultralytics import YOLO

# Project directory
ROOT = Path(__file__).resolve().parents[2]

# Input and output paths
IMAGE_PATH = ROOT / "datasets" / "bus.jpg"
OUTPUT_PATH = ROOT / "outputs"

def main():

    if not IMAGE_PATH.exists():
        print("Error: Image not found:", IMAGE_PATH)
        return

    print("Loading YOLO model...")

    model = YOLO("yolov8n.pt")

    print("Running object detection on CPU...")

    results = model.predict(
        source=str(IMAGE_PATH),
        device="cpu",
        classes=[0, 2, 5, 7],
        conf=0.25,
        save=True,
        project=str(OUTPUT_PATH),
        name="object_detection",
        exist_ok=True
    )

    print("\nDetection Results:")

    for result in results:
        for class_id in result.boxes.cls.tolist():
            print("Detected:", result.names[int(class_id)])

    print("\nDetection completed successfully!")
    print("Results saved in:", OUTPUT_PATH / "object_detection")

if __name__ == "__main__":
    main()
