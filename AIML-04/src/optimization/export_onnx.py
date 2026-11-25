
from pathlib import Path
from ultralytics import YOLO
import shutil

# Project directory
ROOT = Path(__file__).resolve().parents[2]

# Existing YOLO model
MODEL_PATH = ROOT / "yolov8n.pt"

# Output directory
OUTPUT_DIR = ROOT / "models"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def main():

    if not MODEL_PATH.exists():
        print("Error: YOLO model not found!")
        return

    print("Loading YOLO model...")

    model = YOLO(str(MODEL_PATH))

    print("Exporting YOLO to ONNX format...")

    exported_path = model.export(
        format="onnx",
        imgsz=416,
        device="cpu",
        simplify=False,
        opset=12
    )

    destination = OUTPUT_DIR / "yolov8n.onnx"

    shutil.copy2(
        exported_path,
        destination
    )

    print("\nONNX export completed successfully!")
    print("Saved model:", destination)

    size = destination.stat().st_size / (1024 * 1024)

    print("Model size:", round(size, 2), "MB")


if __name__ == "__main__":
    main()
