
import csv
import time
from pathlib import Path
from ultralytics import YOLO

# Project paths
ROOT = Path(__file__).resolve().parents[2]

PT_MODEL = ROOT / "yolov8n.pt"
ONNX_MODEL = ROOT / "models" / "yolov8n.onnx"
IMAGE_PATH = ROOT / "datasets" / "bus.jpg"

OUTPUT_DIR = ROOT / "outputs" / "benchmark"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def benchmark_model(model_path, model_name):

    print(f"\nLoading {model_name}...")

    model = YOLO(str(model_path))

    # Warm-up runs
    for _ in range(3):
        model.predict(
            str(IMAGE_PATH),
            imgsz=416,
            device="cpu",
            verbose=False
        )

    times = []

    # Actual benchmark
    for i in range(20):

        start = time.perf_counter()

        model.predict(
            str(IMAGE_PATH),
            imgsz=416,
            device="cpu",
            verbose=False
        )

        elapsed = time.perf_counter() - start
        times.append(elapsed)

        print(f"Run {i + 1}: {elapsed * 1000:.2f} ms")

    avg_latency = sum(times) / len(times)
    fps = 1 / avg_latency

    return {
        "Model": model_name,
        "Average Latency (ms)": round(avg_latency * 1000, 2),
        "FPS": round(fps, 2)
    }


def main():

    if not PT_MODEL.exists():
        print("PyTorch model not found!")
        return

    if not ONNX_MODEL.exists():
        print("ONNX model not found!")
        return

    if not IMAGE_PATH.exists():
        print("Test image not found!")
        return

    print("Starting model performance comparison...")
    print("Device: CPU")
    print("Image: bus.jpg")
    print("Image size: 416")
    print("Benchmark runs: 20")

    results = []

    results.append(
        benchmark_model(PT_MODEL, "PyTorch YOLOv8n")
    )

    results.append(
        benchmark_model(ONNX_MODEL, "ONNX YOLOv8n")
    )

    # Save results
    csv_path = OUTPUT_DIR / "benchmark_results.csv"

    with open(csv_path, "w", newline="") as file:

        writer = csv.DictWriter(
            file,
            fieldnames=[
                "Model",
                "Average Latency (ms)",
                "FPS"
            ]
        )

        writer.writeheader()
        writer.writerows(results)

    print("\n========== BENCHMARK RESULTS ==========")

    for result in results:
        print(result)

    print("\nResults saved to:", csv_path)


if __name__ == "__main__":
    main()
