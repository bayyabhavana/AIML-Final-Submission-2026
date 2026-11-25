
import pandas as pd
import wandb
from pathlib import Path

# Project location
ROOT = Path(__file__).resolve().parents[2]

# Existing benchmark results
CSV_PATH = ROOT / "outputs" / "benchmark" / "benchmark_results.csv"


def main():

    print("Starting W&B experiment tracking...")

    if not CSV_PATH.exists():
        print("Benchmark CSV file not found:", CSV_PATH)
        return

    # Read actual benchmark results
    df = pd.read_csv(CSV_PATH)

    print("\nBenchmark results:")
    print(df)

    # Start W&B experiment
    run = wandb.init(
        project="autonomous-driving-perception",
        name="yolov8n-cpu-benchmark",
        job_type="model-benchmark",
        config={
            "model": "YOLOv8n",
            "device": "CPU",
            "experiment": "PyTorch and ONNX comparison"
        }
    )

    # Upload benchmark table
    wandb.log({
        "benchmark_results": wandb.Table(dataframe=df)
    })

    # Log numeric metrics from the CSV
    numeric_columns = df.select_dtypes(include="number").columns

    for column in numeric_columns:
        values = df[column].dropna()

        if not values.empty:
            wandb.log({
                f"mean_{column}": float(values.mean())
            })

    run.finish()

    print("\nW&B logging completed successfully!")


if __name__ == "__main__":
    main()
