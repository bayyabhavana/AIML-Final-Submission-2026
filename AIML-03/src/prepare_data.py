
import pandas as pd
from pathlib import Path

# Project paths
BASE_DIR = Path(__file__).resolve().parent.parent
RAW_FILE = BASE_DIR / "data" / "raw" / "ml-latest-small" / "ratings.csv"
PROCESSED_DIR = BASE_DIR / "data" / "processed"

PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

print("Loading ratings...")
ratings = pd.read_csv(RAW_FILE)

# Basic cleaning
ratings = ratings.dropna()
ratings = ratings.drop_duplicates(subset=["userId", "movieId"])
ratings["timestamp"] = pd.to_datetime(
    ratings["timestamp"], unit="s"
)

# Sort by time
ratings = ratings.sort_values("timestamp").reset_index(drop=True)

# Chronological 80/20 split
split_index = int(len(ratings) * 0.8)

train = ratings.iloc[:split_index].copy()
test = ratings.iloc[split_index:].copy()

# Save datasets
ratings.to_csv(PROCESSED_DIR / "clean_ratings.csv", index=False)
train.to_csv(PROCESSED_DIR / "train.csv", index=False)
test.to_csv(PROCESSED_DIR / "test.csv", index=False)

print("\nDATA PREPARATION COMPLETE")
print("Total ratings:", len(ratings))
print("Training records:", len(train))
print("Testing records:", len(test))
print("Training date range:", train["timestamp"].min(), "to", train["timestamp"].max())
print("Testing date range:", test["timestamp"].min(), "to", test["timestamp"].max())
print("Saved files:", PROCESSED_DIR)
