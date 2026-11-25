
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


# =====================================================
# 1. CONFIGURATION
# =====================================================

BASE_DIR = Path(__file__).resolve().parent.parent

TRAIN_FILE = BASE_DIR / "data" / "processed" / "train.csv"
TEST_FILE = BASE_DIR / "data" / "processed" / "test.csv"
MOVIES_FILE = BASE_DIR / "data" / "raw" / "ml-latest-small" / "movies.csv"

MODEL_FILE = BASE_DIR / "models" / "ranking" / "wide_deep_model.pth"

REPORT_DIR = BASE_DIR / "reports"
REPORT_DIR.mkdir(parents=True, exist_ok=True)

RESULT_FILE = REPORT_DIR / "ranking_evaluation_predictions.csv"
METRICS_FILE = REPORT_DIR / "ranking_evaluation_metrics.json"

BATCH_SIZE = 512
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

TOP_K = 10
RELEVANCE_THRESHOLD = 4.0


# =====================================================
# 2. WIDE & DEEP MODEL
# =====================================================

class WideDeepModel(nn.Module):

    def __init__(
        self,
        number_of_users,
        number_of_movies,
        number_of_genres,
        embedding_size=32
    ):
        super().__init__()

        self.user_embedding = nn.Embedding(
            number_of_users, embedding_size
        )

        self.movie_embedding = nn.Embedding(
            number_of_movies, embedding_size
        )

        self.deep_network = nn.Sequential(
            nn.Linear(
                embedding_size * 2 + number_of_genres, 128
            ),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

        self.user_bias = nn.Embedding(number_of_users, 1)
        self.movie_bias = nn.Embedding(number_of_movies, 1)

        self.genre_linear = nn.Linear(
            number_of_genres, 1, bias=False
        )

        self.global_bias = nn.Parameter(torch.zeros(1))

    def forward(self, user_ids, movie_ids, genre_vectors):

        user_vector = self.user_embedding(user_ids)
        movie_vector = self.movie_embedding(movie_ids)

        combined = torch.cat(
            [user_vector, movie_vector, genre_vectors],
            dim=1
        )

        deep_output = self.deep_network(combined)

        wide_output = (
            self.user_bias(user_ids)
            + self.movie_bias(movie_ids)
            + self.genre_linear(genre_vectors)
            + self.global_bias
        )

        return (deep_output + wide_output).squeeze(1)


# =====================================================
# 3. LOAD CHECKPOINT
# =====================================================

def load_model():

    print("\nLoading trained Wide & Deep model...")

    if not MODEL_FILE.exists():
        raise FileNotFoundError(
            f"Model checkpoint not found: {MODEL_FILE}"
        )

    checkpoint = torch.load(
        MODEL_FILE,
        map_location=DEVICE,
        weights_only=True
    )

    user_map = {
        int(k): int(v)
        for k, v in checkpoint["user_map"].items()
    }

    movie_map = {
        int(k): int(v)
        for k, v in checkpoint["movie_map"].items()
    }

    genre_names = checkpoint["genre_names"]

    model = WideDeepModel(
        number_of_users=checkpoint["number_of_users"],
        number_of_movies=checkpoint["number_of_movies"],
        number_of_genres=checkpoint["number_of_genres"],
        embedding_size=checkpoint["embedding_size"]
    ).to(DEVICE)

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    print("Model loaded successfully.")
    print("Users:", len(user_map))
    print("Movies:", len(movie_map))
    print("Genres:", len(genre_names))
    print("Device:", DEVICE)

    return model, user_map, movie_map, genre_names


# =====================================================
# 4. PREPARE TEST DATA
# =====================================================

def prepare_test_data(user_map, movie_map, genre_names):

    print("\nLoading test dataset...")

    test = pd.read_csv(TEST_FILE)
    movies = pd.read_csv(MOVIES_FILE)

    required = {"userId", "movieId", "rating"}

    if not required.issubset(test.columns):
        raise ValueError("Test CSV is missing required columns.")

    for col in ["userId", "movieId", "rating"]:
        test[col] = pd.to_numeric(test[col], errors="coerce")

    test.dropna(subset=list(required), inplace=True)

    test["userId"] = test["userId"].astype(int)
    test["movieId"] = test["movieId"].astype(int)

    # Keep only IDs known to the trained model.
    test = test[
        test["userId"].isin(user_map)
        & test["movieId"].isin(movie_map)
    ].copy()

    if test.empty:
        raise ValueError(
            "No test records match the trained model mappings."
        )

    movies["movieId"] = pd.to_numeric(
        movies["movieId"], errors="coerce"
    )

    movies.dropna(subset=["movieId"], inplace=True)
    movies["movieId"] = movies["movieId"].astype(int)

    # Create genre features in checkpoint order.
    genre_features = np.zeros(
        (len(movie_map), len(genre_names)),
        dtype=np.float32
    )

    genre_map = {
        name: index
        for index, name in enumerate(genre_names)
    }

    for _, row in movies.iterrows():

        movie_id = int(row["movieId"])

        if movie_id not in movie_map:
            continue

        movie_index = movie_map[movie_id]

        for genre in str(row["genres"]).split("|"):

            if genre in genre_map:
                genre_features[
                    movie_index, genre_map[genre]
                ] = 1.0

    test["user_index"] = test["userId"].map(user_map)
    test["movie_index"] = test["movieId"].map(movie_map)

    test["user_index"] = test["user_index"].astype(int)
    test["movie_index"] = test["movie_index"].astype(int)

    print("Usable test records:", len(test))

    return test, genre_features


# =====================================================
# 5. RANKING METRICS: RECALL@K AND NDCG@K
# =====================================================

def calculate_ranking_metrics(
    test,
    predictions,
    k=TOP_K,
    relevance_threshold=RELEVANCE_THRESHOLD
):

    print("\nCalculating ranking metrics...")

    results = test[
        ["userId", "movieId", "rating"]
    ].copy()

    results["predicted_rating"] = predictions

    recalls = []
    ndcgs = []

    for user_id, group in results.groupby("userId"):

        relevant_count = int(
            (group["rating"] >= relevance_threshold).sum()
        )

        # Skip users with no relevant test movies.
        if relevant_count == 0:
            continue

        ranked = group.sort_values(
            "predicted_rating",
            ascending=False
        ).head(k)

        relevance = (
            ranked["rating"].to_numpy() >= relevance_threshold
        ).astype(float)

        hits = int(relevance.sum())

        recall = hits / relevant_count

        discounts = np.log2(
            np.arange(2, len(relevance) + 2)
        )

        dcg = float(
            np.sum(relevance / discounts)
        )

        ideal_count = min(relevant_count, k)

        ideal_relevance = np.ones(ideal_count)

        ideal_discounts = np.log2(
            np.arange(2, ideal_count + 2)
        )

        ideal_dcg = float(
            np.sum(ideal_relevance / ideal_discounts)
        )

        ndcg = (
            dcg / ideal_dcg
            if ideal_dcg > 0
            else 0.0
        )

        recalls.append(recall)
        ndcgs.append(ndcg)

    if not recalls:
        raise ValueError(
            "No users have relevant movies in the test dataset."
        )

    recall_at_k = float(np.mean(recalls))
    ndcg_at_k = float(np.mean(ndcgs))

    print("Users evaluated for ranking:", len(recalls))
    print(f"Recall@{k}: {recall_at_k:.4f}")
    print(f"NDCG@{k}:   {ndcg_at_k:.4f}")

    return {
        f"recall_at_{k}": recall_at_k,
        f"ndcg_at_{k}": ndcg_at_k,
        "ranking_users_evaluated": len(recalls),
        "relevance_threshold": relevance_threshold,
        "ranking_candidate_scope": "Per-user held-out test records"
    }


# =====================================================
# 6. EVALUATE MODEL
# =====================================================

def evaluate(model, test, genre_features):

    print("\nEvaluating model predictions...")

    users = torch.tensor(
        test["user_index"].to_numpy(),
        dtype=torch.long
    )

    movies = torch.tensor(
        test["movie_index"].to_numpy(),
        dtype=torch.long
    )

    genres = torch.tensor(
        genre_features[test["movie_index"].to_numpy()],
        dtype=torch.float32
    )

    ratings = test["rating"].to_numpy(dtype=np.float32)

    dataset = TensorDataset(users, movies, genres)

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False
    )

    predictions = []

    model.eval()

    with torch.no_grad():

        for batch_users, batch_movies, batch_genres in loader:

            batch_users = batch_users.to(DEVICE)
            batch_movies = batch_movies.to(DEVICE)
            batch_genres = batch_genres.to(DEVICE)

            output = model(
                batch_users,
                batch_movies,
                batch_genres
            )

            predictions.extend(
                output.cpu().numpy().tolist()
            )

    predictions = np.clip(
        np.array(predictions),
        0.5,
        5.0
    )

    # Rating prediction metrics.
    rmse = float(
        np.sqrt(np.mean((ratings - predictions) ** 2))
    )

    mae = float(
        np.mean(np.abs(ratings - predictions))
    )

    mse = float(
        np.mean((ratings - predictions) ** 2)
    )

    # Ranking metrics.
    ranking_metrics = calculate_ranking_metrics(
        test,
        predictions,
        k=TOP_K,
        relevance_threshold=RELEVANCE_THRESHOLD
    )

    # Baseline: training-set average rating.
    baseline = float(
        pd.read_csv(TRAIN_FILE)["rating"].mean()
    )

    baseline_predictions = np.full_like(ratings, baseline)

    baseline_rmse = float(
        np.sqrt(
            np.mean((ratings - baseline_predictions) ** 2)
        )
    )

    baseline_mae = float(
        np.mean(
            np.abs(ratings - baseline_predictions)
        )
    )

    print("\n" + "=" * 50)
    print("WIDE & DEEP MODEL EVALUATION")
    print("=" * 50)

    print("Evaluated records:", len(ratings))
    print(f"RMSE: {rmse:.4f}")
    print(f"MAE: {mae:.4f}")
    print(f"MSE: {mse:.4f}")

    print("\nBASELINE COMPARISON")
    print("-" * 35)
    print(f"Baseline RMSE: {baseline_rmse:.4f}")
    print(f"Model RMSE:    {rmse:.4f}")
    print(f"Baseline MAE:  {baseline_mae:.4f}")
    print(f"Model MAE:     {mae:.4f}")

    print("\nPREDICTION DISTRIBUTION")
    print("-" * 35)
    print(f"Minimum: {predictions.min():.4f}")
    print(f"Maximum: {predictions.max():.4f}")
    print(f"Mean:    {predictions.mean():.4f}")
    print(f"Std:     {predictions.std():.4f}")

    print(
        "Unique rounded predictions:",
        len(np.unique(np.round(predictions, 2)))
    )

    # Save detailed predictions.
    results = test[
        ["userId", "movieId", "rating"]
    ].copy()

    results["predicted_rating"] = predictions

    results["absolute_error"] = np.abs(
        ratings - predictions
    )

    results.to_csv(RESULT_FILE, index=False)

    metrics = {
        "evaluated_records": len(ratings),
        "rmse": rmse,
        "mae": mae,
        "mse": mse,
        "baseline_rmse": baseline_rmse,
        "baseline_mae": baseline_mae,
        "prediction_min": float(predictions.min()),
        "prediction_max": float(predictions.max()),
        "prediction_mean": float(predictions.mean()),
        "prediction_std": float(predictions.std())
    }

    # Include Recall@10 and NDCG@10.
    metrics.update(ranking_metrics)

    with open(METRICS_FILE, "w") as file:
        json.dump(metrics, file, indent=4)

    print("\nReports saved:")
    print(RESULT_FILE)
    print(METRICS_FILE)

    print("\nEVALUATION COMPLETED SUCCESSFULLY!")


# =====================================================
# 7. MAIN
# =====================================================

if __name__ == "__main__":

    model, user_map, movie_map, genre_names = load_model()

    test, genre_features = prepare_test_data(
        user_map,
        movie_map,
        genre_names
    )

    evaluate(model, test, genre_features)
