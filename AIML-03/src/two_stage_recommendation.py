import sys
import json
import urllib.request
import urllib.error
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import redis

BASE_DIR = Path(__file__).resolve().parent.parent
SRC_DIR = BASE_DIR / "src"
sys.path.insert(0, str(SRC_DIR))

from faiss_retrieval import load_trained_model, build_faiss_index, recommend_movies

RANKING_MODEL_PATH = BASE_DIR / "models" / "ranking" / "wide_deep_model.pth"
MOVIES_PATH = BASE_DIR / "data" / "raw" / "ml-latest-small" / "movies.csv"
DEVICE = torch.device("cpu")

CANDIDATE_COUNT = 50
FINAL_TOP_K = 10
CACHE_TTL = 3600
# New prefix prevents old cached recommendations from being reused.
CACHE_KEY_PREFIX = "recommendations:v4:user"

TRITON_URL = "http://localhost:8001"
TRITON_MODEL_NAME = "wide_deep_ranker"

RESULT_COLUMNS = [
    "movieId", "title", "genres", "similarity", "predicted_rating"
]


# REDIS CONNECTION
try:
    redis_client = redis.Redis(
        host="localhost",
        port=6379,
        decode_responses=True,
        socket_connect_timeout=3,
    )
    redis_client.ping()
    REDIS_AVAILABLE = True
    print("Redis connection successful.")
except redis.RedisError as error:
    redis_client = None
    REDIS_AVAILABLE = False
    print("Redis unavailable. Continuing without cache:", error)


def cache_key(user_id):
    return f"{CACHE_KEY_PREFIX}:{int(user_id)}"


def get_cached_recommendations(user_id):
    if not REDIS_AVAILABLE:
        return None
    try:
        cached = redis_client.get(cache_key(user_id))
        if cached:
            print(f"REDIS CACHE HIT: User {user_id}")
            return pd.DataFrame(json.loads(cached), columns=RESULT_COLUMNS)
        print(f"REDIS CACHE MISS: User {user_id}")
        return None
    except (redis.RedisError, ValueError, TypeError) as error:
        print("Cache read error:", error)
        return None


def save_cached_recommendations(user_id, results):
    if not REDIS_AVAILABLE:
        return
    try:
        redis_client.set(
            cache_key(user_id),
            results.to_json(orient="records"),
            ex=CACHE_TTL,
        )
        print(f"Recommendations cached for user {user_id}.")
    except redis.RedisError as error:
        print("Cache write error:", error)


# LOAD WIDE & DEEP MODEL METADATA
def load_ranking_metadata():
    print("\nLoading Wide & Deep ranking metadata...")
    if not RANKING_MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Ranking checkpoint not found: {RANKING_MODEL_PATH}"
        )

    checkpoint = torch.load(
        RANKING_MODEL_PATH,
        map_location=DEVICE,
        weights_only=True,
    )
    user_map = {
        int(key): int(value)
        for key, value in checkpoint["user_map"].items()
    }
    movie_map = {
        int(key): int(value)
        for key, value in checkpoint["movie_map"].items()
    }
    genre_names = checkpoint["genre_names"]

    print("Ranking metadata loaded successfully.")
    print("Users:", len(user_map))
    print("Movies:", len(movie_map))
    print("Genres:", len(genre_names))
    return user_map, movie_map, genre_names


# CREATE GENRE FEATURES
def create_candidate_genre_features(candidates, genre_names):
    if not MOVIES_PATH.exists():
        raise FileNotFoundError(f"Movies file not found: {MOVIES_PATH}")

    movies = pd.read_csv(MOVIES_PATH)
    genre_map = {genre: index for index, genre in enumerate(genre_names)}
    movie_genres = dict(
        zip(
            movies["movieId"].astype(int),
            movies["genres"].fillna("").astype(str),
        )
    )

    features = np.zeros(
        (len(candidates), len(genre_names)),
        dtype=np.float32,
    )
    for row_index, movie_id in enumerate(candidates["movieId"]):
        genres = movie_genres.get(int(movie_id), "")
        for genre in genres.split("|"):
            if genre in genre_map:
                features[row_index, genre_map[genre]] = 1.0
    return features


# TRITON INFERENCE
def predict_with_triton(user_indices, movie_indices, genre_features):
    print("\nSending ranking inputs to Triton...")
    user_indices = np.asarray(user_indices, dtype=np.int64)
    movie_indices = np.asarray(movie_indices, dtype=np.int64)
    genre_features = np.asarray(genre_features, dtype=np.float32)

    if not (
        len(user_indices) == len(movie_indices) == len(genre_features)
    ):
        raise ValueError("Triton input batch sizes do not match.")

    payload = {
        "inputs": [
            {
                "name": "user_ids",
                "shape": [len(user_indices)],
                "datatype": "INT64",
                "data": user_indices.tolist(),
            },
            {
                "name": "movie_ids",
                "shape": [len(movie_indices)],
                "datatype": "INT64",
                "data": movie_indices.tolist(),
            },
            {
                "name": "genre_vectors",
                "shape": [len(genre_features), genre_features.shape[1]],
                "datatype": "FP32",
                "data": genre_features.tolist(),
            },
        ]
    }

    request_data = json.dumps(payload).encode("utf-8")
    url = f"{TRITON_URL}/v2/models/{TRITON_MODEL_NAME}/infer"
    request = urllib.request.Request(
        url,
        data=request_data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            result = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as error:
        details = error.read().decode("utf-8", errors="replace")
        raise RuntimeError(
            f"Triton inference HTTP error {error.code}: {details}"
        ) from error
    except urllib.error.URLError as error:
        raise RuntimeError(
            f"Cannot connect to Triton at {TRITON_URL}: {error}"
        ) from error

    outputs = result.get("outputs", [])
    score_output = next(
        (output for output in outputs if output.get("name") == "scores"),
        None,
    )
    if score_output is None:
        raise RuntimeError(
            "Triton response does not contain the scores output."
        )

    predictions = np.asarray(
        score_output["data"], dtype=np.float32
    ).reshape(-1)

    if len(predictions) != len(user_indices):
        raise RuntimeError(
            "Triton returned an unexpected number of predictions."
        )
    if not np.all(np.isfinite(predictions)):
        raise RuntimeError(
            "Triton returned NaN or infinite prediction values."
        )

    print("Triton inference successful.")
    print("Predictions received:", len(predictions))
    return predictions


# RAW PREDICTION DIAGNOSTICS
def print_prediction_diagnostics(predictions):
    predictions = np.asarray(predictions, dtype=np.float32).reshape(-1)
    if predictions.size == 0:
        print("\nNo predictions received for diagnostics.")
        return

    print("\n" + "-" * 60)
    print("RAW TRITON PREDICTION DIAGNOSTICS (BEFORE CLIPPING)")
    print("-" * 60)
    print(f"Minimum raw prediction: {float(np.min(predictions)):.6f}")
    print(f"Maximum raw prediction: {float(np.max(predictions)):.6f}")
    print(f"Mean raw prediction:    {float(np.mean(predictions)):.6f}")
    print(f"Median raw prediction:  {float(np.median(predictions)):.6f}")
    print(
        f"Predictions >= 5.0:    "
        f"{float(np.mean(predictions >= 5.0) * 100):.2f}%"
    )
    print(
        f"Predictions <= 0.5:    "
        f"{float(np.mean(predictions <= 0.5) * 100):.2f}%"
    )
    print("First 10 raw values:   ", np.round(predictions[:10], 4).tolist())
    print("-" * 60)


# TWO-STAGE RECOMMENDATION PIPELINE
def recommend_two_stage(user_id):
    user_id = int(user_id)
    print("\n" + "=" * 65)
    print("TWO-STAGE PERSONALIZED RECOMMENDATION")
    print("=" * 65)

    cached_results = get_cached_recommendations(user_id)
    if cached_results is not None:
        print(cached_results.to_string(index=False))
        return cached_results

    # STAGE 1: TWO-TOWER + FAISS
    print("\nSTAGE 1: FAISS CANDIDATE RETRIEVAL")
    retrieval_model, retrieval_users, retrieval_movies = load_trained_model()

    if str(user_id) not in retrieval_users:
        print(f"User {user_id} not found in retrieval model.")
        return pd.DataFrame(columns=RESULT_COLUMNS)

    index, faiss_movie_ids = build_faiss_index(
        retrieval_model, retrieval_movies
    )
    candidates = recommend_movies(
        user_id=user_id,
        model=retrieval_model,
        index=index,
        movie_ids=faiss_movie_ids,
        user_mapping=retrieval_users,
        top_k=CANDIDATE_COUNT,
    )

    if candidates is None or candidates.empty:
        print("No candidate movies retrieved.")
        return pd.DataFrame(columns=RESULT_COLUMNS)
    print("Candidates retrieved:", len(candidates))

    # STAGE 2: TRITON WIDE & DEEP RANKING
    print("\nSTAGE 2: TRITON WIDE & DEEP RANKING")
    ranking_users, ranking_movies, genre_names = load_ranking_metadata()

    if user_id not in ranking_users:
        print(f"User {user_id} not found in ranking model.")
        return pd.DataFrame(columns=RESULT_COLUMNS)

    candidates = candidates[
        candidates["movieId"].astype(int).isin(ranking_movies)
    ].copy()
    if candidates.empty:
        print("No candidates available in ranking model.")
        return pd.DataFrame(columns=RESULT_COLUMNS)

    user_index = ranking_users[user_id]
    movie_indices = np.array(
        [ranking_movies[int(movie_id)] for movie_id in candidates["movieId"]],
        dtype=np.int64,
    )
    user_indices = np.full(
        len(candidates), user_index, dtype=np.int64
    )
    genre_features = create_candidate_genre_features(
        candidates, genre_names
    )

    predictions = predict_with_triton(
        user_indices, movie_indices, genre_features
    )

    # Diagnostic values are printed before clipping.
    print_prediction_diagnostics(predictions)

    # Keep ratings within the MovieLens rating range.
    candidates["predicted_rating"] = np.clip(
        predictions, 0.5, 5.0
    )

    final_results = (
        candidates
        .sort_values("predicted_rating", ascending=False)
        .head(FINAL_TOP_K)
        .copy()
    )

    for column in RESULT_COLUMNS:
        if column not in final_results.columns:
            final_results[column] = np.nan

    final_results = (
        final_results[RESULT_COLUMNS].reset_index(drop=True)
    )

    save_cached_recommendations(user_id, final_results)
    print("\nFINAL TOP 10 RECOMMENDATIONS")
    print(final_results.to_string(index=False))
    print("\nTwo-stage recommendation completed successfully.")
    return final_results


# DIRECT TEST AND REDIS CACHE VERIFICATION
if __name__ == "__main__":
    print("\nChecking common users in both models...")
    _, retrieval_users, _ = load_trained_model()
    ranking_users, _, _ = load_ranking_metadata()

    retrieval_user_ids = {
        int(user_id) for user_id in retrieval_users.keys()
    }
    common_users = sorted(
        retrieval_user_ids.intersection(ranking_users.keys())
    )
    if not common_users:
        raise ValueError("No common users found in both models.")

    sample_user = common_users[0]
    print("Testing with user ID:", sample_user)

    print("\nFIRST REQUEST")
    first_results = recommend_two_stage(sample_user)

    print("\nSECOND REQUEST (REDIS CACHE CHECK)")
    second_results = recommend_two_stage(sample_user)

    first_check = first_results.copy().reset_index(drop=True)
    second_check = second_results.copy().reset_index(drop=True)

    for dataframe in [first_check, second_check]:
        if not dataframe.empty:
            dataframe["movieId"] = pd.to_numeric(
                dataframe["movieId"]
            ).astype("int64")

            for column in ["similarity", "predicted_rating"]:
                dataframe[column] = (
                    pd.to_numeric(dataframe[column])
                    .astype("float64")
                    .round(6)
                )

            for column in ["title", "genres"]:
                dataframe[column] = (
                    dataframe[column].fillna("").astype(str)
                )

    if first_check.equals(second_check):
        print("\nCACHE VERIFICATION: PASSED")
    else:
        print("\nCACHE VERIFICATION: FAILED")
