
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn


# ============================================================
# 1. CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

TRAIN_FILE = BASE_DIR / "data" / "processed" / "train.csv"
TEST_FILE = BASE_DIR / "data" / "processed" / "test.csv"
MOVIES_FILE = BASE_DIR / "data" / "raw" / "ml-latest-small" / "movies.csv"

TWO_TOWER_FILE = BASE_DIR / "models" / "retrieval" / "two_tower_model.pt"
TWO_USER_MAP_FILE = BASE_DIR / "models" / "retrieval" / "user_mapping.json"
TWO_MOVIE_MAP_FILE = BASE_DIR / "models" / "retrieval" / "movie_mapping.json"

WIDE_DEEP_FILE = BASE_DIR / "models" / "ranking" / "wide_deep_model.pth"

REPORT_DIR = BASE_DIR / "reports"
REPORT_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = torch.device("cpu")
K = 10
BATCH_SIZE = 1024
RELEVANT_RATING = 4.0

torch.set_num_threads(2)

print("=" * 65)
print("OFFLINE A/B MODEL COMPARISON")
print("=" * 65)
print("Device:", DEVICE)


# ============================================================
# 2. VALIDATE AND LOAD DATA
# ============================================================

for path in [
    TRAIN_FILE,
    TEST_FILE,
    MOVIES_FILE,
    TWO_TOWER_FILE,
    TWO_USER_MAP_FILE,
    TWO_MOVIE_MAP_FILE,
    WIDE_DEEP_FILE
]:
    if not path.exists():
        raise FileNotFoundError(f"Required file not found: {path}")

train = pd.read_csv(TRAIN_FILE)
test = pd.read_csv(TEST_FILE)
movies = pd.read_csv(MOVIES_FILE)

required = {"userId", "movieId", "rating"}

for name, frame in [("train", train), ("test", test)]:
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"{name} dataset is missing: {missing}")

for frame in [train, test]:
    frame["userId"] = pd.to_numeric(frame["userId"], errors="coerce")
    frame["movieId"] = pd.to_numeric(frame["movieId"], errors="coerce")
    frame["rating"] = pd.to_numeric(frame["rating"], errors="coerce")
    frame.dropna(subset=["userId", "movieId", "rating"], inplace=True)

    frame["userId"] = frame["userId"].astype(int)
    frame["movieId"] = frame["movieId"].astype(int)

movies["movieId"] = pd.to_numeric(
    movies["movieId"], errors="coerce"
)
movies.dropna(subset=["movieId"], inplace=True)
movies["movieId"] = movies["movieId"].astype(int)

print("\nTraining records:", len(train))
print("Testing records:", len(test))
print("Movie records:", len(movies))


# ============================================================
# 3. DEFINE THE ACTUAL TWO-TOWER ARCHITECTURE
# ============================================================

class TwoTowerModel(nn.Module):

    def __init__(self, num_users, num_movies, embedding_dim=32):
        super().__init__()

        self.user_embedding = nn.Embedding(
            num_users, embedding_dim
        )

        self.movie_embedding = nn.Embedding(
            num_movies, embedding_dim
        )

        self.user_tower = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, embedding_dim)
        )

        self.movie_tower = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, embedding_dim)
        )

    def encode_users(self, user_ids):
        embeddings = self.user_embedding(user_ids)
        return self.user_tower(embeddings)

    def encode_movies(self, movie_ids):
        embeddings = self.movie_embedding(movie_ids)
        return self.movie_tower(embeddings)

    def forward(self, user_ids, movie_ids):
        user_vectors = self.encode_users(user_ids)
        movie_vectors = self.encode_movies(movie_ids)

        scores = torch.sum(
            user_vectors * movie_vectors, dim=1
        )

        return 0.5 + 4.5 * torch.sigmoid(scores)


# ============================================================
# 4. DEFINE THE ACTUAL WIDE & DEEP ARCHITECTURE
# ============================================================

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
                embedding_size * 2 + number_of_genres,
                128
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


# ============================================================
# 5. LOAD TWO-TOWER CHECKPOINT
# ============================================================

print("\nLoading Two-Tower model...")

two_checkpoint = torch.load(
    TWO_TOWER_FILE,
    map_location=DEVICE,
    weights_only=True
)

with open(TWO_USER_MAP_FILE, "r", encoding="utf-8") as file:
    two_user_map = {
        int(k): int(v) for k, v in json.load(file).items()
    }

with open(TWO_MOVIE_MAP_FILE, "r", encoding="utf-8") as file:
    two_movie_map = {
        int(k): int(v) for k, v in json.load(file).items()
    }

two_model = TwoTowerModel(
    num_users=two_checkpoint["num_users"],
    num_movies=two_checkpoint["num_movies"],
    embedding_dim=two_checkpoint["embedding_dim"]
)

two_model.load_state_dict(two_checkpoint["model_state_dict"])
two_model.to(DEVICE)
two_model.eval()

print("Two-Tower checkpoint loaded successfully.")
print("Two-Tower users:", len(two_user_map))
print("Two-Tower movies:", len(two_movie_map))


# ============================================================
# 6. LOAD WIDE & DEEP CHECKPOINT
# ============================================================

print("\nLoading Wide & Deep model...")

wide_checkpoint = torch.load(
    WIDE_DEEP_FILE,
    map_location=DEVICE,
    weights_only=True
)

wide_user_map = {
    int(k): int(v)
    for k, v in wide_checkpoint["user_map"].items()
}

wide_movie_map = {
    int(k): int(v)
    for k, v in wide_checkpoint["movie_map"].items()
}

genre_names = wide_checkpoint["genre_names"]

wide_model = WideDeepModel(
    number_of_users=wide_checkpoint["number_of_users"],
    number_of_movies=wide_checkpoint["number_of_movies"],
    number_of_genres=wide_checkpoint["number_of_genres"],
    embedding_size=wide_checkpoint["embedding_size"]
)

wide_model.load_state_dict(wide_checkpoint["model_state_dict"])
wide_model.to(DEVICE)
wide_model.eval()

print("Wide & Deep checkpoint loaded successfully.")
print("Wide & Deep users:", len(wide_user_map))
print("Wide & Deep movies:", len(wide_movie_map))
print("Genre features:", len(genre_names))


# ============================================================
# 7. PREPARE COMMON CANDIDATES AND RELEVANT ITEMS
# ============================================================

common_users = (
    set(two_user_map)
    & set(wide_user_map)
)

common_movies = sorted(
    set(two_movie_map)
    & set(wide_movie_map)
)

print("\nCommon users:", len(common_users))
print("Common movies:", len(common_movies))

# Exclude items already seen in training
seen_by_user = (
    train.groupby("userId")["movieId"]
    .apply(set)
    .to_dict()
)

# Relevant items are held-out movies rated >= 4
positive_test = test[
    (test["rating"] >= RELEVANT_RATING)
    & test["userId"].isin(common_users)
    & test["movieId"].isin(common_movies)
]

relevant_by_user = (
    positive_test.groupby("userId")["movieId"]
    .apply(set)
    .to_dict()
)

# Use identical eligible candidate pools for both models
eligible_users = []

for user_id, relevant_items in relevant_by_user.items():

    seen = seen_by_user.get(user_id, set())

    candidates = [
        movie_id
        for movie_id in common_movies
        if movie_id not in seen
    ]

    relevant = set(relevant_items).intersection(candidates)

    if len(candidates) >= K and relevant:
        eligible_users.append(int(user_id))

if not eligible_users:
    raise ValueError(
        "No eligible users found. Check mappings and train/test data."
    )

print("Eligible evaluation users:", len(eligible_users))
print("Candidate pool is identical for both models.")


# ============================================================
# 8. CREATE GENRE FEATURE LOOKUP
# ============================================================

genre_to_index = {
    genre: index
    for index, genre in enumerate(genre_names)
}

movie_genre_lookup = dict(
    zip(
        movies["movieId"].astype(int),
        movies["genres"].fillna("").astype(str)
    )
)


def build_genre_features(movie_ids):

    features = np.zeros(
        (len(movie_ids), len(genre_names)),
        dtype=np.float32
    )

    for row_index, movie_id in enumerate(movie_ids):

        genres = movie_genre_lookup.get(movie_id, "")

        for genre in genres.split("|"):
            if genre in genre_to_index:
                features[row_index, genre_to_index[genre]] = 1.0

    return features


# ============================================================
# 9. METRIC CALCULATION
# ============================================================

def calculate_ranking_metrics(hits, number_of_relevant):

    hits = np.asarray(hits, dtype=np.float32)

    precision = float(hits.sum() / K)
    recall = float(hits.sum() / number_of_relevant)

    discounts = np.log2(
        np.arange(2, len(hits) + 2)
    )

    dcg = float(np.sum(hits / discounts))

    ideal_hits = min(number_of_relevant, K)

    idcg = float(
        np.sum(
            1.0 / np.log2(
                np.arange(2, ideal_hits + 2)
            )
        )
    )

    ndcg = dcg / idcg if idcg > 0 else 0.0

    return precision, recall, ndcg


# ============================================================
# 10. EVALUATE BOTH MODELS ON THE SAME CANDIDATES
# ============================================================

def evaluate_models():

    results = {
        "Two-Tower": {
            "precision": [],
            "recall": [],
            "ndcg": []
        },
        "Wide & Deep": {
            "precision": [],
            "recall": [],
            "ndcg": []
        }
    }

    for user_number, user_id in enumerate(eligible_users, start=1):

        seen = seen_by_user.get(user_id, set())

        candidates = [
            movie_id
            for movie_id in common_movies
            if movie_id not in seen
        ]

        relevant = (
            set(relevant_by_user[user_id])
            .intersection(candidates)
        )

        # Same external movie IDs for both models
        two_user_index = two_user_map[user_id]
        wide_user_index = wide_user_map[user_id]

        two_movie_indices = [
            two_movie_map[mid] for mid in candidates
        ]

        wide_movie_indices = [
            wide_movie_map[mid] for mid in candidates
        ]

        two_scores = []
        wide_scores = []

        genre_features = build_genre_features(candidates)

        # Batch inference to reduce CPU memory usage
        with torch.no_grad():

            for start in range(0, len(candidates), BATCH_SIZE):

                end = start + BATCH_SIZE

                batch_two_movies = torch.tensor(
                    two_movie_indices[start:end],
                    dtype=torch.long,
                    device=DEVICE
                )

                batch_wide_movies = torch.tensor(
                    wide_movie_indices[start:end],
                    dtype=torch.long,
                    device=DEVICE
                )

                batch_size = len(batch_two_movies)

                two_users = torch.full(
                    (batch_size,),
                    two_user_index,
                    dtype=torch.long,
                    device=DEVICE
                )

                wide_users = torch.full(
                    (batch_size,),
                    wide_user_index,
                    dtype=torch.long,
                    device=DEVICE
                )

                batch_genres = torch.tensor(
                    genre_features[start:end],
                    dtype=torch.float32,
                    device=DEVICE
                )

                score_a = two_model(
                    two_users,
                    batch_two_movies
                )

                score_b = wide_model(
                    wide_users,
                    batch_wide_movies,
                    batch_genres
                )

                two_scores.extend(
                    score_a.cpu().numpy().tolist()
                )

                wide_scores.extend(
                    score_b.cpu().numpy().tolist()
                )

        # Rank both models over the exact same candidate list
        top_a = np.argsort(
            np.asarray(two_scores)
        )[::-1][:K]

        top_b = np.argsort(
            np.asarray(wide_scores)
        )[::-1][:K]

        recommended_a = [
            candidates[position] for position in top_a
        ]

        recommended_b = [
            candidates[position] for position in top_b
        ]

        hits_a = [
            1 if movie_id in relevant else 0
            for movie_id in recommended_a
        ]

        hits_b = [
            1 if movie_id in relevant else 0
            for movie_id in recommended_b
        ]

        metrics_a = calculate_ranking_metrics(
            hits_a, len(relevant)
        )

        metrics_b = calculate_ranking_metrics(
            hits_b, len(relevant)
        )

        for name, metrics in [
            ("Two-Tower", metrics_a),
            ("Wide & Deep", metrics_b)
        ]:

            results[name]["precision"].append(metrics[0])
            results[name]["recall"].append(metrics[1])
            results[name]["ndcg"].append(metrics[2])

        if user_number % 5 == 0:
            print(
                f"Evaluated {user_number}/"
                f"{len(eligible_users)} users"
            )

    return results


# ============================================================
# 11. RUN OFFLINE COMPARISON
# ============================================================

print("\nStarting offline ranking evaluation...")

evaluation = evaluate_models()

rows = []

for model_name, metrics in evaluation.items():

    rows.append({
        "model": model_name,
        "precision_at_10": float(np.mean(metrics["precision"])),
        "recall_at_10": float(np.mean(metrics["recall"])),
        "ndcg_at_10": float(np.mean(metrics["ndcg"])),
        "evaluated_users": len(metrics["ndcg"]),
        "candidate_pool": len(common_movies)
    })

comparison = pd.DataFrame(rows)

comparison.insert(
    0,
    "experiment",
    ["A", "B"]
)

# ============================================================
# 12. SAVE REPORT
# ============================================================

report_file = REPORT_DIR / "ab_testing_results.csv"

comparison.to_csv(
    report_file,
    index=False
)

print("\n" + "=" * 65)
print("OFFLINE MODEL COMPARISON RESULTS")
print("=" * 65)

print(comparison.to_string(index=False))

a = comparison.iloc[0]
b = comparison.iloc[1]

print("\nMODEL COMPARISON")

if a["ndcg_at_10"] > b["ndcg_at_10"]:
    print("Two-Tower has higher NDCG@10.")
elif b["ndcg_at_10"] > a["ndcg_at_10"]:
    print("Wide & Deep has higher NDCG@10.")
else:
    print("Both models have equal NDCG@10.")

print("\nREPORT SAVED:")
print(report_file)

print("\nOFFLINE A/B MODEL COMPARISON COMPLETED.")
print("This is an offline evaluation, not a live user experiment.")
