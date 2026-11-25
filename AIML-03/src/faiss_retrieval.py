
import json
from pathlib import Path

import faiss
import numpy as np
import pandas as pd
import torch

from two_tower_model import TwoTowerModel


# --------------------------------------------------
# 1. CONFIGURATION
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_DIR = BASE_DIR / "models" / "retrieval"
DATA_DIR = BASE_DIR / "data" / "processed"

MODEL_FILE = MODEL_DIR / "two_tower_model.pt"
USER_MAP_FILE = MODEL_DIR / "user_mapping.json"
MOVIE_MAP_FILE = MODEL_DIR / "movie_mapping.json"

INDEX_FILE = MODEL_DIR / "movie_index.faiss"
MOVIE_IDS_FILE = MODEL_DIR / "movie_ids.npy"

MOVIES_FILE = (
    BASE_DIR / "data" / "raw" /
    "ml-latest-small" / "movies.csv"
)

TRAIN_FILE = DATA_DIR / "train.csv"

DEVICE = torch.device("cpu")


# --------------------------------------------------
# 2. LOAD TRAINED MODEL
# --------------------------------------------------

def load_trained_model():

    print("Loading trained Two-Tower model...")

    checkpoint = torch.load(
        MODEL_FILE,
        map_location=DEVICE,
        weights_only=True
    )

    with open(USER_MAP_FILE, "r") as file:
        user_mapping = json.load(file)

    with open(MOVIE_MAP_FILE, "r") as file:
        movie_mapping = json.load(file)

    model = TwoTowerModel(
        num_users=checkpoint["num_users"],
        num_movies=checkpoint["num_movies"],
        embedding_dim=checkpoint["embedding_dim"]
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model.to(DEVICE)
    model.eval()

    return model, user_mapping, movie_mapping


# --------------------------------------------------
# 3. BUILD AND SAVE FAISS INDEX
# --------------------------------------------------

def build_faiss_index(model, movie_mapping):

    print("Generating movie embeddings...")

    movie_ids = sorted(
        movie_mapping.keys(),
        key=lambda movie_id: int(movie_mapping[movie_id])
    )

    movie_indices = torch.tensor(
        [int(movie_mapping[mid]) for mid in movie_ids],
        dtype=torch.long
    )

    embedding_batches = []

    # Generate embeddings in batches
    with torch.no_grad():

        for start in range(0, len(movie_indices), 1024):

            batch = movie_indices[start:start + 1024]

            embeddings = model.encode_movies(batch)

            embedding_batches.append(
                embeddings.cpu().numpy()
            )

    embeddings = np.concatenate(
        embedding_batches,
        axis=0
    ).astype("float32")

    # Normalize for cosine similarity
    faiss.normalize_L2(embeddings)

    # Inner product on normalized vectors = cosine similarity
    dimension = embeddings.shape[1]

    index = faiss.IndexFlatIP(dimension)

    index.add(embeddings)

    # Save index and original movie IDs
    faiss.write_index(
        index,
        str(INDEX_FILE)
    )

    np.save(
        MOVIE_IDS_FILE,
        np.array(movie_ids, dtype=np.int64)
    )

    print("\nFAISS INDEX CREATED")
    print("Total movie embeddings:", index.ntotal)
    print("Embedding dimensions:", dimension)

    return index, np.array(movie_ids, dtype=np.int64)


# --------------------------------------------------
# 4. RECOMMEND MOVIES
# --------------------------------------------------

def recommend_movies(
    user_id,
    model,
    index,
    movie_ids,
    user_mapping,
    top_k=10
):

    result_columns = [
        "movieId",
        "title",
        "genres",
        "similarity"
    ]

    user_id = str(user_id)

    # Check unknown user
    if user_id not in user_mapping:

        print("Unknown user. Cold-start fallback required.")

        return pd.DataFrame(columns=result_columns)

    user_index = int(user_mapping[user_id])

    user_tensor = torch.tensor(
        [user_index],
        dtype=torch.long
    )

    # Generate user embedding
    with torch.no_grad():

        user_embedding = (
            model.encode_users(user_tensor)
            .cpu()
            .numpy()
            .astype("float32")
        )

    faiss.normalize_L2(user_embedding)

    # Load movies already seen in training
    train_df = pd.read_csv(TRAIN_FILE)

    watched_movies = set(
        train_df.loc[
            train_df["userId"] == int(user_id),
            "movieId"
        ].astype(int)
    )

    # Search enough candidates to replace watched movies
    search_k = min(
        index.ntotal,
        top_k + len(watched_movies)
    )

    scores, positions = index.search(
        user_embedding,
        search_k
    )

    recommendations = []

    for position, score in zip(
        positions[0],
        scores[0]
    ):

        if position < 0:
            continue

        movie_id = int(movie_ids[position])

        # Remove movies already watched
        if movie_id in watched_movies:
            continue

        recommendations.append({
            "movieId": movie_id,
            "similarity": float(score)
        })

        if len(recommendations) >= top_k:
            break

    result_df = pd.DataFrame(recommendations)

    if result_df.empty:
        return pd.DataFrame(columns=result_columns)

    # Add movie titles and genres
    movies_df = pd.read_csv(MOVIES_FILE)

    result_df = result_df.merge(
        movies_df,
        on="movieId",
        how="left"
    )

    return result_df[result_columns]


# --------------------------------------------------
# 5. MAIN EXECUTION
# --------------------------------------------------

if __name__ == "__main__":

    model, user_mapping, movie_mapping = load_trained_model()

    # Build FAISS index only once
    index, movie_ids = build_faiss_index(
        model,
        movie_mapping
    )

    # Test recommendation
    sample_user = next(iter(user_mapping.keys()))

    print("\nTesting recommendations for user:", sample_user)

    recommendations = recommend_movies(
        user_id=sample_user,
        model=model,
        index=index,
        movie_ids=movie_ids,
        user_mapping=user_mapping,
        top_k=10
    )

    print("\nTOP 10 RECOMMENDED MOVIES")
    print("=" * 70)

    if recommendations.empty:

        print("No unseen recommendations found.")

    else:

        print(
            recommendations.to_string(index=False)
        )

    print("\nFAISS RETRIEVAL TEST COMPLETED!")
