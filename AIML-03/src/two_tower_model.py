
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


# --------------------------------------------------
# 1. CONFIGURATION
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent

TRAIN_FILE = BASE_DIR / "data" / "processed" / "train.csv"
TEST_FILE = BASE_DIR / "data" / "processed" / "test.csv"

MODEL_DIR = BASE_DIR / "models" / "retrieval"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

MODEL_FILE = MODEL_DIR / "two_tower_model.pt"
USER_MAP_FILE = MODEL_DIR / "user_mapping.json"
MOVIE_MAP_FILE = MODEL_DIR / "movie_mapping.json"

DEVICE = torch.device("cpu")

EMBEDDING_DIM = 32
BATCH_SIZE = 2048
EPOCHS = 8
LEARNING_RATE = 0.001

torch.manual_seed(42)
np.random.seed(42)
torch.set_num_threads(2)


# --------------------------------------------------
# 2. TWO-TOWER MODEL
# --------------------------------------------------

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

        self._initialize_weights()

    def _initialize_weights(self):
        nn.init.normal_(self.user_embedding.weight, std=0.05)
        nn.init.normal_(self.movie_embedding.weight, std=0.05)

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

        predictions = 0.5 + 4.5 * torch.sigmoid(scores)

        return predictions


# --------------------------------------------------
# 3. TRAINING FUNCTION
# --------------------------------------------------

def train_model():

    print("Loading training and testing datasets...")

    train_df = pd.read_csv(TRAIN_FILE)
    test_df = pd.read_csv(TEST_FILE)

    required_columns = ["userId", "movieId", "rating"]

    for column in required_columns:
        if column not in train_df.columns or column not in test_df.columns:
            raise ValueError(f"Missing required column: {column}")

    print("Training records:", len(train_df))
    print("Testing records:", len(test_df))

    # --------------------------------------------------
    # 4. CREATE ID MAPPINGS
    # --------------------------------------------------

    unique_users = sorted(train_df["userId"].unique())
    unique_movies = sorted(train_df["movieId"].unique())

    user_to_index = {
        int(user_id): index
        for index, user_id in enumerate(unique_users)
    }

    movie_to_index = {
        int(movie_id): index
        for index, movie_id in enumerate(unique_movies)
    }

    train_df["user_index"] = train_df["userId"].map(user_to_index)
    train_df["movie_index"] = train_df["movieId"].map(movie_to_index)

    num_users = len(user_to_index)
    num_movies = len(movie_to_index)

    print("\nUnique users:", num_users)
    print("Unique movies:", num_movies)

    # --------------------------------------------------
    # 5. PREPARE TRAINING TENSORS
    # --------------------------------------------------

    train_users = torch.tensor(
        train_df["user_index"].values,
        dtype=torch.long
    )

    train_movies = torch.tensor(
        train_df["movie_index"].values,
        dtype=torch.long
    )

    train_ratings = torch.tensor(
        train_df["rating"].values,
        dtype=torch.float32
    )

    train_dataset = TensorDataset(
        train_users,
        train_movies,
        train_ratings
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True
    )

    # --------------------------------------------------
    # 6. INITIALIZE MODEL
    # --------------------------------------------------

    model = TwoTowerModel(
        num_users=num_users,
        num_movies=num_movies,
        embedding_dim=EMBEDDING_DIM
    ).to(DEVICE)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE
    )

    criterion = nn.MSELoss()

    print("\nTwo-Tower model initialized.")
    print("Device:", DEVICE)

    # --------------------------------------------------
    # 7. TRAIN MODEL
    # --------------------------------------------------

    print("\nStarting Two-Tower training...")

    for epoch in range(EPOCHS):

        model.train()
        total_loss = 0.0

        for users, movies, ratings in train_loader:

            users = users.to(DEVICE)
            movies = movies.to(DEVICE)
            ratings = ratings.to(DEVICE)

            optimizer.zero_grad()

            predictions = model(users, movies)

            loss = criterion(predictions, ratings)

            loss.backward()
            optimizer.step()

            total_loss += loss.item() * len(users)

        average_loss = total_loss / len(train_dataset)

        print(
            f"Epoch {epoch + 1}/{EPOCHS} "
            f"| Training MSE: {average_loss:.4f}"
        )

    # --------------------------------------------------
    # 8. EVALUATE ON TEST DATA
    # --------------------------------------------------

    model.eval()

    known_test = test_df[
        test_df["userId"].isin(user_to_index)
        & test_df["movieId"].isin(movie_to_index)
    ].copy()

    if len(known_test) == 0:
        raise ValueError("No test records have known users and movies.")

    test_users = torch.tensor(
        known_test["userId"].map(user_to_index).values,
        dtype=torch.long
    )

    test_movies = torch.tensor(
        known_test["movieId"].map(movie_to_index).values,
        dtype=torch.long
    )

    actual_ratings = known_test["rating"].values

    with torch.no_grad():
        predicted_ratings = model(
            test_users,
            test_movies
        ).numpy()

    mae = np.mean(
        np.abs(actual_ratings - predicted_ratings)
    )

    rmse = np.sqrt(
        np.mean((actual_ratings - predicted_ratings) ** 2)
    )

    print("\nMODEL EVALUATION")
    print("------------------------------")
    print("Evaluated test records:", len(known_test))
    print(f"MAE:  {mae:.4f}")
    print(f"RMSE: {rmse:.4f}")

    # --------------------------------------------------
    # 9. SAVE MODEL AND MAPPINGS
    # --------------------------------------------------

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "num_users": num_users,
            "num_movies": num_movies,
            "embedding_dim": EMBEDDING_DIM,
            "mae": float(mae),
            "rmse": float(rmse)
        },
        MODEL_FILE
    )

    with open(USER_MAP_FILE, "w") as file:
        json.dump(user_to_index, file)

    with open(MOVIE_MAP_FILE, "w") as file:
        json.dump(movie_to_index, file)

    print("\nMODEL TRAINING COMPLETED!")
    print("Saved model:", MODEL_FILE)
    print("Saved user mappings:", USER_MAP_FILE)
    print("Saved movie mappings:", MOVIE_MAP_FILE)


# --------------------------------------------------
# 10. RUN TRAINING ONLY WHEN EXECUTED DIRECTLY
# --------------------------------------------------

if __name__ == "__main__":
    train_model()
