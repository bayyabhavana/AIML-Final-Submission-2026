import os
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRAIN_PATH = os.path.join(BASE_DIR, "data", "processed", "train.csv")
TEST_PATH = os.path.join(BASE_DIR, "data", "processed", "test.csv")
MOVIES_PATH = os.path.join(BASE_DIR, "data", "raw", "ml-latest-small", "movies.csv")
MODEL_DIR = os.path.join(BASE_DIR, "models", "ranking")
MODEL_PATH = os.path.join(MODEL_DIR, "wide_deep_model.pth")

BATCH_SIZE = 256
EPOCHS = 5
EMBEDDING_SIZE = 32
LEARNING_RATE = 0.001

os.makedirs(MODEL_DIR, exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class WideDeepModel(nn.Module):
    def __init__(self, number_of_users, number_of_movies, number_of_genres, embedding_size=32):
        super().__init__()
        self.user_embedding = nn.Embedding(number_of_users, embedding_size)
        self.movie_embedding = nn.Embedding(number_of_movies, embedding_size)
        self.deep_network = nn.Sequential(
            nn.Linear(embedding_size * 2 + number_of_genres, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )
        self.user_bias = nn.Embedding(number_of_users, 1)
        self.movie_bias = nn.Embedding(number_of_movies, 1)
        self.genre_linear = nn.Linear(number_of_genres, 1, bias=False)
        self.global_bias = nn.Parameter(torch.zeros(1))

    def forward(self, user_ids, movie_ids, genre_vectors):
        user_vector = self.user_embedding(user_ids)
        movie_vector = self.movie_embedding(movie_ids)
        combined = torch.cat([user_vector, movie_vector, genre_vectors], dim=1)
        deep_output = self.deep_network(combined)
        wide_output = (
            self.user_bias(user_ids)
            + self.movie_bias(movie_ids)
            + self.genre_linear(genre_vectors)
            + self.global_bias
        )
        raw_score = (deep_output + wide_output).squeeze(1)
        return 0.5 + 4.5 * torch.sigmoid(raw_score)


def main():
    print("=" * 60)
    print("WIDE & DEEP RANKING MODEL - BOUNDED RATING OUTPUT")
    print("=" * 60)
    print("Device:", device)

    train = pd.read_csv(TRAIN_PATH)
    test = pd.read_csv(TEST_PATH)
    movies = pd.read_csv(MOVIES_PATH)

    required = {"userId", "movieId", "rating"}
    for name, frame in [("train", train), ("test", test)]:
        missing = required - set(frame.columns)
        if missing:
            raise ValueError(f"{name}.csv missing columns: {missing}")
        for col in ["userId", "movieId", "rating"]:
            frame[col] = pd.to_numeric(frame[col], errors="coerce")
        frame.dropna(subset=["userId", "movieId", "rating"], inplace=True)
        frame["userId"] = frame["userId"].astype(int)
        frame["movieId"] = frame["movieId"].astype(int)
        if not frame["rating"].between(0.5, 5.0).all():
            raise ValueError(f"{name} ratings must be between 0.5 and 5.0.")

    users = sorted(train["userId"].unique())
    movie_ids = sorted(train["movieId"].unique())
    user_map = {int(v): i for i, v in enumerate(users)}
    movie_map = {int(v): i for i, v in enumerate(movie_ids)}

    train = train[train["userId"].isin(user_map) & train["movieId"].isin(movie_map)].copy()
    test = test[test["userId"].isin(user_map) & test["movieId"].isin(movie_map)].copy()
    if train.empty or test.empty:
        raise ValueError("Train or test data is empty after ID filtering.")

    train["user_index"] = train["userId"].map(user_map).astype(int)
    train["movie_index"] = train["movieId"].map(movie_map).astype(int)
    test["user_index"] = test["userId"].map(user_map).astype(int)
    test["movie_index"] = test["movieId"].map(movie_map).astype(int)

    movies["movieId"] = pd.to_numeric(movies["movieId"], errors="coerce")
    movies.dropna(subset=["movieId"], inplace=True)
    movies["movieId"] = movies["movieId"].astype(int)

    genre_set = set()
    for value in movies["genres"].fillna(""):
        for genre in str(value).split("|"):
            if genre and genre != "(no genres listed)":
                genre_set.add(genre)
    genre_names = sorted(genre_set)
    genre_map = {name: i for i, name in enumerate(genre_names)}

    genre_features = np.zeros((len(movie_map), len(genre_names)), dtype=np.float32)
    for _, row in movies.iterrows():
        movie_id = int(row["movieId"])
        if movie_id not in movie_map:
            continue
        idx = movie_map[movie_id]
        for genre in str(row["genres"]).split("|"):
            if genre in genre_map:
                genre_features[idx, genre_map[genre]] = 1.0

    train_users = torch.tensor(train["user_index"].to_numpy(), dtype=torch.long)
    train_movies = torch.tensor(train["movie_index"].to_numpy(), dtype=torch.long)
    train_genres = torch.tensor(genre_features[train["movie_index"].to_numpy()], dtype=torch.float32)
    train_ratings = torch.tensor(train["rating"].to_numpy(dtype=np.float32), dtype=torch.float32)

    test_users = torch.tensor(test["user_index"].to_numpy(), dtype=torch.long)
    test_movies = torch.tensor(test["movie_index"].to_numpy(), dtype=torch.long)
    test_genres = torch.tensor(genre_features[test["movie_index"].to_numpy()], dtype=torch.float32)
    test_ratings = test["rating"].to_numpy(dtype=np.float32)

    train_loader = DataLoader(
        TensorDataset(train_users, train_movies, train_genres, train_ratings),
        batch_size=BATCH_SIZE, shuffle=True
    )
    test_loader = DataLoader(
        TensorDataset(test_users, test_movies, test_genres),
        batch_size=BATCH_SIZE, shuffle=False
    )

    model = WideDeepModel(len(user_map), len(movie_map), len(genre_names), EMBEDDING_SIZE).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    print("Training records:", len(train))
    print("Testing records:", len(test))
    print("Users:", len(user_map), "| Movies:", len(movie_map), "| Genres:", len(genre_names))

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0.0
        for batch_users, batch_movies, batch_genres, batch_ratings in train_loader:
            batch_users = batch_users.to(device)
            batch_movies = batch_movies.to(device)
            batch_genres = batch_genres.to(device)
            batch_ratings = batch_ratings.to(device)

            predictions = model(batch_users, batch_movies, batch_genres)
            loss = criterion(predictions, batch_ratings)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"Epoch {epoch + 1}/{EPOCHS} | Training MSE: {total_loss / len(train_loader):.4f}")

    model.eval()
    prediction_batches = []
    with torch.no_grad():
        for batch_users, batch_movies, batch_genres in test_loader:
            output = model(batch_users.to(device), batch_movies.to(device), batch_genres.to(device))
            prediction_batches.extend(output.cpu().numpy().tolist())

    predictions = np.asarray(prediction_batches, dtype=np.float32)
    if len(predictions) != len(test_ratings):
        raise RuntimeError("Prediction count does not match test records.")

    rmse = float(np.sqrt(np.mean((test_ratings - predictions) ** 2)))
    mae = float(np.mean(np.abs(test_ratings - predictions)))
    print("\nEvaluation")
    print("Records:", len(test_ratings))
    print(f"RMSE: {rmse:.4f} | MAE: {mae:.4f}")
    print(f"Prediction range: {predictions.min():.4f} to {predictions.max():.4f}")
    print("All predictions in 0.5-5.0:", bool(np.all((predictions >= 0.5) & (predictions <= 5.0))))

    checkpoint = {
        "model_state_dict": model.state_dict(),
        "number_of_users": len(user_map),
        "number_of_movies": len(movie_map),
        "number_of_genres": len(genre_names),
        "embedding_size": EMBEDDING_SIZE,
        "user_map": user_map,
        "movie_map": movie_map,
        "genre_names": genre_names,
        "metrics": {"RMSE": rmse, "MAE": mae},
        "output_range": [0.5, 5.0],
    }
    torch.save(checkpoint, MODEL_PATH)

    with open(os.path.join(MODEL_DIR, "user_mapping.json"), "w", encoding="utf-8") as f:
        json.dump(user_map, f, indent=2)
    with open(os.path.join(MODEL_DIR, "movie_mapping.json"), "w", encoding="utf-8") as f:
        json.dump(movie_map, f, indent=2)

    print("\nModel saved:", MODEL_PATH)
    print("WIDE & DEEP TRAINING COMPLETED.")


if __name__ == "__main__":
    main()
