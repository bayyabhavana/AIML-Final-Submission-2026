import os
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import onnx


BASE_DIR = Path(__file__).resolve().parent.parent
CHECKPOINT_PATH = BASE_DIR / "models" / "ranking" / "wide_deep_model.pth"
MODEL_DIR = BASE_DIR / "triton_model_repository" / "wide_deep_ranker"
VERSION_DIR = MODEL_DIR / "1"
ONNX_PATH = VERSION_DIR / "model.onnx"
DEVICE = torch.device("cpu")


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


def create_triton_config(number_of_genres):
    config = f'''
name: "wide_deep_ranker"
platform: "onnxruntime_onnx"
max_batch_size: 0

input [
  {{
    name: "user_ids"
    data_type: TYPE_INT64
    dims: [ -1 ]
  }},
  {{
    name: "movie_ids"
    data_type: TYPE_INT64
    dims: [ -1 ]
  }},
  {{
    name: "genre_vectors"
    data_type: TYPE_FP32
    dims: [ -1, {number_of_genres} ]
  }}
]

output [
  {{
    name: "scores"
    data_type: TYPE_FP32
    dims: [ -1 ]
  }}
]

instance_group [
  {{
    count: 1
    kind: KIND_CPU
  }}
]
'''
    config_path = MODEL_DIR / "config.pbtxt"
    config_path.write_text(config.strip() + "\n", encoding="utf-8")
    print("Triton configuration saved:", config_path)


def main():
    print("=" * 60)
    print("WIDE & DEEP MODEL - TRITON ONNX EXPORT")
    print("=" * 60)

    if not CHECKPOINT_PATH.exists():
        raise FileNotFoundError(f"Checkpoint not found: {CHECKPOINT_PATH}")

    VERSION_DIR.mkdir(parents=True, exist_ok=True)
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE, weights_only=True)

    number_of_users = checkpoint["number_of_users"]
    number_of_movies = checkpoint["number_of_movies"]
    number_of_genres = checkpoint["number_of_genres"]
    embedding_size = checkpoint["embedding_size"]

    model = WideDeepModel(
        number_of_users, number_of_movies, number_of_genres, embedding_size
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(DEVICE)
    model.eval()

    dummy_users = torch.tensor([0, 1], dtype=torch.long)
    dummy_movies = torch.tensor([0, 1], dtype=torch.long)
    dummy_genres = torch.zeros((2, number_of_genres), dtype=torch.float32)

    torch.onnx.export(
        model,
        (dummy_users, dummy_movies, dummy_genres),
        str(ONNX_PATH),
        input_names=["user_ids", "movie_ids", "genre_vectors"],
        output_names=["scores"],
        dynamic_axes={
            "user_ids": {0: "batch_size"},
            "movie_ids": {0: "batch_size"},
            "genre_vectors": {0: "batch_size"},
            "scores": {0: "batch_size"},
        },
        opset_version=18,
        dynamo=False,
        external_data=False,
    )

    onnx_model = onnx.load(str(ONNX_PATH))
    onnx.checker.check_model(onnx_model)

    # Optional local PyTorch/ONNX Runtime parity check.
    try:
        import onnxruntime as ort
        test_users = np.array([0, 1], dtype=np.int64)
        test_movies = np.array([0, 1], dtype=np.int64)
        test_genres = np.zeros((2, number_of_genres), dtype=np.float32)
        with torch.no_grad():
            torch_scores = model(
                torch.from_numpy(test_users),
                torch.from_numpy(test_movies),
                torch.from_numpy(test_genres),
            ).numpy()
        session = ort.InferenceSession(str(ONNX_PATH), providers=["CPUExecutionProvider"])
        onnx_scores = session.run(
            ["scores"],
            {
                "user_ids": test_users,
                "movie_ids": test_movies,
                "genre_vectors": test_genres,
            },
        )[0].reshape(-1)
        difference = float(np.max(np.abs(torch_scores - onnx_scores)))
        print(f"PyTorch/ONNX max difference: {difference:.8f}")
        if difference > 1e-4:
            raise RuntimeError("PyTorch and ONNX outputs do not match.")
    except ImportError:
        print("onnxruntime is not installed locally; skipped parity test.")

    create_triton_config(number_of_genres)
    print("\nONNX EXPORT SUCCESSFUL")
    print("Model:", ONNX_PATH)
    print("Model size KB:", round(os.path.getsize(ONNX_PATH) / 1024, 2))
    print("Output scores are constrained to 0.5-5.0.")


if __name__ == "__main__":
    main()
