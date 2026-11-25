# Large-Scale Personalized Recommendation Engine

A two-stage personalized movie recommendation system inspired by platforms such as Netflix and Amazon. It combines deep learning, vector similarity search, ranking models, caching, and model-serving infrastructure.

## Live Demo

**Streamlit Application:**  
https://bayyabhavana-aiml03-recommendation-engine-app-lmzb2.streamlit.app/

## GitHub Repository

https://github.com/bayyabhavana/AIML-03_Large-Scale-Personalized-Recommendation-Engine

## Project Overview

The recommendation engine follows a two-stage architecture:

1. **Candidate Retrieval:** A Two-Tower model generates user and movie embeddings. FAISS performs similarity search to retrieve relevant movie candidates.
2. **Candidate Ranking:** A Wide & Deep model scores and ranks the retrieved movies to generate personalized recommendations.

Redis is integrated to cache recommendation results and reduce repeated computation.

## Technology Stack

- **Programming:** Python
- **Machine Learning:** PyTorch
- **Data Processing:** Pandas, Apache Spark
- **Retrieval:** FAISS
- **Ranking:** Wide & Deep Neural Network
- **Caching:** Redis
- **Model Serving:** NVIDIA Triton Inference Server, ONNX Runtime
- **Workflow Orchestration:** Apache Airflow
- **API / Application:** FastAPI, Streamlit
- **Containerization:** Docker

## Dataset

This implementation uses the MovieLens Latest Small dataset.

- Total ratings: 100,836
- Training records: 80,668
- Test records: 20,168
- Users in training: 522
- Movies in training: 7,867

The dataset was cleaned and split chronologically into training and testing datasets.

**Note:** MovieLens Latest Small was used instead of MovieLens 25M to accommodate the available computing resources.

## Model Architecture

### 1. Two-Tower Retrieval Model

- Learns user and movie embeddings.
- Uses embedding representations for similarity-based retrieval.
- Generates candidate movies for individual users.
- Uses FAISS for efficient nearest-neighbor search.

### 2. Wide & Deep Ranking Model

- Combines wide linear features and deep neural network features.
- Scores candidate movies.
- Ranks candidates to produce the final recommendations.

### 3. FAISS Vector Search

- Creates a movie embedding index.
- Uses inner-product similarity search.
- Retrieves the top candidate movies.

### 4. Redis Recommendation Cache

- Stores generated recommendations.
- Uses user-specific cache keys.
- Applies a one-hour cache expiration.
- Verifies cache hits and misses.

## Evaluation Results

The Wide & Deep model was evaluated on 1,311 known user-movie test records.

| Metric | Model Result |
|---|---:|
| RMSE | 1.2211 |
| MAE | 0.9446 |
| Baseline RMSE | 1.2222 |
| Baseline MAE | 0.8626 |

The model achieved a slightly lower RMSE than the baseline, while its MAE was higher. These are rating-prediction metrics, not Recall@K or NDCG@K ranking metrics.

## Infrastructure and Deployment

- **Apache Airflow:** A recommendation model retraining DAG was executed successfully.
- **Redis:** Recommendation caching was tested successfully.
- **FAISS:** Movie vector index was generated and used for candidate retrieval.
- **NVIDIA Triton:** The `wide_deep_ranker` ONNX model was loaded and reported READY.
- **Docker:** Containers were used to run the supporting services.

## Project Structure

```text
AIML-03-Large-Scale-Personalized-Recommendation-Engine/
│
├── airflow/
│   └── dags/
│       └── recommendation_retraining.py
│
├── data/
│   ├── raw/
│   │   └── ml-latest-small/
│   └── processed/
│       ├── clean_ratings.csv
│       ├── train.csv
│       └── test.csv
│
├── models/
│   ├── ranking/
│   └── retrieval/
│
├── reports/
│
├── src/
│   ├── prepare_data.py
│   ├── two_tower_model.py
│   ├── faiss_retrieval.py
│   ├── ranking_model.py
│   ├── two_stage_recommendation.py
│   ├── evaluate_model.py
│   ├── redis_cache.py
│   └── recommendation_api.py
│
├── triton_model_repository/
├── app.py
├── requirements.txt
├── docker-compose-airflow.yaml
├── Dockerfile.airflow
└── README.md
```

## Installation and Setup

### 1. Clone the repository

```bash
git clone https://github.com/bayyabhavana/AIML-03_Large-Scale-Personalized-Recommendation-Engine.git
cd AIML-03_Large-Scale-Personalized-Recommendation-Engine
```

### 2. Create a virtual environment

```bash
python -m venv .venv
```

Activate on Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Prepare the dataset

Place the MovieLens Latest Small files inside:

```text
data/raw/ml-latest-small/
```

Run:

```bash
python src/prepare_data.py
```

### 5. Train the retrieval model

```bash
python src/two_tower_model.py
```

### 6. Train the ranking model

```bash
python src/ranking_model.py
```

### 7. Run the recommendation pipeline

```bash
python src/two_stage_recommendation.py
```

### 8. Run the Streamlit application

```bash
streamlit run app.py
```

## Future Improvements

- Implement ranking-specific Recall@K and NDCG@K evaluation.
- Improve cold-start recommendations.
- Integrate a complete online A/B testing workflow.
- Improve model performance with larger datasets and additional computing resources.
- Further optimize model serving and recommendation latency.

## Conclusion

This project demonstrates a two-stage personalized recommendation pipeline involving data preparation, deep learning, vector retrieval, candidate ranking, caching, workflow orchestration, and model serving.

It provides a practical implementation of recommendation-system concepts using an accessible MovieLens dataset and containerized infrastructure.

## Author

**Bhavana Goud**

GitHub: https://github.com/bayyabhavana