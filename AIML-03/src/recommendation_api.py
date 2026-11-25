
from fastapi import FastAPI, HTTPException
from fastapi.responses import RedirectResponse
from pydantic import BaseModel, Field
import sys
from pathlib import Path

# Allow imports from the src folder
SRC_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SRC_DIR))

from two_stage_recommendation import recommend_two_stage

app = FastAPI(
    title="Personalized Movie Recommendation API",
    description="Two-stage recommendation using FAISS, Wide & Deep ranking and Redis caching.",
    version="1.0.0"
)


class RecommendationRequest(BaseModel):
    user_id: int = Field(..., gt=0)
    top_k: int = Field(default=10, ge=1, le=50)


@app.get("/")
def home():
    return {
        "message": "Recommendation API is running",
        "docs": "/docs"
    }


@app.get("/health")
def health():
    return {"status": "healthy"}


@app.post("/recommend")
def get_recommendations(request: RecommendationRequest):
    try:
        results = recommend_two_stage(request.user_id)

        if results is None or results.empty:
            raise HTTPException(
                status_code=404,
                detail="No recommendations found for this user."
            )

        # Return requested number of recommendations
        results = results.head(request.top_k)

        return {
            "user_id": request.user_id,
            "count": len(results),
            "recommendations": results.to_dict(orient="records")
        }

    except HTTPException:
        raise
    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=str(error)
        )
uvicorn recommendation_api:app --app-dir src --reload