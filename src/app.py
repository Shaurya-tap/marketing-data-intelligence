"""
app.py
------
The WEB API that exposes everything over HTTP using FastAPI.

Endpoints (as required by the assignment):
  GET  /                 -> health check ("is the service up?")
  GET  /metrics          -> the model's RMSE / MAE / R2 from training
  POST /predict_discount -> predicts a product's discount percentage
  POST /answer_question  -> answers a product question via RAG + LLM

Run locally:
  uvicorn src.app:app --reload --port 8000
Then open http://localhost:8000/docs for an interactive test page.
"""

import os
import sys
import json

# Make sibling modules (predict, monitoring, ...) importable whether this app is
# started as "python src/app.py" or "uvicorn src.app:app" from the project root.
sys.path.insert(0, os.path.dirname(__file__))

from fastapi import FastAPI
from pydantic import BaseModel, Field

from predict import DiscountPredictor
from monitoring import log_event, check_drift

HERE = os.path.dirname(__file__)
DATA_PATH = os.path.join(HERE, "..", "data", "amazon_sample.csv")
METRICS_PATH = os.path.join(HERE, "..", "models", "metrics.json")

app = FastAPI(title="Marketing Data Intelligence API", version="1.0")

# Load the predictor once at startup (fast). The RAG assistant is loaded lazily
# on first question, because loading the LLM is slow and not always needed.
predictor = DiscountPredictor()
_assistant = None


def get_assistant():
    global _assistant
    if _assistant is None:
        from rag_assistant import RAGAssistant
        _assistant = RAGAssistant(DATA_PATH)
        _assistant.load()
    return _assistant


# ---------- request body shapes (FastAPI validates these automatically) ----------
class DiscountRequest(BaseModel):
    actual_price: float = Field(..., json_schema_extra={"example": 1500}, description="Original price")
    rating: float = Field(..., ge=0, le=5, json_schema_extra={"example": 4.2})
    rating_count: float = Field(..., ge=0, json_schema_extra={"example": 3200})
    main_category: str = Field(..., json_schema_extra={"example": "Electronics"})


class QuestionRequest(BaseModel):
    question: str = Field(..., json_schema_extra={"example": "Which speaker is water resistant?"})
    k: int = Field(3, ge=1, le=10, description="How many products to retrieve")


# ---------- endpoints ----------
@app.get("/")
def health():
    return {"status": "ok", "message": "Marketing Data Intelligence API is running"}


@app.get("/metrics")
def metrics():
    if os.path.exists(METRICS_PATH):
        with open(METRICS_PATH) as f:
            return json.load(f)
    return {"detail": "No metrics yet. Run train_model.py."}


@app.post("/predict_discount")
def predict_discount(req: DiscountRequest):
    pred = predictor.predict(
        actual_price=req.actual_price,
        rating=req.rating,
        rating_count=req.rating_count,
        main_category=req.main_category,
    )
    # Monitoring: log the request and check for data drift on price.
    drift, z = check_drift("actual_price_num", req.actual_price)
    log_event("predict_discount", {"input": req.model_dump(),
                                    "prediction": pred, "drift": drift})
    return {
        "predicted_discount_percentage": pred,
        "drift_warning": bool(drift),
        "drift_zscore": z,
    }


@app.post("/answer_question")
def answer_question(req: QuestionRequest):
    bot = get_assistant()
    result = bot.answer(req.question, k=req.k)
    log_event("answer_question", {"question": req.question})
    return result
