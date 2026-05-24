"""
test_api.py
-----------
Basic tests (the assignment's "unit / integration testing" requirement).

Run from the project root:
  pytest -q

These tests use FastAPI's TestClient, which calls the API in-process — no need
to start a real server. The RAG/LLM endpoint is tested separately (and skipped
if the heavy ML libraries / model downloads aren't available).
"""

import os
import sys
import importlib
import pytest

# Make 'src' importable
SRC = os.path.join(os.path.dirname(__file__), "..", "src")
sys.path.insert(0, SRC)

from fastapi.testclient import TestClient  # noqa: E402
import app as app_module  # noqa: E402

client = TestClient(app_module.app)


def test_health():
    r = client.get("/")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_metrics_present():
    r = client.get("/metrics")
    assert r.status_code == 200
    body = r.json()
    # If the model is trained, we expect R2 to be a sensible number.
    if "r2" in body:
        assert 0.0 <= body["r2"] <= 1.0


def test_predict_returns_valid_range():
    r = client.post("/predict_discount", json={
        "actual_price": 1500, "rating": 4.2,
        "rating_count": 3200, "main_category": "Electronics",
    })
    assert r.status_code == 200
    pred = r.json()["predicted_discount_percentage"]
    assert 0.0 <= pred <= 80.0  # discount % must be in a sane range


def test_predict_rejects_bad_rating():
    # rating of 9 is invalid (must be 0-5) -> FastAPI should reject it (422)
    r = client.post("/predict_discount", json={
        "actual_price": 1500, "rating": 9,
        "rating_count": 3200, "main_category": "Electronics",
    })
    assert r.status_code == 422


def test_drift_flag_on_extreme_input():
    r = client.post("/predict_discount", json={
        "actual_price": 999999, "rating": 4.2,
        "rating_count": 3200, "main_category": "Electronics",
    })
    assert r.json()["drift_warning"] is True


@pytest.mark.skipif(
    importlib.util.find_spec("sentence_transformers") is None,
    reason="RAG libraries not installed",
)
def test_answer_question_smoke():
    r = client.post("/answer_question", json={
        "question": "Which product is good for calls?", "k": 2,
    })
    assert r.status_code == 200
    assert "answer" in r.json()
    assert len(r.json()["sources"]) > 0
