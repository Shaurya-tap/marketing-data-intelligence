# Marketing Data Intelligence

An end-to-end machine learning system for an e-commerce platform. It does two things:

1. **Predicts** the discount percentage a product should have (a number-prediction model).
2. **Answers questions** about products through an AI assistant that reads your real product data (a RAG + LLM chatbot).

Everything is wrapped in a web API (FastAPI) and can run anywhere with Docker.


---


```
marketing-data-intelligence/
├── data/
│   └── amazon_sample.csv        # sample dataset (same columns as the Kaggle one)
├── src/
│   ├── make_sample_data.py      # creates the sample dataset
│   ├── data_prep.py             # cleans the messy raw data into numbers
│   ├── train_model.py           # trains the discount-prediction model
│   ├── predict.py               # loads the model and makes predictions
│   ├── rag_assistant.py         # the AI assistant (retrieval + LLM)
│   ├── monitoring.py            # request logging + data-drift detection
│   ├── retrain.py               # one-command automated retraining
│   └── app.py                   # the FastAPI web server (the endpoints)
├── tests/
│   └── test_api.py              # automated tests
├── models/                      # trained model + metrics + logs land here
├── requirements.txt             # the Python libraries you need
├── Dockerfile                   # recipe to containerize the app
└── docker-compose.yml           # one-command run
```

---

## How to run it (step by step)

### Option A — On your own computer (recommended for learning)

1. **Install Python 3.11+** if you don't have it.

2. **Install the libraries:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Create the data and train the model:**
   ```bash
   python src/make_sample_data.py   # writes data/amazon_sample.csv
   python src/train_model.py        # trains + saves the model, prints RMSE/MAE/R2
   ```

4. **Start the web server:**
   ```bash
   uvicorn src.app:app --reload --port 8000
   ```

5. **Open the interactive test page** in your browser:
   ```
   http://localhost:8000/docs
   ```
   Click any endpoint → "Try it out" → "Execute". No coding needed to test it.

### Option B — With Docker (one command, runs anywhere)

```bash
docker compose up --build
```
Then open http://localhost:8000/docs

---

## Using the real Kaggle dataset (optional)

The project ships with a realistic *sample* dataset so it works immediately.
To use the real one:

1. Download the [Amazon Sales Dataset](https://www.kaggle.com/datasets/karkavelrajaj/amazon-sales-dataset).
2. Save the file as `data/amazon_sample.csv` (replacing the sample), **or** edit the
   `DATA_PATH` in `src/train_model.py` and `src/app.py` to point at it.
3. Re-run `python src/train_model.py`.

The column names match, so nothing else needs to change.

---

## The API endpoints

| Method | Path | What it does |
|--------|------|--------------|
| GET | `/` | Health check — is the service alive? |
| GET | `/metrics` | The model's RMSE / MAE / R² scores |
| POST | `/predict_discount` | Predict a product's discount % |
| POST | `/answer_question` | Answer a product question (RAG + LLM) |

**Example — predict a discount** (using `curl`):
```bash
curl -X POST http://localhost:8000/predict_discount \
  -H "Content-Type: application/json" \
  -d '{"actual_price": 1500, "rating": 4.2, "rating_count": 3200, "main_category": "Electronics"}'
```
Response:
```json
{"predicted_discount_percentage": 43.12, "drift_warning": false, "drift_zscore": 0.46}
```

**Example — ask a question:**
```bash
curl -X POST http://localhost:8000/answer_question \
  -H "Content-Type: application/json" \
  -d '{"question": "Which product is good for calls and noise isolation?", "k": 3}'
```

---

## How the AI assistant works (RAG)

Think of an **open-book exam**:

1. **Retrieve** — your question is compared against every product description to
   find the most relevant ones. We do this with *embeddings* (turning text into
   numbers so similar meanings sit close together).
2. **Augment** — those top product descriptions are pasted into a prompt.
3. **Generate** — an open-source language model (FLAN-T5) writes a natural-language
   answer using *only* that retrieved context, and we return the sources too.

This keeps answers grounded in your real data instead of made-up facts.

---

## Testing

```bash
pytest -q
```
This checks the health endpoint, the prediction range, input validation, and the
drift warning. (The LLM test is skipped automatically if the heavy ML libraries
aren't installed.)

---


| Assignment requirement | Where it lives |
|------------------------|----------------|
| Predict business outcomes (discount) | `train_model.py`, `predict.py` |
| Regression metrics: RMSE, MAE, R² | printed by `train_model.py`, served at `/metrics` |
| AI assistant with open-source LLM | `rag_assistant.py` (FLAN-T5) |
| RAG layer (grounded answers) | `rag_assistant.py` (retrieve → generate) |
| Containerized API | `Dockerfile`, `docker-compose.yml` |
| `/predict_discount` & `/answer_question` | `app.py` |
| Monitoring | `monitoring.py` (request logging) |
| Drift detection | `monitoring.py` (`check_drift`) |
| Automated retraining | `retrain.py` |
| Explainability / observability | sources returned with each answer; logs in `models/requests.log` |
| Unit / integration testing | `tests/test_api.py` |

---

