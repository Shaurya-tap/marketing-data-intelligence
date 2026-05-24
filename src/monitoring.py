"""
monitoring.py
-------------
Lightweight MONITORING + DRIFT DETECTION (the assignment's "monitoring" and
"drift detection" requirements), with no external services needed.

Two simple but real ideas:
  1. Request logging: every prediction is appended to a log file with a
     timestamp. In production you'd ship these to a dashboard (Grafana, etc.).
  2. Drift detection: "data drift" means live data starts to look different
     from the data the model was trained on (e.g. prices suddenly much higher).
     We store the training averages, and flag when recent live inputs stray too
     far from them. That's a signal it may be time to RETRAIN.
"""

import os
import json
import time
import threading

HERE = os.path.dirname(__file__)
LOG_DIR = os.path.join(HERE, "..", "models")
LOG_PATH = os.path.join(LOG_DIR, "requests.log")
BASELINE_PATH = os.path.join(LOG_DIR, "baseline.json")

_lock = threading.Lock()  # keep file writes safe across requests


def log_event(kind, payload):
    """Append one JSON line describing a request. Cheap and dependency-free."""
    os.makedirs(LOG_DIR, exist_ok=True)
    record = {"ts": time.strftime("%Y-%m-%dT%H:%M:%S"), "kind": kind, "data": payload}
    with _lock:
        with open(LOG_PATH, "a") as f:
            f.write(json.dumps(record) + "\n")


def save_baseline(df, numeric_cols):
    """Save the training-data averages so we can compare against them later."""
    os.makedirs(LOG_DIR, exist_ok=True)
    baseline = {c: {"mean": float(df[c].mean()), "std": float(df[c].std())}
                for c in numeric_cols}
    with open(BASELINE_PATH, "w") as f:
        json.dump(baseline, f, indent=2)
    return baseline


def check_drift(feature_name, value, z_threshold=3.0):
    """
    Compare a single live value to the training baseline.
    Returns (is_drift, z_score). A z-score above ~3 means the value is unusually
    far from what the model saw during training.
    """
    if not os.path.exists(BASELINE_PATH):
        return False, None
    with open(BASELINE_PATH) as f:
        baseline = json.load(f)
    if feature_name not in baseline:
        return False, None
    mean = baseline[feature_name]["mean"]
    std = baseline[feature_name]["std"] or 1.0
    z = abs((value - mean) / std)
    return (z > z_threshold), round(z, 2)
