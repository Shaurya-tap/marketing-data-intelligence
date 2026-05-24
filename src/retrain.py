"""
retrain.py
----------
Automated RETRAINING entry point (the assignment's "automated retraining"
nice-to-have).

Idea: in production you'd run this on a schedule (e.g. a nightly cron job or a
GitHub Action). It re-reads the latest data and retrains the model, so the model
keeps up as new products and reviews arrive. If retraining is triggered by drift
alerts from monitoring.py, that's a simple "closed loop" MLOps setup.

Run:  python src/retrain.py
"""

import train_model

if __name__ == "__main__":
    print("=== Automated retraining started ===")
    train_model.main()
    print("=== Retraining complete. New model + metrics saved. ===")
