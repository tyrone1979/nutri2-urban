#!/usr/bin/env python3
"""Urban probability threshold tuning for 3-class XGBoost (masked 30% labels)."""
import os

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

from model import DataPipeline


def predict_with_urban_threshold(proba, urban_threshold):
    pred = np.argmax(proba, axis=1)
    urban_mask = proba[:, 2] >= urban_threshold
    pred = pred.copy()
    pred[urban_mask] = 2
    return pred


def run_threshold_tuning(missing_rate=0.3, random_state=42):
    data = DataPipeline().load(verbose=False)
    X, y = data.X_test, data.y_test
    model = joblib.load("./saved_models/Balanced_XGBoost.pkl")
    proba = model.predict_proba(X)

    np.random.seed(random_state)
    miss = np.random.choice([True, False], size=len(y), p=[missing_rate, 1 - missing_rate])

    rows = []
    for thr in [0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60]:
        pred = predict_with_urban_threshold(proba, thr)
        y_imp = y.copy()
        y_imp[miss] = pred[miss]
        urban_true = y == 2
        urban_miss = urban_true & miss
        rural_miss = (y == 0) & miss
        rows.append({
            "urban_threshold": thr,
            "overall_acc": round(accuracy_score(y[miss], y_imp[miss]), 4),
            "macro_f1": round(f1_score(y[miss], y_imp[miss], average="macro"), 4),
            "urban_recall": round((y_imp[urban_miss] == 2).mean(), 4) if urban_miss.sum() else np.nan,
            "urban_precision": round(
                ((y_imp[miss] == 2) & urban_true[miss]).sum() / max(1, (y_imp[miss] == 2).sum()), 4,
            ),
            "rural_recall": round((y_imp[rural_miss] == 0).mean(), 4) if rural_miss.sum() else np.nan,
        })

    df = pd.DataFrame(rows)
    os.makedirs("./results", exist_ok=True)
    df.to_csv("./results/threshold_tuning.csv", index=False)
    print(df.to_string(index=False))
    return df


if __name__ == "__main__":
    run_threshold_tuning()
