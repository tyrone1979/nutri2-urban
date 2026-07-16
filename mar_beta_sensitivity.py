#!/usr/bin/env python3
"""MAR propensity slope sensitivity: beta1 in {1.0, 1.5, 2.0, 2.5}."""
import os

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

from missingness_simulation import simulate_mar_missing
from model import DataPipeline


def main():
    data = DataPipeline().load(verbose=False)
    model = joblib.load("./saved_models/Balanced_XGBoost.pkl")
    scaler = joblib.load("./saved_models/scaler.pkl")
    X_orig = scaler.inverse_transform(data.X_test)
    y = data.y_test
    X = data.X_test

    rows = []
    for beta1 in [1.0, 1.5, 2.0, 2.5]:
        mask, params = simulate_mar_missing(
            y, X_orig, feature_idx=0, beta1=beta1, base_rate=0.3, seed=42
        )
        proba = model.predict_proba(X)
        y_imp = y.copy()
        y_imp[mask] = np.argmax(proba[mask], axis=1)
        acc = accuracy_score(y[mask], y_imp[mask])
        f1 = f1_score(y[mask], y_imp[mask], average="macro")
        rows.append({
            "beta1": beta1,
            "beta0": round(params["beta0"], 4),
            "realized_rate": round(params["realized_rate"], 4),
            "accuracy": round(acc, 4),
            "macro_f1": round(f1, 4),
        })
        print(f"beta1={beta1:.1f} beta0={params['beta0']:.3f} "
              f"rate={params['realized_rate']:.3f} acc={acc:.4f}")

    out = pd.DataFrame(rows)
    os.makedirs("./results", exist_ok=True)
    path = "./results/mar_beta_sensitivity.csv"
    out.to_csv(path, index=False)
    print(f"Wrote {path}")


if __name__ == "__main__":
    main()
