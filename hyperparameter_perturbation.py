#!/usr/bin/env python3
"""Hyperparameter perturbation (±20%) for Balanced XGBoost key knobs."""
import os

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score
from xgboost import XGBClassifier

from model import DataPipeline

BASE = dict(
    n_estimators=600, max_depth=4, learning_rate=0.05,
    subsample=0.9, colsample_bytree=0.9, min_child_weight=5,
    gamma=0.2, reg_alpha=0.5, reg_lambda=1.2,
    objective="multi:softprob", num_class=3, random_state=42, n_jobs=1,
)


def eval_params(X_tr, y_tr, X_te, y_te, params, missing_rate=0.3, seed=42):
    clf = XGBClassifier(**params)
    clf.fit(X_tr, y_tr)
    rng = np.random.default_rng(seed)
    miss = rng.random(len(y_te)) < missing_rate
    pred = np.argmax(clf.predict_proba(X_te)[miss], axis=1)
    return accuracy_score(y_te[miss], pred), f1_score(y_te[miss], pred, average="macro")


def main():
    data = DataPipeline().load(verbose=False)
    X_tr, y_tr = data.X_train, data.y_train
    X_te, y_te = data.X_test, data.y_test

    rows = []
    base_acc, base_f1 = eval_params(X_tr, y_tr, X_te, y_te, BASE)
    rows.append({
        "setting": "baseline", "max_depth": 4, "learning_rate": 0.05,
        "subsample": 0.9, "accuracy": round(base_acc, 4), "macro_f1": round(base_f1, 4),
        "delta_acc_pp": 0.0,
    })
    print(f"baseline acc={base_acc:.4f}")

    # Perturb each key param ±20% (depth as integer ±1 ≈ 25%)
    grid = []
    for depth in [3, 4, 5]:
        for lr in [0.04, 0.05, 0.06]:
            for sub in [0.72, 0.9, 1.0]:
                grid.append((depth, lr, min(sub, 1.0)))

    # Deduplicate and evaluate
    seen = set()
    accs = [base_acc]
    for depth, lr, sub in grid:
        key = (depth, round(lr, 3), round(sub, 3))
        if key in seen:
            continue
        seen.add(key)
        p = dict(BASE)
        p["max_depth"] = depth
        p["learning_rate"] = lr
        p["subsample"] = sub
        acc, f1 = eval_params(X_tr, y_tr, X_te, y_te, p)
        accs.append(acc)
        rows.append({
            "setting": f"depth{depth}_lr{lr}_sub{sub}",
            "max_depth": depth, "learning_rate": lr, "subsample": sub,
            "accuracy": round(acc, 4), "macro_f1": round(f1, 4),
            "delta_acc_pp": round(100 * (acc - base_acc), 2),
        })
        print(f"  depth={depth} lr={lr} sub={sub:.2f} acc={acc:.4f}")

    summary = {
        "n_settings": len(rows),
        "acc_mean": round(float(np.mean(accs)), 4),
        "acc_sd": round(float(np.std(accs, ddof=1)), 4),
        "acc_min": round(float(np.min(accs)), 4),
        "acc_max": round(float(np.max(accs)), 4),
        "max_abs_delta_pp": round(100 * float(np.max(np.abs(np.array(accs) - base_acc))), 2),
    }
    os.makedirs("./results", exist_ok=True)
    pd.DataFrame(rows).to_csv("./results/hyperparameter_perturbation.csv", index=False)
    pd.DataFrame([summary]).to_csv("./results/hyperparameter_perturbation_summary.csv", index=False)
    print(summary)
    print("Wrote results/hyperparameter_perturbation*.csv")


if __name__ == "__main__":
    main()
