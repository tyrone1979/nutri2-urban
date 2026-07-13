#!/usr/bin/env python3
"""Binary T2 Rural vs Urban (exclude transitional stratum) under 30% masked labels."""
import os

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, cohen_kappa_score, f1_score, precision_score
from xgboost import XGBClassifier

from model import DataPipeline


def run_binary_analysis(missing_rate=0.3, random_state=42):
    data = DataPipeline().load(verbose=False)
    nt = data.y_test != 1
    X = data.X_test[nt]
    y = (data.y_test[nt] == 2).astype(int)

    nt_tr = data.y_train != 1
    bin_clf = XGBClassifier(
        n_estimators=600, max_depth=4, learning_rate=0.05,
        subsample=0.9, colsample_bytree=0.9, min_child_weight=5,
        gamma=0.2, reg_alpha=0.5, reg_lambda=1.2,
        objective="binary:logistic", random_state=42, n_jobs=1,
    )
    bin_clf.fit(data.X_train[nt_tr], (data.y_train[nt_tr] == 2).astype(int))

    holdout_pred = bin_clf.predict(X)
    holdout_acc = accuracy_score(y, holdout_pred)
    holdout_f1 = f1_score(y, holdout_pred)

    np.random.seed(random_state)
    miss = np.random.choice([True, False], size=len(y), p=[missing_rate, 1 - missing_rate])
    proba = bin_clf.predict_proba(X)
    y_imp = y.copy()
    y_imp[miss] = (proba[miss, 1] >= 0.5).astype(int)

    acc = accuracy_score(y[miss], y_imp[miss])
    macro_f1 = f1_score(y[miss], y_imp[miss])
    weighted_f1 = f1_score(y[miss], y_imp[miss], average="weighted")
    kappa = cohen_kappa_score(y[miss], y_imp[miss])
    urban_rec = ((y_imp[miss] == 1) & (y[miss] == 1)).sum() / max(1, (y[miss] == 1).sum())
    rural_rec = ((y_imp[miss] == 0) & (y[miss] == 0)).sum() / max(1, (y[miss] == 0).sum())
    rural_precision = precision_score(y[miss], y_imp[miss], pos_label=0, zero_division=0)
    urban_precision = precision_score(y[miss], y_imp[miss], pos_label=1, zero_division=0)

    row = {
        "scenario": "binary_nontransitional",
        "missing_rate": missing_rate,
        "n_test": int(len(y)),
        "n_masked": int(miss.sum()),
        "accuracy_masked": round(acc, 4),
        "macro_f1_masked": round(macro_f1, 4),
        "weighted_f1_masked": round(weighted_f1, 4),
        "kappa_masked": round(kappa, 4),
        "urban_recall_masked": round(urban_rec, 4),
        "rural_recall_masked": round(rural_rec, 4),
        "rural_precision_masked": round(rural_precision, 4),
        "urban_precision_masked": round(urban_precision, 4),
        "holdout_accuracy": round(holdout_acc, 4),
        "holdout_f1": round(holdout_f1, 4),
    }
    os.makedirs("./results", exist_ok=True)
    pd.DataFrame([row]).to_csv("./results/binary_classification.csv", index=False)
    print(pd.DataFrame([row]).T)
    return row


if __name__ == "__main__":
    run_binary_analysis()
