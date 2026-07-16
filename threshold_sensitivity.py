#!/usr/bin/env python3
"""Threshold sensitivity for Plan B: vary transitional FatER band on T2 base labels."""
import os

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, cohen_kappa_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from data_pipeline import engineer_columns
from features import FEATURE_COLS
from labels import assign_t2_three_class_labels

THRESHOLDS = [(0.21, 0.28), (0.23, 0.30), (0.25, 0.32), (0.27, 0.34)]


def load_df():
    import pyreadstat

    df, _ = pyreadstat.read_sas7bdat("./data/c12diet.sas7bdat")
    df.columns = df.columns.str.upper()
    df = df[["T2", "T1", "WAVE", "D3KCAL", "D3CARBO", "D3FAT", "D3PROTN"]].dropna()
    df = df[(df.D3KCAL > 500) & (df.D3KCAL < 5000)]
    return engineer_columns(df)


def main():
    os.makedirs("./results", exist_ok=True)
    model = joblib.load("./saved_models/Balanced_XGBoost.pkl")
    df = load_df()
    df["label_default"] = assign_t2_three_class_labels(df["T2"].astype(int), df["fat_pct"])
    train_idx, test_idx = train_test_split(
        np.arange(len(df)), test_size=0.2, random_state=42, stratify=df["label_default"],
    )
    df_test = df.iloc[test_idx].copy()
    scaler = StandardScaler()
    scaler.fit(df.iloc[train_idx][FEATURE_COLS].values)
    X_test = scaler.transform(df_test[FEATURE_COLS].values)

    rows = []
    for low, high in THRESHOLDS:
        y_true = assign_t2_three_class_labels(
            df_test["T2"].astype(int).values,
            df_test["fat_pct"].values,
            trans_low=low,
            trans_high=high,
        )
        y_pred = model.predict(X_test)
        rows.append({
            "trans_low": low,
            "trans_high": high,
            "accuracy": round(accuracy_score(y_true, y_pred), 4),
            "macro_f1": round(f1_score(y_true, y_pred, average="macro"), 4),
            "kappa": round(cohen_kappa_score(y_true, y_pred), 4),
            "transitional_prev": round(float((y_true == 1).mean()), 4),
        })
        print(f"  [{low:.2f}, {high:.2f}] acc={rows[-1]['accuracy']:.4f}")

    pd.DataFrame(rows).to_csv("./results/threshold_sensitivity.csv", index=False)
    print(pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":
    main()
