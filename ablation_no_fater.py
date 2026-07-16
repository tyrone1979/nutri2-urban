#!/usr/bin/env python3
"""Ablation: exclude FatER from predictors under Plan B T2-based labels."""
import os

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, cohen_kappa_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from data_pipeline import engineer_columns
from features import FEATURE_SETS


def load_df():
    import pyreadstat

    df, _ = pyreadstat.read_sas7bdat("./data/c12diet.sas7bdat")
    df.columns = df.columns.str.upper()
    df = df[["T2", "T1", "WAVE", "D3KCAL", "D3CARBO", "D3FAT", "D3PROTN"]].dropna()
    df = df[(df.D3KCAL > 500) & (df.D3KCAL < 5000)]
    return engineer_columns(df)


def train_eval(df, cols, name):
    X = df[cols].values
    y = df["label"].values
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y,
    )
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    model = XGBClassifier(
        n_estimators=600, max_depth=4, learning_rate=0.05,
        subsample=0.9, colsample_bytree=0.9, min_child_weight=5,
        gamma=0.2, reg_alpha=0.5, reg_lambda=1.2,
        objective="multi:softprob", num_class=3, random_state=42, n_jobs=1,
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    return {
        "Feature_Set": name,
        "n_features": len(cols),
        "Accuracy": round(accuracy_score(y_test, y_pred), 4),
        "Macro_F1": round(f1_score(y_test, y_pred, average="macro"), 4),
        "Kappa": round(cohen_kappa_score(y_test, y_pred), 4),
    }


def main():
    os.makedirs("./results", exist_ok=True)
    df = load_df()
    rows = []
    for name in ("full", "no_fater", "nutrients_only", "spatiotemporal_only"):
        spec = FEATURE_SETS[name]
        row = train_eval(df, spec["cols"], name)
        rows.append(row)
        print(row)

    out = pd.DataFrame(rows)
    out.to_csv("./results/ablation_no_fater.csv", index=False)
    # Also refresh feature_ablation.csv for docx updater compatibility
    out.to_csv("./results/feature_ablation.csv", index=False)
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()
