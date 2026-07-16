#!/usr/bin/env python3
"""
Concern 1 sensitivity: binary performance when Transitional is NOT excluded.

Collapses three-class labels to binary by assigning Transitional either by:
  (A) true administrative T2, or
  (B) random 50/50 Rural/Urban,
then evaluates masked-label accuracy on the full test set (no exclusion of the
FatER-overlap stratum). Compares to the primary non-transitional binary task.
"""
import os

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score
from xgboost import XGBClassifier

from data_pipeline import engineer_columns
from features import FEATURE_COLS
from model import DataPipeline


def load_raw():
    import pyreadstat

    df, _ = pyreadstat.read_sas7bdat("./data/c12diet.sas7bdat")
    df.columns = df.columns.str.upper()
    df = df[["T2", "T1", "WAVE", "D3KCAL", "D3CARBO", "D3FAT", "D3PROTN"]].dropna()
    df = df[(df.D3KCAL > 500) & (df.D3KCAL < 5000)]
    return engineer_columns(df)


def eval_binary(X_train, y_train, X_test, y_test, missing_rate=0.3, seed=42):
    clf = XGBClassifier(
        n_estimators=600, max_depth=4, learning_rate=0.05,
        subsample=0.9, colsample_bytree=0.9, min_child_weight=5,
        gamma=0.2, reg_alpha=0.5, reg_lambda=1.2,
        objective="binary:logistic", random_state=42, n_jobs=1,
    )
    clf.fit(X_train, y_train)
    holdout_acc = accuracy_score(y_test, clf.predict(X_test))
    holdout_f1 = f1_score(y_test, clf.predict(X_test))

    rng = np.random.default_rng(seed)
    miss = rng.random(len(y_test)) < missing_rate
    proba = clf.predict_proba(X_test)[:, 1]
    y_imp = y_test.copy()
    y_imp[miss] = (proba[miss] >= 0.5).astype(int)
    return {
        "n_test": int(len(y_test)),
        "n_masked": int(miss.sum()),
        "holdout_accuracy": round(holdout_acc, 4),
        "holdout_f1": round(holdout_f1, 4),
        "masked_accuracy": round(accuracy_score(y_test[miss], y_imp[miss]), 4),
        "masked_f1": round(f1_score(y_test[miss], y_imp[miss]), 4),
        "urban_share": round(float(y_test.mean()), 4),
    }


def main():
    data = DataPipeline().load(verbose=False)
    df = load_raw()
    # Reconstruct train/test indices via same split as pipeline
    from sklearn.model_selection import train_test_split

    idx = np.arange(len(df))
    train_idx, test_idx = train_test_split(
        idx, test_size=0.2, random_state=42, stratify=df["label"].values,
    )
    df_tr, df_te = df.iloc[train_idx].copy(), df.iloc[test_idx].copy()

    # Use scaled features from pipeline for consistency
    X_tr, X_te = data.X_train, data.X_test
    y3_tr, y3_te = data.y_train, data.y_test

    rows = []

    # Primary: exclude transitional
    nt_tr = y3_tr != 1
    nt_te = y3_te != 1
    yb_tr = (y3_tr[nt_tr] == 2).astype(int)
    yb_te = (y3_te[nt_te] == 2).astype(int)
    r = eval_binary(X_tr[nt_tr], yb_tr, X_te[nt_te], yb_te)
    r["scenario"] = "primary_exclude_transitional"
    rows.append(r)

    # (A) Assign transitional by true T2
    t2_tr = df_tr["T2"].astype(int).values
    t2_te = df_te["T2"].astype(int).values
    # T2: 1=urban, 2=rural -> binary urban=1
    y_t2_tr = (t2_tr == 1).astype(int)
    y_t2_te = (t2_te == 1).astype(int)
    r = eval_binary(X_tr, y_t2_tr, X_te, y_t2_te)
    r["scenario"] = "collapse_transitional_by_T2"
    rows.append(r)

    # (B) Random 50/50 for transitional only; keep non-transitional as T2/binary
    rng = np.random.default_rng(42)
    y_rand_tr = (y3_tr == 2).astype(int)
    y_rand_te = (y3_te == 2).astype(int)
    # For transitional, overwrite with random
    y_rand_tr[y3_tr == 1] = rng.integers(0, 2, size=int((y3_tr == 1).sum()))
    y_rand_te[y3_te == 1] = rng.integers(0, 2, size=int((y3_te == 1).sum()))
    # Non-transitional already set from ==2; Rural (0) stays 0. Good.
    # Wait: for Rural y3==0, y_rand=0; Urban y3==2, y_rand=1; Trans random. OK.
    r = eval_binary(X_tr, y_rand_tr, X_te, y_rand_te)
    r["scenario"] = "collapse_transitional_random50"
    rows.append(r)

    out = pd.DataFrame(rows)
    os.makedirs("./results", exist_ok=True)
    path = "./results/binary_noise_injection_sensitivity.csv"
    out.to_csv(path, index=False)
    print(out.to_string(index=False))
    print(f"Wrote {path}")


if __name__ == "__main__":
    main()
