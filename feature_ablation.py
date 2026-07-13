#!/usr/bin/env python3
"""Feature-set ablation: nutrients only vs spatiotemporal only vs full (no FatER)."""
import os

import pandas as pd
from sklearn.metrics import accuracy_score, cohen_kappa_score, f1_score
from xgboost import XGBClassifier

from data_pipeline import DataPipeline
from features import FEATURE_SETS

os.makedirs("./results", exist_ok=True)


def train_xgb(X_train, y_train):
    model = XGBClassifier(
        n_estimators=600,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        min_child_weight=5,
        gamma=0.2,
        reg_alpha=0.5,
        reg_lambda=1.2,
        eval_metric="mlogloss",
        objective="multi:softprob",
        num_class=3,
        random_state=42,
        n_jobs=1,
    )
    model.fit(X_train, y_train)
    return model


def eval_split(name, cols):
    data = DataPipeline(feature_cols=cols).load(verbose=False)
    model = train_xgb(data.X_train, data.y_train)
    pred = model.predict(data.X_test)
    return {
        "Feature_Set": name,
        "N_Features": len(cols),
        "Features": ", ".join(cols),
        "Accuracy": round(accuracy_score(data.y_test, pred), 4),
        "Macro_F1": round(f1_score(data.y_test, pred, average="macro"), 4),
        "Kappa": round(cohen_kappa_score(data.y_test, pred), 4),
    }


def main():
    print("=" * 70)
    print("Feature-set ablation (Plan B: T2 labels, 4 macros + Year + Province)")
    print("=" * 70)
    rows = []
    for key in ("nutrients_only", "spatiotemporal_only", "full"):
        spec = FEATURE_SETS[key]
        row = eval_split(key, spec["cols"])
        rows.append(row)
        print(f"  {key}: Acc={row['Accuracy']}, F1={row['Macro_F1']}, κ={row['Kappa']}")

    out = pd.DataFrame(rows)
    out.to_csv("./results/feature_ablation.csv", index=False)
    print("\nSaved: results/feature_ablation.csv")
    return out


if __name__ == "__main__":
    main()
