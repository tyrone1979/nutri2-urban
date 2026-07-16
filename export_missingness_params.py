#!/usr/bin/env python3
"""Export explicit missingness-mechanism parameters used in simulation."""
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.special import expit, logit

from model import DataPipeline

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "results" / "missingness_mechanism_params.csv"


def fit_mar_logit(fat, base_rate=0.3, beta1=2.0, seed=42):
    """
    MAR: logit(P(R=1 | FatER)) = β0 + β1 * z(FatER),
    where z is standardised FatER and β0 is calibrated so E[P] ≈ base_rate.
    """
    rng = np.random.default_rng(seed)
    z = (fat - fat.mean()) / fat.std(ddof=0)
    # Calibrate intercept for target mean missingness
    beta0 = float(logit(base_rate))
    for _ in range(40):
        p = expit(beta0 + beta1 * z)
        mean_p = float(p.mean())
        if abs(mean_p - base_rate) < 1e-4:
            break
        # Newton-style adjustment on intercept
        beta0 += float(logit(base_rate) - logit(np.clip(mean_p, 1e-4, 1 - 1e-4)))
    p = expit(beta0 + beta1 * z)
    mask = rng.random(len(fat)) < p
    return {
        "mechanism": "MAR",
        "formula": "logit(P(R=1|FatER))=beta0+beta1*z(FatER)",
        "beta0": beta0,
        "beta1": beta1,
        "base_rate_target": base_rate,
        "realized_rate": float(mask.mean()),
        "fat_mean": float(fat.mean()),
        "fat_sd": float(fat.std(ddof=0)),
        "note": "z(FatER)=(FatER-mean)/sd on the held-out test set",
    }, mask


def main():
    data = DataPipeline().load(verbose=False)
    scaler = joblib.load("./saved_models/scaler.pkl")
    X_orig = scaler.inverse_transform(data.X_test)
    fat = X_orig[:, 0]

    rows = []
    # MCAR
    rng = np.random.default_rng(42)
    mcar_rate = 0.3
    mask_mcar = rng.random(len(fat)) < mcar_rate
    rows.append({
        "mechanism": "MCAR",
        "formula": "P(R=1)=pi (independent of covariates and label)",
        "beta0": np.nan,
        "beta1": np.nan,
        "base_rate_target": mcar_rate,
        "realized_rate": float(mask_mcar.mean()),
        "fat_mean": float(fat.mean()),
        "fat_sd": float(fat.std(ddof=0)),
        "note": "Simple random masking on held-out test labels",
    })

    mar, _ = fit_mar_logit(fat, base_rate=0.3, beta1=2.0, seed=42)
    rows.append(mar)

    # Spatial: province-cluster differential rates (not whole-province deletion)
    high = {11, 31, 55}
    high_rate, low_rate = 0.5, 0.2
    prov = data.province_test
    p_sp = np.where(np.isin(prov, list(high)), high_rate, low_rate)
    mask_sp = rng.random(len(prov)) < p_sp
    rows.append({
        "mechanism": "Spatial",
        "formula": "P(R=1|Province)=0.50 if Beijing/Shanghai/Chongqing else 0.20",
        "beta0": np.nan,
        "beta1": np.nan,
        "base_rate_target": float(p_sp.mean()),
        "realized_rate": float(mask_sp.mean()),
        "fat_mean": float(fat.mean()),
        "fat_sd": float(fat.std(ddof=0)),
        "note": "Observation-level masking with province-specific rates (not listwise province deletion)",
    })

    df = pd.DataFrame(rows)
    OUT.parent.mkdir(exist_ok=True)
    df.to_csv(OUT, index=False)
    print(df.to_string(index=False))
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
