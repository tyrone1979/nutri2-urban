#!/usr/bin/env python3
"""
Downstream effect preservation under covariate-adjusted linear models.

Realistic epidemiology scenario (within available CHNS diet extract):
  Outcome: macronutrient feature (e.g. FatER)
  Exposure: Urban vs Rural (binary; Transitional excluded)
  Covariates: Year + Province (age/sex unavailable in this analytic extract)

Compare Urban coefficients and SEs using true vs imputed labels under 30% MCAR.
"""
import os

import joblib
import numpy as np
import pandas as pd
import statsmodels.api as sm

from model import DataPipeline, NUTRIENT_ALL_NAMES


def fit_urban_model(y, urban, year, province):
    """OLS: y ~ Urban + Year + C(Province)."""
    df = pd.DataFrame({
        "y": y,
        "urban": urban.astype(float),
        "year": year.astype(float),
        "province": province.astype(int),
    })
    # Drop rare provinces if needed for dummy rank
    dummies = pd.get_dummies(df["province"], prefix="prov", drop_first=True)
    X = pd.concat([df[["urban", "year"]], dummies], axis=1)
    X = sm.add_constant(X, has_constant="add")
    # Ensure float
    X = X.astype(float)
    model = sm.OLS(df["y"].astype(float), X).fit()
    return {
        "coef_urban": float(model.params["urban"]),
        "se_urban": float(model.bse["urban"]),
        "pval_urban": float(model.pvalues["urban"]),
        "n": int(model.nobs),
        "r2": float(model.rsquared),
    }


def run():
    print("=" * 70)
    print("Covariate-adjusted downstream regression (Year + Province)")
    print("=" * 70)

    data = DataPipeline().load(verbose=False)
    model = joblib.load("./saved_models/Balanced_XGBoost.pkl")

    y_true = data.y_test
    # Binary non-transitional stratum
    keep = y_true != 1
    y_bin = (y_true[keep] == 2).astype(int)  # Urban=1, Rural=0
    X_nut = data.nutrients_test[keep]
    year = data.year_test[keep]
    prov = data.province_test[keep]
    X_scaled = data.X_test[keep]

    np.random.seed(42)
    mask = np.random.choice([True, False], size=len(y_bin), p=[0.3, 0.7])
    y_imp = y_bin.copy()
    # Use three-class model probs on non-transitional rows; map to Urban vs Rural
    proba = model.predict_proba(X_scaled)
    # Urban probability among Rural/Urban: P(Urban) / (P(Rural)+P(Urban))
    p_u = proba[:, 2] / np.clip(proba[:, 0] + proba[:, 2], 1e-12, None)
    y_imp[mask] = (p_u[mask] >= 0.5).astype(int)

    rows = []
    for j, name in enumerate(NUTRIENT_ALL_NAMES):
        outcome = X_nut[:, j]
        true_fit = fit_urban_model(outcome, y_bin, year, prov)
        imp_fit = fit_urban_model(outcome, y_imp, year, prov)
        rows.append({
            "outcome": name,
            "true_coef": true_fit["coef_urban"],
            "true_se": true_fit["se_urban"],
            "true_pval": true_fit["pval_urban"],
            "imp_coef": imp_fit["coef_urban"],
            "imp_se": imp_fit["se_urban"],
            "imp_pval": imp_fit["pval_urban"],
            "coef_rel_change_pct": (
                100.0 * abs(imp_fit["coef_urban"] - true_fit["coef_urban"])
                / abs(true_fit["coef_urban"])
                if true_fit["coef_urban"] != 0 else np.nan
            ),
            "se_rel_change_pct": (
                100.0 * abs(imp_fit["se_urban"] - true_fit["se_urban"])
                / abs(true_fit["se_urban"])
                if true_fit["se_urban"] != 0 else np.nan
            ),
            "sign_consistent": int(np.sign(true_fit["coef_urban"]) == np.sign(imp_fit["coef_urban"])),
            "sig_consistent": int(
                (true_fit["pval_urban"] < 0.05) == (imp_fit["pval_urban"] < 0.05)
            ),
            "n": true_fit["n"],
            "covariates": "Year + Province dummies",
        })
        print(
            f"{name:10s} true={true_fit['coef_urban']:+.4f}±{true_fit['se_urban']:.4f} "
            f"imp={imp_fit['coef_urban']:+.4f}±{imp_fit['se_urban']:.4f} "
            f"Δcoef%={rows[-1]['coef_rel_change_pct']:.1f}"
        )

    os.makedirs("./results", exist_ok=True)
    out = "./results/downstream_adjusted_regression.csv"
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"Wrote {out}")


if __name__ == "__main__":
    run()
