#!/usr/bin/env python3
"""
非随机缺失（MAR/MNAR）模拟
测试模型在结构化缺失下的稳健性
"""
import numpy as np
import pandas as pd
import joblib
from sklearn.metrics import accuracy_score, f1_score, cohen_kappa_score
from sklearn.neighbors import KNeighborsClassifier
from scipy.special import expit, logit
from scipy.spatial.distance import jensenshannon
from collections import Counter
import warnings
import os

warnings.filterwarnings('ignore')

from model import DataPipeline, PROVINCE_SHORT


def sigmoid(x):
    return expit(x)


def simulate_mar_missing(y_test, X_original, feature_idx=0, beta1=2.0, base_rate=0.3, seed=42):
    """
    MAR via logistic propensity:
        logit(P(R=1 | FatER)) = β0 + β1 * z(FatER),
    where z is standardised FatER and β0 is calibrated so E[P] ≈ base_rate.
    Returns (mask, params_dict).
    """
    rng = np.random.default_rng(seed)
    fat = X_original[:, feature_idx]
    z = (fat - fat.mean()) / fat.std(ddof=0)
    beta0 = float(logit(base_rate))
    for _ in range(40):
        p = expit(beta0 + beta1 * z)
        mean_p = float(p.mean())
        if abs(mean_p - base_rate) < 1e-4:
            break
        beta0 += float(logit(base_rate) - logit(np.clip(mean_p, 1e-4, 1 - 1e-4)))
    p = expit(beta0 + beta1 * z)
    mask = rng.random(len(y_test)) < p
    params = {
        "beta0": beta0,
        "beta1": beta1,
        "realized_rate": float(mask.mean()),
        "fat_mean": float(fat.mean()),
        "fat_sd": float(fat.std(ddof=0)),
    }
    return mask, params


def simulate_spatial_missing(y_test, province_codes, high_missing_provs=[11, 31, 55],
                             high_rate=0.5, low_rate=0.2, seed=42):
    """
    Spatially structured missingness (observation-level, province-specific rates):
    Beijing/Shanghai/Chongqing masked at high_rate; other provinces at low_rate.
    This is NOT whole-province deletion.
    """
    rng = np.random.default_rng(seed)
    mask = np.zeros(len(y_test), dtype=bool)
    for i, prov in enumerate(province_codes):
        rate = high_rate if prov in high_missing_provs else low_rate
        mask[i] = rng.random() < rate
    return mask


def compute_fidelity(y_true, y_imp):
    true_counts = np.bincount(y_true, minlength=3)
    imp_counts = np.bincount(y_imp, minlength=3)
    true_dist = true_counts / len(y_true)
    imp_dist = imp_counts / len(y_imp)
    js_div = jensenshannon(true_dist, imp_dist)
    max_diff = np.max(np.abs(imp_dist - true_dist))
    return js_div, max_diff


def evaluate_missingness_scenario(X_test, y_test, province_test, model, mask, scenario_name):
    """评估单个缺失场景"""
    y_masked = y_test.copy()
    y_masked[mask] = -1

    actual_rate = mask.mean()

    # 多数类
    majority_class = Counter(y_test[~mask]).most_common(1)[0][0]
    y_imp_majority = y_masked.copy()
    y_imp_majority[mask] = majority_class

    # KNN
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(X_test[~mask], y_test[~mask])
    y_imp_knn = y_masked.copy()
    y_imp_knn[mask] = knn.predict(X_test[mask])

    # Proposed
    y_proba = model.predict_proba(X_test)
    y_imp_proposed = y_masked.copy()
    y_imp_proposed[mask] = np.argmax(y_proba[mask], axis=1)

    # 评估
    results = []
    for name, y_imp in [('Proposed', y_imp_proposed), ('KNN', y_imp_knn), ('Majority', y_imp_majority)]:
        acc = accuracy_score(y_test[mask], y_imp[mask])
        f1 = f1_score(y_test[mask], y_imp[mask], average='macro')
        kappa = cohen_kappa_score(y_test[mask], y_imp[mask])
        js_div, max_diff = compute_fidelity(y_test, y_imp)

        results.append({
            'Scenario': scenario_name,
            'Method': name,
            'Actual_Rate': actual_rate,
            'Accuracy': acc,
            'Macro_F1': f1,
            'Kappa': kappa,
            'JS_Div': js_div,
            'Max_Diff': max_diff
        })

    return results


def run_missingness_simulation():
    print("=" * 70)
    print("🧪 非随机缺失（MAR / Spatial）稳健性测试")
    print("=" * 70)

    # 加载
    print("\n[1/3] 加载数据...")
    data = DataPipeline().load()
    X_test = data.X_test
    y_test = data.y_test
    province_test = data.province_test

    scaler = joblib.load("./saved_models/scaler.pkl")
    X_original = scaler.inverse_transform(X_test)

    print(f"   ✅ 测试集: {len(y_test)} 样本")

    print("\n[2/3] 加载模型...")
    model = joblib.load("./saved_models/Balanced_XGBoost.pkl")
    print("   ✅ 模型加载完成")

    all_results = []

    # Scenario 1: MCAR (baseline)
    print("\n[3/4] 场景 1: MCAR (完全随机缺失)")
    np.random.seed(42)
    mask_mcar = np.random.choice([True, False], size=len(y_test), p=[0.3, 0.7])
    results_mcar = evaluate_missingness_scenario(
        X_test, y_test, province_test, model, mask_mcar, "MCAR"
    )
    all_results.extend(results_mcar)
    for r in results_mcar:
        if r['Method'] == 'Proposed':
            print(f"   Proposed: Acc={r['Accuracy']:.4f}, F1={r['Macro_F1']:.4f}, JS={r['JS_Div']:.6f}")

    # Scenario 2: MAR (logistic in FatER)
    print("\n[4/4] 场景 2: MAR logit(P)=beta0+beta1*z(FatER)")
    mask_mar, mar_params = simulate_mar_missing(
        y_test, X_original, feature_idx=0, beta1=2.0, base_rate=0.3, seed=42
    )
    print(f"   MAR params: beta0={mar_params['beta0']:.4f}, beta1={mar_params['beta1']:.4f}, "
          f"realized={mar_params['realized_rate']:.4f}")
    results_mar = evaluate_missingness_scenario(
        X_test, y_test, province_test, model, mask_mar, "MAR (FatER)"
    )
    all_results.extend(results_mar)
    for r in results_mar:
        if r['Method'] == 'Proposed':
            print(f"   Proposed: Acc={r['Accuracy']:.4f}, F1={r['Macro_F1']:.4f}, JS={r['JS_Div']:.6f}")

    # Scenario 3: Spatial missing
    print("\n[5/5] 场景 3: Spatial Missing (直辖市缺失率 50% vs 其他 20%)")
    mask_spatial = simulate_spatial_missing(
        y_test, province_test,
        high_missing_provs=[11, 31, 55],  # Beijing, Shanghai, Chongqing
        high_rate=0.5, low_rate=0.2, seed=42
    )
    results_spatial = evaluate_missingness_scenario(
        X_test, y_test, province_test, model, mask_spatial, "Spatial"
    )
    all_results.extend(results_spatial)
    for r in results_spatial:
        if r['Method'] == 'Proposed':
            print(f"   Proposed: Acc={r['Accuracy']:.4f}, F1={r['Macro_F1']:.4f}, JS={r['JS_Div']:.6f}")

    # 汇总
    df_all = pd.DataFrame(all_results)

    print("\n" + "=" * 70)
    print("📊 非随机缺失稳健性汇总")
    print("=" * 70)
    print(df_all.to_string(index=False))

    # 保存
    os.makedirs("./results", exist_ok=True)
    df_all.to_csv("./results/missingness_simulation.csv", index=False)
    print("\n✅ 结果已保存至: results/missingness_simulation.csv")

    # Explicit mechanism parameters for manuscript Methods
    mech = pd.DataFrame([
        {
            "mechanism": "MCAR",
            "formula": "P(R=1)=0.30 (independent of covariates/labels)",
            "beta0": np.nan,
            "beta1": np.nan,
            "realized_rate": float(mask_mcar.mean()),
            "note": "Simple random masking on held-out test labels",
        },
        {
            "mechanism": "MAR",
            "formula": "logit(P(R=1|FatER))=beta0+beta1*z(FatER)",
            "beta0": mar_params["beta0"],
            "beta1": mar_params["beta1"],
            "realized_rate": mar_params["realized_rate"],
            "note": f"z=(FatER-{mar_params['fat_mean']:.4f})/{mar_params['fat_sd']:.4f}",
        },
        {
            "mechanism": "Spatial",
            "formula": "P(R=1|Province)=0.50 if BJ/SH/CQ else 0.20",
            "beta0": np.nan,
            "beta1": np.nan,
            "realized_rate": float(mask_spatial.mean()),
            "note": "Observation-level province-specific rates (not whole-province deletion)",
        },
    ])
    mech.to_csv("./results/missingness_mechanism_params.csv", index=False)
    print("Wrote results/missingness_mechanism_params.csv")

    return df_all


if __name__ == "__main__":
    df_all = run_missingness_simulation()