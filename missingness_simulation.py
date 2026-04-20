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
from scipy.special import expit
from scipy.spatial.distance import jensenshannon
from collections import Counter
import warnings

warnings.filterwarnings('ignore')

from model import DataPipeline, PROVINCE_SHORT


def sigmoid(x):
    return expit(x)


def simulate_mar_missing(y_test, X_original, feature_idx=0, alpha=5.0, base_rate=0.3):
    """
    MAR: 高 FatER 样本更容易缺失
    """
    fat_vals = X_original[:, feature_idx]
    fat_centered = fat_vals - np.mean(fat_vals)
    prob = sigmoid(alpha * fat_centered)
    prob = prob * (base_rate / np.mean(prob))
    prob = np.clip(prob, 0.01, 0.99)
    mask = np.random.random(len(y_test)) < prob
    return mask


def simulate_spatial_missing(y_test, province_codes, high_missing_provs=[11, 31, 55],
                             high_rate=0.5, low_rate=0.2):
    """
    空间相关缺失：直辖市缺失率更高
    11=Beijing, 31=Shanghai, 55=Chongqing
    """
    mask = np.zeros(len(y_test), dtype=bool)
    for i, prov in enumerate(province_codes):
        if prov in high_missing_provs:
            mask[i] = np.random.random() < high_rate
        else:
            mask[i] = np.random.random() < low_rate
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

    # Scenario 2: MAR (high FatER → more missing)
    print("\n[4/4] 场景 2: MAR (高脂肪供能比 → 更容易缺失)")
    np.random.seed(42)
    mask_mar = simulate_mar_missing(y_test, X_original, feature_idx=0, alpha=5.0, base_rate=0.3)
    results_mar = evaluate_missingness_scenario(
        X_test, y_test, province_test, model, mask_mar, "MAR (FatER)"
    )
    all_results.extend(results_mar)
    for r in results_mar:
        if r['Method'] == 'Proposed':
            print(f"   Proposed: Acc={r['Accuracy']:.4f}, F1={r['Macro_F1']:.4f}, JS={r['JS_Div']:.6f}")

    # Scenario 3: Spatial missing
    print("\n[5/5] 场景 3: Spatial Missing (直辖市缺失率 50% vs 其他 20%)")
    np.random.seed(42)
    mask_spatial = simulate_spatial_missing(
        y_test, province_test,
        high_missing_provs=[11, 31, 55],  # Beijing, Shanghai, Chongqing
        high_rate=0.5, low_rate=0.2
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

    return df_all


if __name__ == "__main__":
    import os

    df_all = run_missingness_simulation()