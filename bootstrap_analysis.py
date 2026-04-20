#!/usr/bin/env python3
"""
Bootstrap 不确定性分析
计算填补性能、效应量偏差的 95% 置信区间
"""
import numpy as np
import pandas as pd
import joblib
from sklearn.metrics import accuracy_score, f1_score, cohen_kappa_score
from sklearn.neighbors import KNeighborsClassifier
from collections import Counter
from scipy.spatial.distance import jensenshannon
import warnings

warnings.filterwarnings('ignore')

from model import DataPipeline


def compute_fidelity(y_true, y_imp, n_classes=3):
    """计算分布保真度"""
    true_counts = np.bincount(y_true, minlength=n_classes)
    imp_counts = np.bincount(y_imp, minlength=n_classes)
    true_dist = true_counts / len(y_true)
    imp_dist = imp_counts / len(y_imp)
    js_div = jensenshannon(true_dist, imp_dist)
    max_diff = np.max(np.abs(imp_dist - true_dist))
    return js_div, max_diff


def compute_effect_size(X_original, y_labels, feature_idx=0):
    """计算城乡效应量（Urban vs Rural）"""
    urban = X_original[y_labels == 2, feature_idx]
    rural = X_original[y_labels == 0, feature_idx]
    if len(urban) == 0 or len(rural) == 0:
        return np.nan
    return urban.mean() - rural.mean()


def bootstrap_imputation_evaluation(data, model, scaler, n_boot=1000, missing_rate=0.3, random_seed=42):
    """
    Bootstrap 评估填补性能
    """
    np.random.seed(random_seed)

    X_test = data.X_test
    y_test = data.y_test
    n_samples = len(y_test)

    # 还原原始特征
    X_original = scaler.inverse_transform(X_test)

    results = []

    print(f"\n🔄 Bootstrap 抽样 (n={n_boot})...")

    for i in range(n_boot):
        if (i + 1) % 200 == 0:
            print(f"   进度: {i + 1}/{n_boot}")

        # Bootstrap 重抽样
        indices = np.random.choice(n_samples, size=n_samples, replace=True)
        X_boot = X_test[indices]
        y_boot = y_test[indices]
        X_orig_boot = X_original[indices]

        # 模拟缺失
        mask = np.random.choice([True, False], size=n_samples, p=[missing_rate, 1 - missing_rate])
        y_masked = y_boot.copy()
        y_masked[mask] = -1

        # 多数类填补
        majority_class = Counter(y_boot[~mask]).most_common(1)[0][0]
        y_imp_majority = y_masked.copy()
        y_imp_majority[mask] = majority_class

        # KNN 填补
        knn = KNeighborsClassifier(n_neighbors=5)
        knn.fit(X_boot[~mask], y_boot[~mask])
        y_imp_knn = y_masked.copy()
        y_imp_knn[mask] = knn.predict(X_boot[mask])

        # Proposed 填补
        y_proba = model.predict_proba(X_boot)
        y_imp_proposed = y_masked.copy()
        y_imp_proposed[mask] = np.argmax(y_proba[mask], axis=1)

        # 评估指标（仅在缺失样本上）
        mask_eval = mask

        # Proposed
        acc_p = accuracy_score(y_boot[mask_eval], y_imp_proposed[mask_eval])
        f1_p = f1_score(y_boot[mask_eval], y_imp_proposed[mask_eval], average='macro')
        kappa_p = cohen_kappa_score(y_boot[mask_eval], y_imp_proposed[mask_eval])
        js_p, maxdiff_p = compute_fidelity(y_boot, y_imp_proposed)

        # KNN
        acc_k = accuracy_score(y_boot[mask_eval], y_imp_knn[mask_eval])
        f1_k = f1_score(y_boot[mask_eval], y_imp_knn[mask_eval], average='macro')

        # 效应量偏差（FatER）
        true_eff = compute_effect_size(X_orig_boot, y_boot, feature_idx=0)
        imp_eff = compute_effect_size(X_orig_boot, y_imp_proposed, feature_idx=0)
        bias = imp_eff - true_eff
        bias_pct = abs(bias) / abs(true_eff) * 100 if true_eff != 0 else 0

        results.append({
            'proposed_acc': acc_p,
            'proposed_f1': f1_p,
            'proposed_kappa': kappa_p,
            'proposed_js': js_p,
            'proposed_maxdiff': maxdiff_p,
            'knn_acc': acc_k,
            'knn_f1': f1_k,
            'true_effect': true_eff,
            'imp_effect': imp_eff,
            'bias': bias,
            'bias_pct': bias_pct
        })

    df_results = pd.DataFrame(results)

    # 计算百分位数
    metrics = ['proposed_acc', 'proposed_f1', 'proposed_kappa', 'proposed_js',
               'proposed_maxdiff', 'knn_acc', 'knn_f1', 'true_effect', 'imp_effect', 'bias', 'bias_pct']

    summary = {}
    for m in metrics:
        summary[m] = {
            'median': np.percentile(df_results[m], 50),
            'ci_lower': np.percentile(df_results[m], 2.5),
            'ci_upper': np.percentile(df_results[m], 97.5)
        }

    return df_results, summary


def run_bootstrap_analysis(n_boot=1000):
    print("=" * 70)
    print("🧪 Bootstrap 不确定性分析")
    print("=" * 70)

    # 加载数据
    print("\n[1/3] 加载数据...")
    data = DataPipeline().load()
    print(f"   ✅ 测试集: {len(data.y_test)} 样本")

    print("\n[2/3] 加载模型和标准化器...")
    model = joblib.load("./saved_models/Balanced_XGBoost.pkl")
    scaler = joblib.load("./saved_models/scaler.pkl")
    print("   ✅ 加载完成")

    print("\n[3/3] 运行 Bootstrap...")
    df_results, summary = bootstrap_imputation_evaluation(
        data, model, scaler, n_boot=n_boot, missing_rate=0.3
    )

    # 打印结果
    print("\n" + "=" * 70)
    print("📊 Bootstrap 结果 (95% CI)")
    print("=" * 70)

    print("\n【填补性能 - Proposed BXGB】")
    print(f"  Accuracy:   {summary['proposed_acc']['median']:.4f} "
          f"[{summary['proposed_acc']['ci_lower']:.4f}, {summary['proposed_acc']['ci_upper']:.4f}]")
    print(f"  Macro-F1:   {summary['proposed_f1']['median']:.4f} "
          f"[{summary['proposed_f1']['ci_lower']:.4f}, {summary['proposed_f1']['ci_upper']:.4f}]")
    print(f"  Kappa:      {summary['proposed_kappa']['median']:.4f} "
          f"[{summary['proposed_kappa']['ci_lower']:.4f}, {summary['proposed_kappa']['ci_upper']:.4f}]")
    print(f"  JS Div:     {summary['proposed_js']['median']:.6f} "
          f"[{summary['proposed_js']['ci_lower']:.6f}, {summary['proposed_js']['ci_upper']:.6f}]")
    print(f"  Max Diff:   {summary['proposed_maxdiff']['median']:.4f} "
          f"[{summary['proposed_maxdiff']['ci_lower']:.4f}, {summary['proposed_maxdiff']['ci_upper']:.4f}]")

    print("\n【填补性能 - KNN】")
    print(f"  Accuracy:   {summary['knn_acc']['median']:.4f} "
          f"[{summary['knn_acc']['ci_lower']:.4f}, {summary['knn_acc']['ci_upper']:.4f}]")
    print(f"  Macro-F1:   {summary['knn_f1']['median']:.4f} "
          f"[{summary['knn_f1']['ci_lower']:.4f}, {summary['knn_f1']['ci_upper']:.4f}]")

    print("\n【效应量偏差 - FatER】")
    print(f"  True Effect: {summary['true_effect']['median']:.4f} "
          f"[{summary['true_effect']['ci_lower']:.4f}, {summary['true_effect']['ci_upper']:.4f}]")
    print(f"  Imp Effect:  {summary['imp_effect']['median']:.4f} "
          f"[{summary['imp_effect']['ci_lower']:.4f}, {summary['imp_effect']['ci_upper']:.4f}]")
    print(f"  Bias:        {summary['bias']['median']:.4f} "
          f"[{summary['bias']['ci_lower']:.4f}, {summary['bias']['ci_upper']:.4f}]")
    print(f"  Bias %:      {summary['bias_pct']['median']:.1f}% "
          f"[{summary['bias_pct']['ci_lower']:.1f}%, {summary['bias_pct']['ci_upper']:.1f}%]")

    # 保存
    os.makedirs("./results", exist_ok=True)
    df_results.to_csv("./results/bootstrap_results.csv", index=False)

    summary_df = pd.DataFrame([
        {'Metric': k, 'Median': v['median'], 'CI_Lower': v['ci_lower'], 'CI_Upper': v['ci_upper']}
        for k, v in summary.items()
    ])
    summary_df.to_csv("./results/bootstrap_summary.csv", index=False)

    print("\n✅ 结果已保存至: results/bootstrap_results.csv, results/bootstrap_summary.csv")

    return df_results, summary


if __name__ == "__main__":
    import os

    df_results, summary = run_bootstrap_analysis(n_boot=1000)