#!/usr/bin/env python3
"""
Paired Bootstrap Difference Test (优化版)
H0: Proposed BXGB 与 KNN 性能无差异
"""
import numpy as np
import pandas as pd
import joblib
from sklearn.metrics import accuracy_score, f1_score
from sklearn.neighbors import KNeighborsClassifier
import warnings

warnings.filterwarnings('ignore')

from model import DataPipeline


def paired_bootstrap_test_optimized(data, model, n_boot=2000, missing_rate=0.3, random_seed=42):
    """
    优化版 Paired bootstrap test
    - 减少 bootstrap 次数到 2000（足够计算 p-value）
    - 预先计算 Proposed 概率，避免重复推理
    - 使用向量化操作
    """
    np.random.seed(random_seed)

    X_test = data.X_test
    y_test = data.y_test
    n_samples = len(y_test)

    # 预先计算 Proposed 概率（最耗时的部分只做一次）
    print("   预计算模型概率...")
    y_proba_full = model.predict_proba(X_test)

    diff_acc = []
    diff_f1 = []

    print(f"\n🔄 Paired Bootstrap (n={n_boot})...")

    for i in range(n_boot):
        if (i + 1) % 500 == 0:
            print(f"   进度: {i + 1}/{n_boot}")

        # Bootstrap 重抽样
        indices = np.random.choice(n_samples, size=n_samples, replace=True)
        X_boot = X_test[indices]
        y_boot = y_test[indices]
        y_proba_boot = y_proba_full[indices]

        # 模拟缺失（固定比例，加快速度）
        n_missing = int(n_samples * missing_rate)
        mask = np.zeros(n_samples, dtype=bool)
        mask[:n_missing] = True
        np.random.shuffle(mask)

        y_masked = y_boot.copy()
        y_masked[mask] = -1

        # KNN 填补
        knn = KNeighborsClassifier(n_neighbors=5)
        knn.fit(X_boot[~mask], y_boot[~mask])
        y_imp_knn = y_masked.copy()
        y_imp_knn[mask] = knn.predict(X_boot[mask])

        # Proposed 填补（使用预计算概率）
        y_imp_proposed = y_masked.copy()
        y_imp_proposed[mask] = np.argmax(y_proba_boot[mask], axis=1)

        # 计算性能差
        acc_p = accuracy_score(y_boot[mask], y_imp_proposed[mask])
        acc_k = accuracy_score(y_boot[mask], y_imp_knn[mask])
        diff_acc.append(acc_p - acc_k)

        f1_p = f1_score(y_boot[mask], y_imp_proposed[mask], average='macro')
        f1_k = f1_score(y_boot[mask], y_imp_knn[mask], average='macro')
        diff_f1.append(f1_p - f1_k)

    diff_acc = np.array(diff_acc)
    diff_f1 = np.array(diff_f1)

    # 计算 p-value
    p_acc = (diff_acc <= 0).mean()
    p_f1 = (diff_f1 <= 0).mean()

    # 置信区间
    ci_acc = np.percentile(diff_acc, [2.5, 50, 97.5])
    ci_f1 = np.percentile(diff_f1, [2.5, 50, 97.5])

    results = {
        'diff_acc_mean': diff_acc.mean(),
        'diff_acc_median': ci_acc[1],
        'diff_acc_ci_lower': ci_acc[0],
        'diff_acc_ci_upper': ci_acc[2],
        'p_acc': p_acc,
        'diff_f1_mean': diff_f1.mean(),
        'diff_f1_median': ci_f1[1],
        'diff_f1_ci_lower': ci_f1[0],
        'diff_f1_ci_upper': ci_f1[2],
        'p_f1': p_f1,
        'n_boot': n_boot
    }

    return results


def run_paired_bootstrap_test(n_boot=2000):
    print("=" * 70)
    print("🧪 Paired Bootstrap Difference Test (Optimized)")
    print("=" * 70)

    print("\n[1/3] 加载数据...")
    data = DataPipeline().load()
    print(f"   ✅ 测试集: {len(data.y_test)} 样本")

    print("\n[2/3] 加载模型...")
    model = joblib.load("./saved_models/Balanced_XGBoost.pkl")
    print("   ✅ 模型加载完成")

    print("\n[3/3] 运行 Paired Bootstrap...")
    results = paired_bootstrap_test_optimized(data, model, n_boot=n_boot)

    print("\n" + "=" * 70)
    print("📊 Paired Bootstrap Test Results")
    print("=" * 70)

    print("\n【Accuracy Difference (Proposed - KNN)】")
    print(f"  Mean Difference:  {results['diff_acc_mean']:.4f}")
    print(f"  Median Difference: {results['diff_acc_median']:.4f}")
    print(f"  95% CI:           [{results['diff_acc_ci_lower']:.4f}, {results['diff_acc_ci_upper']:.4f}]")
    print(f"  p-value (H0: diff ≤ 0): {results['p_acc']:.6f}")

    if results['p_acc'] < 0.001:
        print(f"  Conclusion:       ✅ Proposed > KNN (p < 0.001)")
    elif results['p_acc'] < 0.05:
        print(f"  Conclusion:       ✅ Proposed > KNN (p < 0.05)")
    else:
        print(f"  Conclusion:       ❌ Not significant")

    print("\n【Macro-F1 Difference (Proposed - KNN)】")
    print(f"  Mean Difference:  {results['diff_f1_mean']:.4f}")
    print(f"  Median Difference: {results['diff_f1_median']:.4f}")
    print(f"  95% CI:           [{results['diff_f1_ci_lower']:.4f}, {results['diff_f1_ci_upper']:.4f}]")
    print(f"  p-value (H0: diff ≤ 0): {results['p_f1']:.6f}")

    if results['p_f1'] < 0.001:
        print(f"  Conclusion:       ✅ Proposed > KNN (p < 0.001)")
    elif results['p_f1'] < 0.05:
        print(f"  Conclusion:       ✅ Proposed > KNN (p < 0.05)")
    else:
        print(f"  Conclusion:       ❌ Not significant")

    # 保存
    import os
    os.makedirs("./results", exist_ok=True)
    pd.DataFrame([results]).to_csv("./results/paired_bootstrap_test.csv", index=False)

    print("\n✅ 结果已保存至: results/paired_bootstrap_test.csv")

    return results


if __name__ == "__main__":
    import os

    results = run_paired_bootstrap_test(n_boot=2000)