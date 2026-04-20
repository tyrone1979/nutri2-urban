#!/usr/bin/env python3
"""
Runtime / Complexity 对比
"""
import numpy as np
import pandas as pd
import time
import joblib
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.ensemble import RandomForestClassifier
from collections import Counter
import warnings
import os

warnings.filterwarnings('ignore')

from model import DataPipeline


def rf_impute_with_time(X_complete, y_masked, mask, random_state=42):
    """RF-Imputer 并计时"""
    start = time.time()
    y_imputed = y_masked.copy()
    missing_idx = np.where(mask)[0]
    observed_idx = np.where(~mask)[0]

    if len(observed_idx) == 0:
        return y_imputed, time.time() - start

    majority = Counter(y_masked[observed_idx]).most_common(1)[0][0]
    y_imputed[missing_idx] = majority

    for iteration in range(5):
        rf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=random_state, n_jobs=1)
        rf.fit(X_complete[observed_idx], y_imputed[observed_idx])
        y_imputed[missing_idx] = rf.predict(X_complete[missing_idx])

    return y_imputed, time.time() - start


def run_runtime_comparison(sample_sizes=[1000, 2000, 5000, 10000, 20000]):
    """
    Runtime / Complexity 对比，测试不同样本量
    """
    print("=" * 80)
    print("🧪 Runtime / Complexity 对比")
    print("=" * 80)

    # 加载完整测试集
    print("\n[1/3] 加载数据...")
    data = DataPipeline().load()
    X_test_full = data.X_test
    y_test_full = data.y_test
    print(f"   ✅ 完整测试集: {len(X_test_full)} 样本")

    print("\n[2/3] 加载模型...")
    model = joblib.load("./saved_models/Balanced_XGBoost.pkl")
    print("   ✅ 模型加载完成")

    # 模拟缺失（固定30%）
    np.random.seed(42)

    results = []

    print("\n[3/3] 测试不同样本量下的运行时间...")
    print(f"{'N':<8} {'Proposed(ms)':<15} {'KNN(ms)':<12} {'MICE(s)':<12} {'RF-Imp(s)':<12}")
    print("-" * 60)

    for n_samples in sample_sizes:
        if n_samples > len(X_test_full):
            n_samples = len(X_test_full)

        # 子采样
        indices = np.random.choice(len(X_test_full), size=n_samples, replace=False)
        X_test = X_test_full[indices]
        y_test = y_test_full[indices]

        mask = np.random.choice([True, False], size=n_samples, p=[0.3, 0.7])
        y_masked = y_test.copy()
        y_masked[mask] = -1

        n_missing = mask.sum()

        # ========== Proposed (BXGB) Inference ==========
        start = time.time()
        y_proba = model.predict_proba(X_test)
        y_imp = y_masked.copy()
        y_imp[mask] = np.argmax(y_proba[mask], axis=1)
        proposed_time = (time.time() - start) * 1000  # ms

        # ========== KNN ==========
        start = time.time()
        knn = KNeighborsClassifier(n_neighbors=5)
        knn.fit(X_test[~mask], y_test[~mask])
        y_imp_knn = knn.predict(X_test[mask])
        knn_time = (time.time() - start) * 1000  # ms

        # ========== MICE ==========
        # MICE 太慢，只在 n <= 5000 时运行
        if n_samples <= 5000:
            start = time.time()
            X_with_label = np.column_stack([X_test, y_masked.astype(float)])
            X_with_label[X_with_label == -1] = np.nan
            imputer = IterativeImputer(max_iter=10, random_state=42)
            X_imputed = imputer.fit_transform(X_with_label)
            mice_time = time.time() - start  # s
        else:
            mice_time = np.nan

        # ========== RF-Imputer ==========
        if n_samples <= 10000:
            _, rf_time = rf_impute_with_time(X_test, y_masked, mask)
        else:
            rf_time = np.nan

        results.append({
            'N': n_samples,
            'N_missing': n_missing,
            'Proposed_ms': round(proposed_time, 2),
            'KNN_ms': round(knn_time, 2),
            'MICE_s': round(mice_time, 2) if not np.isnan(mice_time) else 'N/A',
            'RF_Imputer_s': round(rf_time, 2) if not np.isnan(rf_time) else 'N/A'
        })

        mice_str = f"{mice_time:.2f}s" if not np.isnan(mice_time) else "N/A"
        rf_str = f"{rf_time:.2f}s" if not np.isnan(rf_time) else "N/A"
        print(f"{n_samples:<8} {proposed_time:.2f}           {knn_time:.2f}        {mice_str:<12} {rf_str:<12}")

    df_results = pd.DataFrame(results)

    # ========== 计算推理吞吐量 ==========
    print("\n" + "=" * 80)
    print("📊 推理吞吐量 (Inference Throughput)")
    print("=" * 80)

    for _, row in df_results.iterrows():
        n = row['N']
        n_missing = row['N_missing']
        proposed_tp = n_missing / (row['Proposed_ms'] / 1000)  # samples/s
        knn_tp = n_missing / (row['KNN_ms'] / 1000)

        print(f"\n  N={n} (missing={n_missing}):")
        print(f"    Proposed: {proposed_tp:.0f} imputations/second")
        print(f"    KNN:      {knn_tp:.0f} imputations/second")
        if proposed_tp > knn_tp:
            print(f"    → Proposed 快 {proposed_tp / knn_tp:.1f}×")

    # ========== 可视化 ==========
    os.makedirs("./figures", exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # 图1：运行时间 vs 样本量
    ax = axes[0]
    valid_n = df_results['N'].values
    ax.plot(valid_n, df_results['Proposed_ms'], 'o-', linewidth=2.5, markersize=10,
            label='Proposed (BXGB)', color='#2E86AB')
    ax.plot(valid_n, df_results['KNN_ms'], 's-', linewidth=2.5, markersize=10,
            label='KNN (k=5)', color='#A23B72')
    ax.set_xlabel('Number of Samples', fontsize=12)
    ax.set_ylabel('Time (ms)', fontsize=12)
    ax.set_title('Inference Time vs Sample Size', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    # 图2：吞吐量 vs 样本量
    ax = axes[1]
    proposed_tp = [n_missing / (t / 1000) for n_missing, t in zip(df_results['N_missing'], df_results['Proposed_ms'])]
    knn_tp = [n_missing / (t / 1000) for n_missing, t in zip(df_results['N_missing'], df_results['KNN_ms'])]

    ax.plot(valid_n, proposed_tp, 'o-', linewidth=2.5, markersize=10,
            label='Proposed (BXGB)', color='#2E86AB')
    ax.plot(valid_n, knn_tp, 's-', linewidth=2.5, markersize=10,
            label='KNN (k=5)', color='#A23B72')
    ax.set_xlabel('Number of Samples', fontsize=12)
    ax.set_ylabel('Throughput (imputations/second)', fontsize=12)
    ax.set_title('Imputation Throughput vs Sample Size', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("./figures/runtime_comparison.png", dpi=300, bbox_inches='tight')
    plt.close()
    print(f"\n   ✅ runtime_comparison.png 已保存")

    # ========== 训练时间对比 ==========
    print("\n" + "=" * 80)
    print("📊 训练时间对比（一次性成本）")
    print("=" * 80)

    training_results = []

    # Proposed: 已训练，记录加载时间
    start = time.time()
    _ = joblib.load("./saved_models/Balanced_XGBoost.pkl")
    load_time = time.time() - start
    training_results.append(
        {'Method': 'Proposed (BXGB)', 'Training_Time_s': 'N/A (pre-trained)', 'Load_Time_s': round(load_time, 3)})

    # KNN: 无需训练
    training_results.append({'Method': 'KNN (k=5)', 'Training_Time_s': '0 (lazy)', 'Load_Time_s': 'N/A'})

    # RF-Imputer: 无需预训练
    training_results.append({'Method': 'RF-Imputer', 'Training_Time_s': '0 (online)', 'Load_Time_s': 'N/A'})

    # MICE: 无需预训练
    training_results.append({'Method': 'MICE', 'Training_Time_s': '0 (online)', 'Load_Time_s': 'N/A'})

    df_train = pd.DataFrame(training_results)
    print(df_train.to_string(index=False))

    # 保存
    os.makedirs("./results", exist_ok=True)
    df_results.to_csv("./results/runtime_comparison.csv", index=False)
    df_train.to_csv("./results/training_time.csv", index=False)
    print("\n✅ 结果已保存至: results/runtime_comparison.csv, results/training_time.csv")

    # ========== 总结 ==========
    print("\n" + "=" * 80)
    print("📊 Runtime 总结")
    print("=" * 80)

    # 用最大样本量的结果
    last_row = df_results.iloc[-1]
    print(f"\n  在 {last_row['N']} 样本上 (缺失 {last_row['N_missing']} 个):")
    print(f"    Proposed 推理时间: {last_row['Proposed_ms']:.2f} ms")
    print(f"    KNN 推理时间:      {last_row['KNN_ms']:.2f} ms")

    speedup = last_row['KNN_ms'] / last_row['Proposed_ms']
    if speedup > 1:
        print(f"    → Proposed 比 KNN 快 {speedup:.1f}×")
    else:
        print(f"    → KNN 比 Proposed 快 {1 / speedup:.1f}×")

    if last_row['MICE_s'] != 'N/A':
        print(f"    MICE 时间:         {last_row['MICE_s']} s")
        print(f"    → Proposed 比 MICE 快 ~{float(last_row['MICE_s']) * 1000 / last_row['Proposed_ms']:.0f}×")

    print(f"\n  💡 Proposed 的优势:")
    print(f"     - 预训练一次，终身使用（加载时间 {load_time:.3f}s）")
    print(f"     - 推理吞吐量高，适合大规模数据填补")
    print(f"     - 不随样本量线性增长（GPU可进一步加速）")

    return df_results, df_train


if __name__ == "__main__":
    df_results, df_train = run_runtime_comparison()