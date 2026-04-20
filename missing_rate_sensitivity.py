#!/usr/bin/env python3
"""
缺失率敏感性分析：10%, 20%, 30%, 40%, 50%, 60%, 70%
"""
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, f1_score, cohen_kappa_score
from sklearn.neighbors import KNeighborsClassifier
from scipy.spatial.distance import jensenshannon
from collections import Counter
import warnings
import os

warnings.filterwarnings('ignore')

from model import DataPipeline


def compute_fidelity(y_true, y_imp, n_classes=3):
    """计算分布保真度"""
    true_counts = np.bincount(y_true, minlength=n_classes)
    imp_counts = np.bincount(y_imp, minlength=n_classes)
    true_dist = true_counts / len(y_true)
    imp_dist = imp_counts / len(y_imp)
    js_div = jensenshannon(true_dist, imp_dist)
    return js_div


def run_missing_rate_sensitivity(missing_rates=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7], random_state=42):
    """
    缺失率敏感性分析
    """
    print("=" * 80)
    print("🧪 缺失率敏感性分析")
    print("=" * 80)

    # 加载数据
    print("\n[1/3] 加载数据...")
    data = DataPipeline().load()
    X_test = data.X_test
    y_test = data.y_test
    print(f"   ✅ 测试集: {len(X_test)} 样本")

    print("\n[2/3] 加载模型...")
    model = joblib.load("./saved_models/Balanced_XGBoost.pkl")
    print("   ✅ 模型加载完成")

    all_results = []
    class_acc_results = []

    print("\n[3/3] 运行不同缺失率测试...")
    print(f"{'Rate':<8} {'Proposed Acc':<14} {'Proposed F1':<14} {'KNN Acc':<12} {'KNN F1':<12} {'Proposed JS':<12}")
    print("-" * 75)

    class_names = {0: 'Rural', 1: 'Transitional', 2: 'Urban'}

    for rate in missing_rates:
        np.random.seed(random_state)
        n_samples = len(y_test)
        mask = np.random.choice([True, False], size=n_samples, p=[rate, 1 - rate])
        y_masked = y_test.copy()
        y_masked[mask] = -1

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
        acc_p = accuracy_score(y_test[mask], y_imp_proposed[mask])
        f1_p = f1_score(y_test[mask], y_imp_proposed[mask], average='macro')
        kappa_p = cohen_kappa_score(y_test[mask], y_imp_proposed[mask])
        js_p = compute_fidelity(y_test, y_imp_proposed)

        acc_k = accuracy_score(y_test[mask], y_imp_knn[mask])
        f1_k = f1_score(y_test[mask], y_imp_knn[mask], average='macro')
        js_k = compute_fidelity(y_test, y_imp_knn)

        all_results.append({
            'missing_rate': rate,
            'n_missing': mask.sum(),
            'proposed_acc': acc_p,
            'proposed_f1': f1_p,
            'proposed_kappa': kappa_p,
            'proposed_js': js_p,
            'knn_acc': acc_k,
            'knn_f1': f1_k,
            'knn_js': js_k
        })

        print(f"{rate * 100:>3.0f}%   {acc_p:.4f}        {f1_p:.4f}        {acc_k:.4f}      {f1_k:.4f}      {js_p:.6f}")

        # 记录各类别准确率
        for class_idx, class_name in class_names.items():
            class_mask = (y_test == class_idx) & mask
            if class_mask.sum() > 0:
                acc_class_p = accuracy_score(y_test[class_mask], y_imp_proposed[class_mask])
                acc_class_k = accuracy_score(y_test[class_mask], y_imp_knn[class_mask])
                class_acc_results.append({
                    'missing_rate': rate,
                    'class': class_name,
                    'n': class_mask.sum(),
                    'proposed_acc': acc_class_p,
                    'knn_acc': acc_class_k
                })

    df_results = pd.DataFrame(all_results)
    df_class = pd.DataFrame(class_acc_results)

    # 保存
    os.makedirs("./results", exist_ok=True)
    os.makedirs("./figures", exist_ok=True)
    df_results.to_csv("./results/missing_rate_sensitivity.csv", index=False)
    df_class.to_csv("./results/missing_rate_class_acc.csv", index=False)

    # ========== 可视化 ==========
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    # 1. Accuracy vs Missing Rate
    ax = axes[0, 0]
    ax.plot(np.array(missing_rates) * 100, df_results['proposed_acc'],
            'o-', linewidth=2.5, markersize=10, label='Proposed (BXGB)', color='#2E86AB')
    ax.plot(np.array(missing_rates) * 100, df_results['knn_acc'],
            's-', linewidth=2.5, markersize=10, label='KNN (k=5)', color='#A23B72')
    ax.set_xlabel('Missing Rate (%)', fontsize=12)
    ax.set_ylabel('Accuracy', fontsize=12)
    ax.set_title('Accuracy vs Missing Rate', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0.65, 0.85)

    # 2. Macro-F1 vs Missing Rate
    ax = axes[0, 1]
    ax.plot(np.array(missing_rates) * 100, df_results['proposed_f1'],
            'o-', linewidth=2.5, markersize=10, label='Proposed (BXGB)', color='#2E86AB')
    ax.plot(np.array(missing_rates) * 100, df_results['knn_f1'],
            's-', linewidth=2.5, markersize=10, label='KNN (k=5)', color='#A23B72')
    ax.set_xlabel('Missing Rate (%)', fontsize=12)
    ax.set_ylabel('Macro-F1', fontsize=12)
    ax.set_title('Macro-F1 vs Missing Rate', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0.65, 0.80)

    # 3. JS Divergence vs Missing Rate
    ax = axes[1, 0]
    ax.plot(np.array(missing_rates) * 100, df_results['proposed_js'],
            'o-', linewidth=2.5, markersize=10, label='Proposed (BXGB)', color='#2E86AB')
    ax.plot(np.array(missing_rates) * 100, df_results['knn_js'],
            's-', linewidth=2.5, markersize=10, label='KNN (k=5)', color='#A23B72')
    ax.set_xlabel('Missing Rate (%)', fontsize=12)
    ax.set_ylabel('JS Divergence', fontsize=12)
    ax.set_title('Distribution Fidelity vs Missing Rate', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    # 4. Class-specific Accuracy
    ax = axes[1, 1]
    for class_name in ['Rural', 'Transitional', 'Urban']:
        class_data = df_class[df_class['class'] == class_name]
        ax.plot(np.array(missing_rates) * 100, class_data['proposed_acc'],
                'o-', linewidth=2, markersize=8, label=f'{class_name} (Proposed)')
    ax.set_xlabel('Missing Rate (%)', fontsize=12)
    ax.set_ylabel('Accuracy', fontsize=12)
    ax.set_title('Class-Specific Accuracy (Proposed)', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1.05)

    plt.tight_layout()
    plt.savefig("./figures/missing_rate_sensitivity.png", dpi=300, bbox_inches='tight')
    plt.close()
    print(f"\n   ✅ missing_rate_sensitivity.png 已保存")

    # ========== 额外：Proposed vs KNN 差异图 ==========
    fig, ax = plt.subplots(figsize=(10, 6))

    x = np.array(missing_rates) * 100
    acc_diff = df_results['proposed_acc'] - df_results['knn_acc']
    f1_diff = df_results['proposed_f1'] - df_results['knn_f1']

    ax.plot(x, acc_diff, 'o-', linewidth=2.5, markersize=10,
            label='Accuracy Difference', color='#F18F01')
    ax.plot(x, f1_diff, 's-', linewidth=2.5, markersize=10,
            label='Macro-F1 Difference', color='#006E90')
    ax.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    ax.set_xlabel('Missing Rate (%)', fontsize=12)
    ax.set_ylabel('Proposed - KNN', fontsize=12)
    ax.set_title('Performance Advantage of Proposed over KNN', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("./figures/missing_rate_advantage.png", dpi=300, bbox_inches='tight')
    plt.close()
    print(f"   ✅ missing_rate_advantage.png 已保存")

    # 打印汇总
    print("\n" + "=" * 80)
    print("📊 缺失率敏感性汇总")
    print("=" * 80)
    print(df_results[['missing_rate', 'proposed_acc', 'proposed_f1', 'knn_acc', 'knn_f1', 'proposed_js']].to_string(
        index=False))

    # 计算性能下降率
    acc_10 = df_results[df_results['missing_rate'] == 0.1]['proposed_acc'].values[0]
    acc_70 = df_results[df_results['missing_rate'] == 0.7]['proposed_acc'].values[0]
    f1_10 = df_results[df_results['missing_rate'] == 0.1]['proposed_f1'].values[0]
    f1_70 = df_results[df_results['missing_rate'] == 0.7]['proposed_f1'].values[0]

    print(f"\n📊 极端缺失率下性能保持:")
    print(f"   Accuracy: {acc_10:.4f} (10%) → {acc_70:.4f} (70%)  下降: {(acc_10 - acc_70) * 100:.1f} 个百分点")
    print(f"   Macro-F1: {f1_10:.4f} (10%) → {f1_70:.4f} (70%)  下降: {(f1_10 - f1_70) * 100:.1f} 个百分点")

    # 平均优势
    avg_acc_adv = (df_results['proposed_acc'] - df_results['knn_acc']).mean()
    avg_f1_adv = (df_results['proposed_f1'] - df_results['knn_f1']).mean()
    print(f"\n📊 Proposed 相对于 KNN 的平均优势:")
    print(f"   Accuracy: +{avg_acc_adv:.4f} ({avg_acc_adv * 100:.1f} 个百分点)")
    print(f"   Macro-F1: +{avg_f1_adv:.4f} ({avg_f1_adv * 100:.1f} 个百分点)")

    # 各类别汇总
    print("\n" + "=" * 80)
    print("📊 各类别准确率随缺失率变化")
    print("=" * 80)
    for class_name in ['Rural', 'Transitional', 'Urban']:
        class_data = df_class[df_class['class'] == class_name]
        print(f"\n{class_name}:")
        for _, row in class_data.iterrows():
            print(f"  {row['missing_rate'] * 100:.0f}%: Proposed={row['proposed_acc']:.4f}, KNN={row['knn_acc']:.4f}")

    return df_results, df_class


if __name__ == "__main__":
    df_results, df_class = run_missing_rate_sensitivity()