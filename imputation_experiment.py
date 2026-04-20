#!/usr/bin/env python3
"""
缺失标签填补实验（增强版）
- 支持不同缺失率下的稳健性测试
- 支持填补后标签分布保真度评估
"""
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, f1_score, cohen_kappa_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from scipy.spatial.distance import jensenshannon
from scipy.stats import wasserstein_distance
from collections import Counter
import os
import warnings

warnings.filterwarnings('ignore')

from model import DataPipeline, PROVINCE_SHORT


def compute_distribution_fidelity(y_true, y_imputed, n_classes=3):
    """
    计算填补后标签分布与真实分布的保真度

    Returns:
        dict: 包含各类指标
    """
    true_counts = np.bincount(y_true, minlength=n_classes)
    imp_counts = np.bincount(y_imputed, minlength=n_classes)

    true_dist = true_counts / len(y_true)
    imp_dist = imp_counts / len(y_imputed)

    # Jensen-Shannon散度
    js_div = jensenshannon(true_dist, imp_dist)

    # Wasserstein距离
    wass_dist = wasserstein_distance(range(n_classes), range(n_classes),
                                     true_dist, imp_dist)

    # 各类别比例差异
    class_diffs = imp_dist - true_dist
    max_class_diff = np.max(np.abs(class_diffs))

    return {
        'true_dist': true_dist,
        'imp_dist': imp_dist,
        'js_divergence': js_div,
        'wasserstein': wass_dist,
        'max_class_diff': max_class_diff,
        'class_diffs': class_diffs
    }


def run_single_imputation_experiment(data, model, missing_rate=0.3, random_state=42, verbose=True):
    """
    单次填补实验

    Args:
        data: DataPipeline 实例（已加载）
        model: 已训练的模型
        missing_rate: 缺失率
        random_state: 随机种子
        verbose: 是否打印详细信息

    Returns:
        results, class_acc_results, fidelity_results
    """

    X_test = data.X_test
    y_test = data.y_test

    # 模拟缺失
    np.random.seed(random_state)
    n_samples = len(y_test)
    mask = np.random.choice([True, False], size=n_samples, p=[missing_rate, 1 - missing_rate])
    y_masked = y_test.copy()
    y_masked[mask] = -1

    if verbose:
        print("\n" + "=" * 70)
        print(f"🧪 缺失标签填补实验 (Missing Rate = {missing_rate * 100:.0f}%)")
        print("=" * 70)
        print(f"\n📊 数据统计:")
        print(f"   总样本数: {n_samples}")
        print(f"   缺失样本数: {mask.sum()} ({missing_rate * 100:.0f}%)")
        print(f"   完整样本数: {(~mask).sum()}")

    # 填补方法
    # 1. 多数类
    majority_class = Counter(y_test[~mask]).most_common(1)[0][0]
    y_imp_majority = y_masked.copy()
    y_imp_majority[mask] = majority_class

    # 2. KNN
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(X_test[~mask], y_test[~mask])
    y_imp_knn = y_masked.copy()
    y_imp_knn[mask] = knn.predict(X_test[mask])

    # 3. MICE
    X_with_label = np.column_stack([X_test, y_masked.astype(float)])
    X_with_label[X_with_label == -1] = np.nan
    imputer = IterativeImputer(max_iter=10, random_state=random_state)
    X_imputed = imputer.fit_transform(X_with_label)
    y_imp_mice = X_imputed[:, -1].round().astype(int)
    y_imp_mice = np.clip(y_imp_mice, 0, 2)

    # 4. Proposed
    y_proba = model.predict_proba(X_test)
    y_imp_proposed = y_masked.copy()
    y_imp_proposed[mask] = np.argmax(y_proba[mask], axis=1)

    methods = {
        'Majority': y_imp_majority,
        'KNN (k=5)': y_imp_knn,
        'MICE': y_imp_mice,
        'Proposed (BXGB)': y_imp_proposed
    }

    # 评估
    results = []
    fidelity_results = {}

    for name, y_imp in methods.items():
        y_true_masked = y_test[mask]
        y_pred_masked = y_imp[mask]

        acc = accuracy_score(y_true_masked, y_pred_masked)
        f1 = f1_score(y_true_masked, y_pred_masked, average='macro')
        kappa = cohen_kappa_score(y_true_masked, y_pred_masked)

        fidelity = compute_distribution_fidelity(y_test, y_imp)
        fidelity_results[name] = fidelity

        results.append({
            'missing_rate': missing_rate,
            'Method': name,
            'Accuracy': round(acc, 4),
            'Macro_F1': round(f1, 4),
            'Kappa': round(kappa, 4),
            'JS_Div': round(fidelity['js_divergence'], 6),
            'Wasserstein': round(fidelity['wasserstein'], 6),
            'Max_Class_Diff': round(fidelity['max_class_diff'], 4)
        })

    # 各类别填补准确率
    class_names = {0: 'Rural', 1: 'Transitional', 2: 'Urban'}
    class_acc_results = []

    for class_idx, class_name in class_names.items():
        class_mask = (y_test == class_idx) & mask
        if class_mask.sum() > 0:
            row = {'Class': class_name, 'n': class_mask.sum()}
            for name, y_imp in methods.items():
                acc = accuracy_score(y_test[class_mask], y_imp[class_mask])
                row[name] = round(acc, 4)
            class_acc_results.append(row)

    if verbose:
        print("\n" + "=" * 70)
        print("📊 填补效果对比（仅在缺失样本上评估）")
        print("=" * 70)
        print(f"{'Method':<20} {'Accuracy':<12} {'Macro-F1':<12} {'Kappa':<12} {'JS Div':<10}")
        print("-" * 66)
        for r in results:
            print(f"{r['Method']:<20} {r['Accuracy']:.4f}       {r['Macro_F1']:.4f}       "
                  f"{r['Kappa']:.4f}     {r['JS_Div']:.6f}")

        print("\n" + "=" * 70)
        print("📊 各类别填补准确率")
        print("=" * 70)
        df_class = pd.DataFrame(class_acc_results)
        print(df_class.to_string(index=False))

    return results, class_acc_results, fidelity_results


def run_robustness_test(missing_rates=[0.1, 0.2, 0.3, 0.4, 0.5], random_state=42):
    """
    不同缺失率下的稳健性测试
    """
    print("\n" + "=" * 80)
    print("🧪 稳健性测试：不同缺失率下的填补性能")
    print("=" * 80)

    # 只加载一次数据
    print("\n[1/2] 加载数据...")
    data = DataPipeline().load()
    print("   ✅ 数据加载完成")

    print("\n[2/2] 加载模型...")
    model = joblib.load("./saved_models/Balanced_XGBoost.pkl")
    print("   ✅ 模型加载完成")

    all_results = []
    all_class_results = []
    all_fidelity = {}

    for rate in missing_rates:
        print(f"\n{'=' * 50}")
        print(f"测试缺失率: {rate * 100:.0f}%")
        print(f"{'=' * 50}")

        results, class_results, fidelity = run_single_imputation_experiment(
            data=data,
            model=model,
            missing_rate=rate,
            random_state=random_state,
            verbose=False
        )

        # 只打印Proposed的结果
        for r in results:
            if r['Method'] == 'Proposed (BXGB)':
                print(f"  Proposed: Acc={r['Accuracy']:.4f}, F1={r['Macro_F1']:.4f}, "
                      f"Kappa={r['Kappa']:.4f}, JS={r['JS_Div']:.6f}")

        all_results.extend(results)
        all_fidelity[rate] = fidelity

        for row in class_results:
            row['missing_rate'] = rate
            all_class_results.append(row)

    # 汇总表格
    df_all = pd.DataFrame(all_results)
    df_pivot = df_all[df_all['Method'] == 'Proposed (BXGB)'].copy()

    print("\n" + "=" * 80)
    print("📊 稳健性测试汇总 - Proposed (BXGB)")
    print("=" * 80)
    print(
        df_pivot[['missing_rate', 'Accuracy', 'Macro_F1', 'Kappa', 'JS_Div', 'Max_Class_Diff']].to_string(index=False))

    # 保存
    os.makedirs("./results", exist_ok=True)
    os.makedirs("./figures", exist_ok=True)
    df_all.to_csv("./results/imputation_robustness_all.csv", index=False)
    df_pivot.to_csv("./results/imputation_robustness_proposed.csv", index=False)

    # 可视化
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # 图1：Accuracy
    ax = axes[0]
    for method in ['Proposed (BXGB)', 'KNN (k=5)', 'MICE']:
        method_data = df_all[df_all['Method'] == method]
        ax.plot(method_data['missing_rate'], method_data['Accuracy'],
                marker='o', linewidth=2, markersize=8, label=method)
    ax.set_xlabel('Missing Rate')
    ax.set_ylabel('Accuracy')
    ax.set_title('Imputation Accuracy vs Missing Rate')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1)

    # 图2：Macro-F1
    ax = axes[1]
    for method in ['Proposed (BXGB)', 'KNN (k=5)', 'MICE']:
        method_data = df_all[df_all['Method'] == method]
        ax.plot(method_data['missing_rate'], method_data['Macro_F1'],
                marker='o', linewidth=2, markersize=8, label=method)
    ax.set_xlabel('Missing Rate')
    ax.set_ylabel('Macro-F1')
    ax.set_title('Macro-F1 vs Missing Rate')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1)

    # 图3：JS Divergence
    ax = axes[2]
    for method in ['Proposed (BXGB)', 'KNN (k=5)', 'MICE', 'Majority']:
        method_data = df_all[df_all['Method'] == method]
        ax.plot(method_data['missing_rate'], method_data['JS_Div'],
                marker='o', linewidth=2, markersize=8, label=method)
    ax.set_xlabel('Missing Rate')
    ax.set_ylabel('Jensen-Shannon Divergence')
    ax.set_title('Distribution Fidelity vs Missing Rate')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("./figures/imputation_robustness.png", dpi=300, bbox_inches='tight')
    plt.close()
    print("\n✅ 稳健性测试图已保存至: figures/imputation_robustness.png")

    return df_all, all_fidelity, data, model


def print_distribution_fidelity_report(fidelity_results, missing_rate=0.3):
    """
    打印分布保真度详细报告
    """
    print("\n" + "=" * 80)
    print(f"📊 分布保真度详细报告 (Missing Rate = {missing_rate * 100:.0f}%)")
    print("=" * 80)

    class_names = ['Rural', 'Transitional', 'Urban']

    for method, fidelity in fidelity_results.items():
        print(f"\n{method}:")
        print(f"  Jensen-Shannon Divergence: {fidelity['js_divergence']:.6f}")
        print(f"  Wasserstein Distance:      {fidelity['wasserstein']:.6f}")
        print(f"  Max Class Proportion Diff: {fidelity['max_class_diff']:.4f}")
        print(f"  Class Proportion Differences:")
        for i, name in enumerate(class_names):
            diff = fidelity['class_diffs'][i]
            arrow = "↑" if diff > 0 else "↓" if diff < 0 else "="
            print(f"    {name:<12}: True={fidelity['true_dist'][i]:.4f}, "
                  f"Imp={fidelity['imp_dist'][i]:.4f}, Diff={diff:+.4f} {arrow}")


def run_full_analysis():
    """
    完整分析：单次实验 + 稳健性测试 + 分布保真度报告
    """
    import os
    os.makedirs("./results", exist_ok=True)
    os.makedirs("./figures", exist_ok=True)

    # 1. 加载数据和模型
    print("=" * 70)
    print("🚀 开始完整分析")
    print("=" * 70)

    print("\n[1/3] 加载数据...")
    data = DataPipeline().load()
    print("   ✅ 数据加载完成")

    print("\n[2/3] 加载模型...")
    model = joblib.load("./saved_models/Balanced_XGBoost.pkl")
    print("   ✅ 模型加载完成")

    # 2. 单次实验（30%缺失率，详细输出）
    print("\n[3/3] 运行填补实验...")
    results, class_results, fidelity = run_single_imputation_experiment(
        data=data,
        model=model,
        missing_rate=0.3,
        verbose=True
    )

    # 3. 分布保真度详细报告
    print_distribution_fidelity_report(fidelity, missing_rate=0.3)

    # 4. 稳健性测试
    df_all, all_fidelity, _, _ = run_robustness_test(
        missing_rates=[0.1, 0.2, 0.3, 0.4, 0.5]
    )

    # 5. 保存分布保真度汇总
    print("\n" + "=" * 80)
    print("📊 不同缺失率下的分布保真度汇总 (Proposed BXGB)")
    print("=" * 80)

    fidelity_summary = []
    for rate, f_dict in all_fidelity.items():
        f = f_dict['Proposed (BXGB)']
        fidelity_summary.append({
            'missing_rate': rate,
            'JS_Div': f['js_divergence'],
            'Wasserstein': f['wasserstein'],
            'Max_Class_Diff': f['max_class_diff']
        })
        print(f"  {rate * 100:.0f}%: JS={f['js_divergence']:.6f}, "
              f"Wasserstein={f['wasserstein']:.6f}, Max_Diff={f['max_class_diff']:.4f}")

    df_fidelity = pd.DataFrame(fidelity_summary)
    df_fidelity.to_csv("./results/distribution_fidelity.csv", index=False)

    print("\n" + "=" * 80)
    print("✅ 所有结果已保存:")
    print("   - results/imputation_robustness_all.csv")
    print("   - results/imputation_robustness_proposed.csv")
    print("   - results/distribution_fidelity.csv")
    print("   - figures/imputation_robustness.png")
    print("=" * 80)

    return df_all, all_fidelity


if __name__ == "__main__":
    df_all, all_fidelity = run_full_analysis()