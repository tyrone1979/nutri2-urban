#!/usr/bin/env python3
"""
增强版 Baseline 对比：Majority, KNN, MICE, RandomForest Imputer, Proposed
"""
import numpy as np
import pandas as pd
import joblib
from sklearn.metrics import accuracy_score, f1_score, cohen_kappa_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.ensemble import RandomForestClassifier
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


def rf_impute(X_complete, y_masked, mask, random_state=42):
    """
    用 RandomForest 迭代填补缺失标签
    模拟 MissForest 的核心思想
    """
    y_imputed = y_masked.copy()
    missing_idx = np.where(mask)[0]
    observed_idx = np.where(~mask)[0]

    if len(observed_idx) == 0:
        return y_imputed

    # 初始填补（用多数类）
    majority = Counter(y_masked[observed_idx]).most_common(1)[0][0]
    y_imputed[missing_idx] = majority

    # 迭代填补
    for iteration in range(5):
        # 用当前填补后的标签训练 RF
        rf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=random_state, n_jobs=1)
        rf.fit(X_complete[observed_idx], y_imputed[observed_idx])

        # 预测缺失样本
        y_imputed[missing_idx] = rf.predict(X_complete[missing_idx])

    return y_imputed


def run_enhanced_baseline_comparison(missing_rate=0.3, random_state=42):
    """
    增强版 Baseline 对比
    """
    print("=" * 80)
    print(f"🧪 增强版 Baseline 对比 (Missing Rate = {missing_rate * 100:.0f}%)")
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

    # 模拟缺失
    np.random.seed(random_state)
    n_samples = len(y_test)
    mask = np.random.choice([True, False], size=n_samples, p=[missing_rate, 1 - missing_rate])
    y_masked = y_test.copy()
    y_masked[mask] = -1

    print(f"\n[3/3] 运行填补对比...")
    print(f"   缺失样本数: {mask.sum()} ({missing_rate * 100:.0f}%)")

    # ========== 1. Majority ==========
    majority_class = Counter(y_test[~mask]).most_common(1)[0][0]
    y_imp_majority = y_masked.copy()
    y_imp_majority[mask] = majority_class

    # ========== 2. KNN ==========
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(X_test[~mask], y_test[~mask])
    y_imp_knn = y_masked.copy()
    y_imp_knn[mask] = knn.predict(X_test[mask])

    # ========== 3. MICE (IterativeImputer) ==========
    print("   运行 MICE...")
    X_with_label = np.column_stack([X_test, y_masked.astype(float)])
    X_with_label[X_with_label == -1] = np.nan
    imputer = IterativeImputer(max_iter=10, random_state=random_state)
    X_imputed = imputer.fit_transform(X_with_label)
    y_imp_mice = X_imputed[:, -1].round().astype(int)
    y_imp_mice = np.clip(y_imp_mice, 0, 2)

    # ========== 4. RandomForest Imputer (代替 MissForest) ==========
    print("   运行 RandomForest Imputer...")
    y_imp_rf = rf_impute(X_test, y_masked, mask, random_state=random_state)

    # ========== 5. Proposed ==========
    y_proba = model.predict_proba(X_test)
    y_imp_proposed = y_masked.copy()
    y_imp_proposed[mask] = np.argmax(y_proba[mask], axis=1)

    # ========== 评估 ==========
    methods = {
        'Majority': y_imp_majority,
        'KNN (k=5)': y_imp_knn,
        'MICE': y_imp_mice,
        'RF-Imputer': y_imp_rf,
        'Proposed (BXGB)': y_imp_proposed
    }

    results = []
    print("\n" + "=" * 90)
    print("📊 填补效果对比")
    print("=" * 90)
    print(f"{'Method':<20} {'Accuracy':<12} {'Macro-F1':<12} {'Kappa':<12} {'JS Div':<10} {'Max Diff':<10}")
    print("-" * 78)

    for name, y_imp in methods.items():
        acc = accuracy_score(y_test[mask], y_imp[mask])
        f1 = f1_score(y_test[mask], y_imp[mask], average='macro')
        kappa = cohen_kappa_score(y_test[mask], y_imp[mask])
        js_div, max_diff = compute_fidelity(y_test, y_imp)

        results.append({
            'Method': name,
            'Accuracy': round(acc, 4),
            'Macro_F1': round(f1, 4),
            'Kappa': round(kappa, 4),
            'JS_Div': round(js_div, 6),
            'Max_Diff': round(max_diff, 4)
        })
        print(f"{name:<20} {acc:.4f}       {f1:.4f}       {kappa:.4f}     {js_div:.6f}   {max_diff:.4f}")

    # 各类别准确率
    print("\n" + "=" * 90)
    print("📊 各类别填补准确率")
    print("=" * 90)
    class_names = {0: 'Rural', 1: 'Transitional', 2: 'Urban'}

    class_acc = []
    for class_idx, class_name in class_names.items():
        class_mask = (y_test == class_idx) & mask
        if class_mask.sum() > 0:
            row = {'Class': class_name, 'n': class_mask.sum()}
            for name, y_imp in methods.items():
                row[name] = round(accuracy_score(y_test[class_mask], y_imp[class_mask]), 4)
            class_acc.append(row)

    df_class = pd.DataFrame(class_acc)
    print(df_class.to_string(index=False))

    # 保存
    import os
    os.makedirs("./results", exist_ok=True)
    df_results = pd.DataFrame(results)
    df_results.to_csv("./results/enhanced_baseline_comparison.csv", index=False)
    df_class.to_csv("./results/enhanced_baseline_class_acc.csv", index=False)

    print("\n✅ 结果已保存至: results/enhanced_baseline_comparison.csv")

    # 打印优势总结
    print("\n" + "=" * 80)
    print("📊 Proposed vs 最强 Baseline")
    print("=" * 80)

    # 找最强 baseline（排除 Proposed）
    baseline_methods = [m for m in methods.keys() if m != 'Proposed (BXGB)']
    best_acc = max([r['Accuracy'] for r in results if r['Method'] in baseline_methods])
    best_f1 = max([r['Macro_F1'] for r in results if r['Method'] in baseline_methods])

    proposed_acc = [r['Accuracy'] for r in results if r['Method'] == 'Proposed (BXGB)'][0]
    proposed_f1 = [r['Macro_F1'] for r in results if r['Method'] == 'Proposed (BXGB)'][0]

    print(f"   最强 Baseline Accuracy: {best_acc:.4f}")
    print(f"   Proposed Accuracy:      {proposed_acc:.4f} (+{(proposed_acc - best_acc) * 100:.1f} 个百分点)")
    print(f"   最强 Baseline F1:       {best_f1:.4f}")
    print(f"   Proposed F1:            {proposed_f1:.4f} (+{(proposed_f1 - best_f1) * 100:.1f} 个百分点)")

    return df_results, df_class


if __name__ == "__main__":
    df_results, df_class = run_enhanced_baseline_comparison(missing_rate=0.3)