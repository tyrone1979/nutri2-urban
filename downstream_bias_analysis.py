#!/usr/bin/env python3
"""
下游营养学分析偏差评估
证明：用填补标签计算出的结论，与真实标签结论一致
"""
import numpy as np
import pandas as pd
from scipy import stats
import joblib
from model import DataPipeline


def compute_effect_size(y_true, y_imp, X_original, feature_idx=0):
    """计算城乡差异效应量"""
    # 真实标签下的效应量
    urban_true = X_original[y_true == 2, feature_idx]
    rural_true = X_original[y_true == 0, feature_idx]
    true_diff = urban_true.mean() - rural_true.mean()
    true_cohen_d = true_diff / np.sqrt((urban_true.var() + rural_true.var()) / 2)

    # 填补标签下的效应量
    urban_imp = X_original[y_imp == 2, feature_idx]
    rural_imp = X_original[y_imp == 0, feature_idx]
    imp_diff = urban_imp.mean() - rural_imp.mean()
    imp_cohen_d = imp_diff / np.sqrt((urban_imp.var() + rural_imp.var()) / 2)

    return {
        'true_diff': true_diff,
        'imp_diff': imp_diff,
        'bias': abs(imp_diff - true_diff),
        'bias_pct': abs(imp_diff - true_diff) / abs(true_diff) * 100 if true_diff != 0 else 0,
        'true_d': true_cohen_d,
        'imp_d': imp_cohen_d
    }


def run_downstream_analysis():
    print("=" * 70)
    print("🧪 下游营养流行病学分析偏差评估")
    print("=" * 70)

    # 加载数据
    data = DataPipeline().load()
    model = joblib.load("./saved_models/Balanced_XGBoost.pkl")

    # 模拟缺失并填补
    np.random.seed(42)
    mask = np.random.choice([True, False], size=len(data.y_test), p=[0.3, 0.7])
    y_masked = data.y_test.copy()
    y_masked[mask] = -1

    y_proba = model.predict_proba(data.X_test)
    y_imputed = y_masked.copy()
    y_imputed[mask] = np.argmax(y_proba[mask], axis=1)

    # 还原原始特征值（逆标准化）
    scaler = joblib.load("./saved_models/scaler.pkl")
    X_original = scaler.inverse_transform(data.X_test)

    features = ['FatER', 'CarbER', 'ProtER', 'Fat/Carb']

    print("\n📊 效应量偏差分析")
    print("=" * 70)
    print(f"{'Feature':<12} {'True Diff':<12} {'Imp Diff':<12} {'Bias':<10} {'Bias%':<8} {'Cohen d':<10}")
    print("-" * 70)

    results = []
    for idx, feat in enumerate(features):
        eff = compute_effect_size(data.y_test, y_imputed, X_original, idx)
        results.append(eff)
        print(f"{feat:<12} {eff['true_diff']:.4f}       {eff['imp_diff']:.4f}       "
              f"{eff['bias']:.4f}     {eff['bias_pct']:.1f}%      {eff['imp_d']:.3f}")

    # 显著性一致性检验
    print("\n📊 显著性一致性检验")
    print("=" * 70)

    for idx, feat in enumerate(features):
        # 真实标签下的p值
        urban_true = X_original[data.y_test == 2, idx]
        rural_true = X_original[data.y_test == 0, idx]
        _, p_true = stats.ttest_ind(urban_true, rural_true)

        # 填补标签下的p值
        urban_imp = X_original[y_imputed == 2, idx]
        rural_imp = X_original[y_imputed == 0, idx]
        _, p_imp = stats.ttest_ind(urban_imp, rural_imp)

        sig_true = "***" if p_true < 0.001 else "**" if p_true < 0.01 else "*" if p_true < 0.05 else "ns"
        sig_imp = "***" if p_imp < 0.001 else "**" if p_imp < 0.01 else "*" if p_imp < 0.05 else "ns"
        consistent = "✅" if sig_true == sig_imp else "❌"

        print(f"{feat:<12} True: p={p_true:.2e} {sig_true:<3} | Imp: p={p_imp:.2e} {sig_imp:<3} {consistent}")

    # 保存
    df_results = pd.DataFrame(results)
    df_results['feature'] = features
    df_results.to_csv("./results/downstream_bias.csv", index=False)
    print("\n✅ 结果已保存至: results/downstream_bias.csv")


if __name__ == "__main__":
    run_downstream_analysis()