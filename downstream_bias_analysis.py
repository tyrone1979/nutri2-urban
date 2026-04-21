#!/usr/bin/env python3
"""
下游营养学分析偏差评估 + 生成 Figure 5
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
import joblib
import warnings
import os

warnings.filterwarnings('ignore')

from model import DataPipeline


def save_figure(fig, filename_base, dpi=300):
    """保存图片为 PNG 和 TIFF 格式"""
    os.makedirs("./figures", exist_ok=True)
    fig.savefig(f"./figures/{filename_base}.png", dpi=dpi, bbox_inches='tight')
    fig.savefig(f"./figures/{filename_base}.tiff", dpi=dpi, bbox_inches='tight',
                format='tiff', pil_kwargs={'compression': 'tiff_lzw'})
    print(f"   ✅ {filename_base}.png 和 {filename_base}.tiff 已保存")


def bootstrap_effect_size(y_true, X_original, feature_idx, n_boot=1000):
    """Bootstrap 计算效应量的置信区间"""
    n = len(y_true)
    true_d = []
    imp_d = []

    for _ in range(n_boot):
        idx = np.random.choice(n, n, replace=True)
        y_boot = y_true[idx]
        X_boot = X_original[idx]

        urban = X_boot[y_boot == 2, feature_idx]
        rural = X_boot[y_boot == 0, feature_idx]

        if len(urban) > 1 and len(rural) > 1:
            diff = urban.mean() - rural.mean()
            d = diff / np.sqrt((urban.var() + rural.var()) / 2)
            true_d.append(d)

    return np.percentile(true_d, [2.5, 50, 97.5])


def compute_effect_size(y_true, y_imp, X_original, feature_idx=0):
    """计算城乡差异效应量"""
    urban_true = X_original[y_true == 2, feature_idx]
    rural_true = X_original[y_true == 0, feature_idx]
    true_diff = urban_true.mean() - rural_true.mean()
    true_cohen_d = true_diff / np.sqrt((urban_true.var() + rural_true.var()) / 2)

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
    print("=" * 80)
    print("🧪 下游营养流行病学分析偏差评估")
    print("=" * 80)

    # 加载数据
    print("\n[1/3] 加载数据...")
    data = DataPipeline().load()
    model = joblib.load("./saved_models/Balanced_XGBoost.pkl")
    print(f"   ✅ 测试集: {len(data.y_test)} 样本")

    # 模拟缺失并填补
    print("\n[2/3] 模拟缺失并填补...")
    np.random.seed(42)
    mask = np.random.choice([True, False], size=len(data.y_test), p=[0.3, 0.7])
    y_masked = data.y_test.copy()
    y_masked[mask] = -1

    y_proba = model.predict_proba(data.X_test)
    y_imputed = y_masked.copy()
    y_imputed[mask] = np.argmax(y_proba[mask], axis=1)

    # 还原原始特征值
    scaler = joblib.load("./saved_models/scaler.pkl")
    X_original = scaler.inverse_transform(data.X_test)

    features = ['FatER', 'CarbER', 'ProtER', 'Fat/Carb']
    feature_labels = ['Fat Energy\nRatio', 'Carbohydrate\nEnergy Ratio',
                      'Protein Energy\nRatio', 'Fat-to-Carbohydrate\nRatio']

    print("\n[3/3] 计算效应量...")
    print("\n📊 效应量偏差分析")
    print("=" * 80)
    print(f"{'Feature':<12} {'True Diff':<12} {'Imp Diff':<12} {'Bias':<10} {'Bias%':<8} {'True d':<8} {'Imp d':<8}")
    print("-" * 80)

    results = []
    ci_true = []
    ci_imp = []

    for idx, feat in enumerate(features):
        eff = compute_effect_size(data.y_test, y_imputed, X_original, idx)
        results.append(eff)

        # Bootstrap CI
        ci_t = bootstrap_effect_size(data.y_test, X_original, idx, n_boot=500)
        ci_i = bootstrap_effect_size(y_imputed, X_original, idx, n_boot=500)
        ci_true.append(ci_t)
        ci_imp.append(ci_i)

        print(f"{feat:<12} {eff['true_diff']:>+.4f}     {eff['imp_diff']:>+.4f}     "
              f"{eff['bias']:.4f}     {eff['bias_pct']:>5.1f}%   {eff['true_d']:.3f}   {eff['imp_d']:.3f}")

    # 显著性一致性检验
    print("\n📊 显著性一致性检验")
    print("=" * 80)

    for idx, feat in enumerate(features):
        urban_true = X_original[data.y_test == 2, idx]
        rural_true = X_original[data.y_test == 0, idx]
        _, p_true = stats.ttest_ind(urban_true, rural_true)

        urban_imp = X_original[y_imputed == 2, idx]
        rural_imp = X_original[y_imputed == 0, idx]
        _, p_imp = stats.ttest_ind(urban_imp, rural_imp)

        sig_true = "***" if p_true < 0.001 else "**" if p_true < 0.01 else "*" if p_true < 0.05 else "ns"
        sig_imp = "***" if p_imp < 0.001 else "**" if p_imp < 0.01 else "*" if p_imp < 0.05 else "ns"
        consistent = "✅" if sig_true == sig_imp else "❌"

        print(f"{feat:<12} True: p={p_true:.2e} {sig_true:<3} | Imp: p={p_imp:.2e} {sig_imp:<3} {consistent}")

    # 保存 CSV
    os.makedirs("./results", exist_ok=True)
    df_results = pd.DataFrame(results)
    df_results['feature'] = features
    df_results.to_csv("./results/downstream_bias.csv", index=False)

    # ========== 生成 Figure 5 ==========
    print("\n[4/4] 生成 Figure 5...")

    fig, ax = plt.subplots(figsize=(10, 6))

    x = np.arange(len(features))
    width = 0.35

    true_d = [r['true_d'] for r in results]
    imp_d = [r['imp_d'] for r in results]

    # 误差线（95% CI）
    true_err = np.array([[true_d[i] - ci_true[i][0] for i in range(len(features))],
                         [ci_true[i][2] - true_d[i] for i in range(len(features))]])
    imp_err = np.array([[imp_d[i] - ci_imp[i][0] for i in range(len(features))],
                        [ci_imp[i][2] - imp_d[i] for i in range(len(features))]])

    bars1 = ax.bar(x - width / 2, true_d, width, label='True labels',
                   color='#2E86AB', alpha=0.85)
    bars2 = ax.bar(x + width / 2, imp_d, width, label='Inferred labels',
                   color='#F18F01', alpha=0.85)

    ax.errorbar(x - width / 2, true_d, yerr=true_err, fmt='none',
                ecolor='black', capsize=4, capthick=1.2)
    ax.errorbar(x + width / 2, imp_d, yerr=imp_err, fmt='none',
                ecolor='black', capsize=4, capthick=1.2)

    ax.set_xlabel('Macronutrient Feature', fontsize=12)
    ax.set_ylabel("Cohen's d", fontsize=12)
    ax.set_title('Downstream Epidemiological Bias Assessment', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(feature_labels, fontsize=11)
    ax.legend(loc='upper right', fontsize=11)
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.8, alpha=0.3)
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    save_figure(fig, "Fig5_downstream_bias", dpi=300)
    plt.close()

    print("\n" + "=" * 80)
    print("✅ 结果已保存至: results/downstream_bias.csv")
    print("✅ 图片已保存至: figures/Fig5_downstream_bias.tiff")

    return df_results


if __name__ == "__main__":
    df_results = run_downstream_analysis()