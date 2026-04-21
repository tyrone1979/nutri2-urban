#!/usr/bin/env python3
"""
生成校准曲线图（Figure 3）- 完整版（带直方图和置信区间）
"""
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss
import warnings
import os

warnings.filterwarnings('ignore')
from model import DataPipeline


def expected_calibration_error(y_true, y_prob, n_bins=10):
    """计算 Expected Calibration Error (ECE)"""
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]

    ece = 0.0
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        in_bin = (y_prob >= bin_lower) & (y_prob < bin_upper)
        if in_bin.sum() > 0:
            prop_in_bin = in_bin.sum() / len(y_prob)
            avg_prob = y_prob[in_bin].mean()
            avg_true = y_true[in_bin].mean()
            ece += prop_in_bin * np.abs(avg_prob - avg_true)
    return ece


def bootstrap_calibration_curve(y_true, y_prob, n_boot=1000, n_bins=10):
    """Bootstrap 校准曲线生成置信区间"""
    n_samples = len(y_true)
    curves = []

    for _ in range(n_boot):
        idx = np.random.choice(n_samples, n_samples, replace=True)
        y_true_boot = y_true[idx]
        y_prob_boot = y_prob[idx]

        prob_true, prob_pred = calibration_curve(
            y_true_boot, y_prob_boot, n_bins=n_bins, strategy='uniform'
        )
        curves.append((prob_pred, prob_true))

    return curves


def save_figure(fig, filename_base, dpi=300):
    """保存图片为 TIFF 格式"""

    fig.savefig(f"./figures/{filename_base}.tiff", dpi=dpi, bbox_inches='tight',
                format='tiff', pil_kwargs={'compression': 'tiff_lzw'})
    print(f"   ✅{filename_base}.tiff 已保存")


def run_calibration_analysis():
    print("=" * 80)
    print("🧪 模型校准分析 (Calibration Analysis)")
    print("=" * 80)

    print("\n[1/3] 加载数据...")
    data = DataPipeline().load()
    X_test = data.X_test
    y_test = data.y_test
    print(f"   ✅ 测试集: {len(X_test)} 样本")

    print("\n[2/3] 加载模型...")
    model = joblib.load("./saved_models/Balanced_XGBoost.pkl")
    print("   ✅ 模型加载完成")

    y_proba = model.predict_proba(X_test)

    os.makedirs("./figures", exist_ok=True)
    os.makedirs("./results", exist_ok=True)

    class_names = ['Rural', 'Transitional', 'Urban']
    results = {}

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    for class_idx, (ax, class_name) in enumerate(zip(axes, class_names)):
        y_true_bin = (y_test == class_idx).astype(int)
        y_prob_class = y_proba[:, class_idx]

        # 主校准曲线
        prob_true, prob_pred = calibration_curve(
            y_true_bin, y_prob_class, n_bins=10, strategy='uniform'
        )

        # Bootstrap 置信区间
        boot_curves = bootstrap_calibration_curve(y_true_bin, y_prob_class, n_boot=500)
        all_prob_true = []
        all_prob_pred = []
        for pred, true in boot_curves:
            all_prob_true.append(true)
            all_prob_pred.append(pred)

        # 计算每个预测概率点的置信区间
        unique_preds = np.unique(np.concatenate([c[0] for c in boot_curves]))
        ci_lower = []
        ci_upper = []
        for up in unique_preds:
            nearby_true = []
            for pred, true in boot_curves:
                idx = np.argmin(np.abs(pred - up))
                nearby_true.append(true[idx])
            nearby_true = np.array(nearby_true)
            ci_lower.append(np.percentile(nearby_true, 2.5))
            ci_upper.append(np.percentile(nearby_true, 97.5))

        # 绘制置信区间阴影
        ax.fill_between(unique_preds, ci_lower, ci_upper, alpha=0.2, color='#2E86AB')

        # 绘制主校准曲线
        ax.plot(prob_pred, prob_true, 'o-', linewidth=2, markersize=6,
                color='#2E86AB', label='Calibration curve')

        # 绘制完美校准线
        ax.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Perfect calibration')

        # 添加直方图（预测概率分布）
        ax_hist = ax.twinx()
        ax_hist.hist(y_prob_class, bins=20, alpha=0.3, color='gray', edgecolor='none')
        ax_hist.set_ylim(0, len(y_prob_class) // 4)
        ax_hist.set_ylabel('Count', fontsize=10)

        # 计算指标
        brier = brier_score_loss(y_true_bin, y_prob_class)
        ece = expected_calibration_error(y_true_bin, y_prob_class)

        results[class_name] = {
            'Brier_Score': brier,
            'ECE': ece,
            'Mean_Predicted': y_prob_class.mean(),
            'True_Prevalence': y_true_bin.mean()
        }

        ax.set_xlabel('Predicted Probability', fontsize=11)
        ax.set_ylabel('Observed Frequency', fontsize=11)
        ax.set_title(f'{class_name}\nBrier={brier:.4f}, ECE={ece:.4f}', fontsize=12)
        ax.legend(loc='upper left', fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)

    plt.tight_layout()
    save_figure(fig, "Fig3_calibration_curves", dpi=300)
    plt.close()

    # 打印结果
    print("\n" + "=" * 80)
    print("📊 校准分析结果")
    print("=" * 80)
    df_results = pd.DataFrame(results).T
    print(df_results.to_string())

    df_results.to_csv("./results/calibration_results.csv")
    print("\n✅ 结果已保存至: results/calibration_results.csv")

    return results


if __name__ == "__main__":
    results = run_calibration_analysis()