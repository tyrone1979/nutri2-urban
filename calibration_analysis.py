#!/usr/bin/env python3
"""
模型校准分析：Brier Score + Reliability Curve
"""
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from sklearn.calibration import calibration_curve, CalibrationDisplay
from sklearn.metrics import brier_score_loss
import warnings

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


def run_calibration_analysis():
    print("=" * 70)
    print("🧪 模型校准分析 (Calibration Analysis)")
    print("=" * 70)

    print("\n[1/3] 加载数据...")
    data = DataPipeline().load()
    X_test = data.X_test
    y_test = data.y_test
    print(f"   ✅ 测试集: {len(X_test)} 样本")

    print("\n[2/3] 加载模型...")
    model = joblib.load("./saved_models/Balanced_XGBoost.pkl")
    print("   ✅ 模型加载完成")

    # 预测概率
    y_proba = model.predict_proba(X_test)

    results = {}

    print("\n[3/3] 计算校准指标...")

    os.makedirs("./figures", exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    class_names = ['Rural', 'Transitional', 'Urban']

    for class_idx, class_name in enumerate(class_names):
        y_true_bin = (y_test == class_idx).astype(int)
        y_prob_class = y_proba[:, class_idx]

        # Brier Score
        brier = brier_score_loss(y_true_bin, y_prob_class)

        # ECE
        ece = expected_calibration_error(y_true_bin, y_prob_class, n_bins=10)

        # 校准曲线数据
        prob_true, prob_pred = calibration_curve(y_true_bin, y_prob_class, n_bins=10)

        results[class_name] = {
            'Brier_Score': brier,
            'ECE': ece,
            'Mean_Predicted': y_prob_class.mean(),
            'True_Prevalence': y_true_bin.mean()
        }

        # 绘图
        ax = axes[class_idx]
        CalibrationDisplay(prob_true, prob_pred, y_prob_class).plot(ax=ax)
        ax.set_title(f'{class_name}\nBrier={brier:.4f}, ECE={ece:.4f}')
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("./figures/calibration_curves.png", dpi=300, bbox_inches='tight')
    plt.close()
    print("   ✅ calibration_curves.png 已保存")

    # 打印结果
    print("\n" + "=" * 70)
    print("📊 校准分析结果")
    print("=" * 70)

    df_results = pd.DataFrame(results).T
    print(df_results.to_string())

    # 解读
    print("\n📊 解读:")
    for class_name in class_names:
        brier = results[class_name]['Brier_Score']
        ece = results[class_name]['ECE']
        if brier < 0.15 and ece < 0.05:
            print(f"   {class_name:<12}: ✅ 校准良好 (Brier<0.15, ECE<0.05)")
        elif brier < 0.20 and ece < 0.10:
            print(f"   {class_name:<12}: ⚠️ 校准可接受")
        else:
            print(f"   {class_name:<12}: ❌ 校准需改进")

    # 保存
    os.makedirs("./results", exist_ok=True)
    df_results.to_csv("./results/calibration_results.csv")
    print("\n✅ 结果已保存至: results/calibration_results.csv")

    return results


if __name__ == "__main__":
    import os

    results = run_calibration_analysis()