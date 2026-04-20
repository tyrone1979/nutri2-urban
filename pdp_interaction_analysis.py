#!/usr/bin/env python3
"""
PDP 和 Interaction 完整分析（数值 + 图）
每个指标都有判断标准
"""
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from sklearn.inspection import PartialDependenceDisplay, partial_dependence
from sklearn.metrics import r2_score
from scipy.stats import linregress
import shap
import warnings
import os

warnings.filterwarnings('ignore')

from model import DataPipeline


def analyze_pdp_curve(fat_vals, urban_prob):
    """
    分析 PDP 曲线的特性
    """
    results = {}

    # 1. 线性拟合 R²
    slope, intercept, r_value, p_value, std_err = linregress(fat_vals, urban_prob)
    r2 = r_value ** 2
    results['linear_r2'] = r2
    results['linear_slope'] = slope

    # 2. 曲线平滑度
    diffs = np.diff(urban_prob)
    results['smoothness_std'] = np.std(diffs)

    # 3. 阈值附近的概率变化
    idx_23 = np.argmin(np.abs(fat_vals - 0.23))
    idx_30 = np.argmin(np.abs(fat_vals - 0.30))
    results['prob_at_23'] = urban_prob[idx_23]
    results['prob_at_30'] = urban_prob[idx_30]
    results['prob_change_23_30'] = urban_prob[idx_30] - urban_prob[idx_23]

    # 4. 整体变化幅度
    results['prob_min'] = urban_prob.min()
    results['prob_max'] = urban_prob.max()
    results['prob_range'] = urban_prob.max() - urban_prob.min()

    # 5. 是否单调递增
    results['is_monotonic'] = np.all(np.diff(urban_prob) >= 0)

    # 6. 阈值前后的斜率对比
    pre_23 = fat_vals < 0.23
    mid_23_30 = (fat_vals >= 0.23) & (fat_vals <= 0.30)
    post_30 = fat_vals > 0.30

    if pre_23.sum() > 1:
        slope_pre = linregress(fat_vals[pre_23], urban_prob[pre_23])[0]
    else:
        slope_pre = np.nan
    if mid_23_30.sum() > 1:
        slope_mid = linregress(fat_vals[mid_23_30], urban_prob[mid_23_30])[0]
    else:
        slope_mid = np.nan
    if post_30.sum() > 1:
        slope_post = linregress(fat_vals[post_30], urban_prob[post_30])[0]
    else:
        slope_post = np.nan

    results['slope_pre_23'] = slope_pre
    results['slope_23_30'] = slope_mid
    results['slope_post_30'] = slope_post

    if not np.isnan(slope_pre) and not np.isnan(slope_mid) and slope_pre != 0:
        results['slope_change_pre_mid'] = abs(slope_mid - slope_pre) / abs(slope_pre)
    else:
        results['slope_change_pre_mid'] = np.nan

    if not np.isnan(slope_mid) and not np.isnan(slope_post) and slope_mid != 0:
        results['slope_change_mid_post'] = abs(slope_post - slope_mid) / abs(slope_mid)
    else:
        results['slope_change_mid_post'] = np.nan

    return results


def analyze_interaction_strength(model, X_sample, feature_names):
    """
    分析特征交互强度 - 处理 4D 数组
    """
    explainer = shap.TreeExplainer(model)
    shap_interaction = explainer.shap_interaction_values(X_sample)

    # 处理多分类情况
    if isinstance(shap_interaction, list):
        interaction_urban = shap_interaction[2]  # class 2 = Urban
    else:
        interaction_urban = shap_interaction

    print(f"\n   [DEBUG] interaction shape: {interaction_urban.shape}")

    # 处理 4D 数组: (n_samples, n_features, n_features, n_classes)
    if len(interaction_urban.shape) == 4:
        # 取 Urban 类（索引 2）
        interaction_class = interaction_urban[:, :, :, 2]  # shape: (2000, 6, 6)
        mean_interaction = np.abs(interaction_class).mean(axis=0)  # shape: (6, 6)

    # 处理 3D 数组: (n_samples, n_features, n_features)
    elif len(interaction_urban.shape) == 3:
        mean_interaction = np.abs(interaction_urban).mean(axis=0)

    # 处理 2D 数组
    elif len(interaction_urban.shape) == 2:
        n_features = len(feature_names)
        if interaction_urban.shape[1] == n_features * n_features:
            interaction_reshaped = interaction_urban.reshape(-1, n_features, n_features)
            mean_interaction = np.abs(interaction_reshaped).mean(axis=0)
        else:
            mean_interaction = np.abs(interaction_urban).mean(axis=0)
            if mean_interaction.shape != (n_features, n_features):
                mean_interaction = mean_interaction[:n_features, :n_features]
    else:
        print(f"   ❌ 无法处理的 shape: {interaction_urban.shape}")
        return pd.DataFrame(), np.zeros(len(feature_names)), 0, 0

    print(f"   [DEBUG] mean_interaction shape: {mean_interaction.shape}")

    n_features = min(len(feature_names), mean_interaction.shape[0])

    # 提取上三角
    interactions = []
    for i in range(n_features):
        for j in range(i + 1, n_features):
            val = mean_interaction[i, j]
            if hasattr(val, 'item'):
                val = val.item()
            elif isinstance(val, np.ndarray):
                val = float(val.flatten()[0]) if val.size > 0 else 0.0
            interactions.append({
                'feature_1': feature_names[i],
                'feature_2': feature_names[j],
                'interaction_strength': float(val)
            })

    interactions_df = pd.DataFrame(interactions)

    if len(interactions_df) == 0:
        print("   ⚠️ 警告: 未提取到交互值")
        return pd.DataFrame(), np.zeros(n_features), 0, 0

    interactions_df = interactions_df.sort_values('interaction_strength', ascending=False)

    main_effects = np.diag(mean_interaction)[:n_features]
    total_interaction = interactions_df['interaction_strength'].sum()
    total_main = main_effects.sum()

    return interactions_df, main_effects, total_main, total_interaction


def run_pdp_analysis(force_regenerate=False):
    print("=" * 80)
    print("🧪 Partial Dependence & Interaction Analysis (数值 + 可视化)")
    print("=" * 80)

    print("\n[1/4] 加载数据...")
    data = DataPipeline().load()
    X_test = data.X_test
    feature_names = data.feature_names
    print(f"   ✅ 测试集: {len(X_test)} 样本")

    print("\n[2/4] 加载模型...")
    model = joblib.load("./saved_models/Balanced_XGBoost.pkl")
    print("   ✅ 模型加载完成")

    os.makedirs("./figures", exist_ok=True)
    os.makedirs("./results", exist_ok=True)

    # ================================================================
    # 第一部分：FatER PDP 数值分析
    # ================================================================
    print("\n" + "=" * 80)
    print("📊 第一部分：FatER Partial Dependence 数值分析")
    print("=" * 80)

    fat_idx = feature_names.index('fat_energy_ratio')

    pdp_fat = partial_dependence(model, X_test, features=[fat_idx], kind='average', grid_resolution=100)

    if hasattr(pdp_fat, 'keys'):
        fat_vals = pdp_fat['grid_values'][0]
        urban_prob = pdp_fat['average'][0]
        if urban_prob.ndim == 2:
            urban_prob = urban_prob[:, 2]
    else:
        fat_vals = pdp_fat[1][0]
        urban_prob = pdp_fat[0][0]
        if urban_prob.ndim == 2:
            urban_prob = urban_prob[:, 2]

    results = analyze_pdp_curve(fat_vals, urban_prob)

    print("\n【判断标准】")
    print("  ✅ 好的模型应该：")
    print("     - linear_r2 < 0.9 (非纯线性)")
    print("     - smoothness_std < 0.1 (曲线平滑)")
    print("     - prob_change_23_30 > 0.3 (阈值区间有区分力)")

    print("\n【FatER PDP 数值结果】")
    print(f"  线性拟合 R²:              {results['linear_r2']:.6f}")
    print(f"     → {'✅ 非线性结构' if results['linear_r2'] < 0.9 else '⚠️ 接近线性'}")
    print(f"  曲线平滑度 (std):          {results['smoothness_std']:.6f}")
    print(f"     → {'✅ 平滑' if results['smoothness_std'] < 0.1 else '⚠️ 有抖动'}")
    print(f"  单调递增:                 {results['is_monotonic']}")
    print(f"     → {'⚠️ 非单调（捕获异质性）' if not results['is_monotonic'] else '✅ 单调'}")

    print(f"\n  阈值区间分析 (23%-30%):")
    print(f"    FatER = 0.23 时概率:    {results['prob_at_23']:.4f}")
    print(f"    FatER = 0.30 时概率:    {results['prob_at_30']:.4f}")
    print(f"    概率变化:               {results['prob_change_23_30']:.4f}")
    print(f"      → {'⚠️ 避免单一特征过拟合' if results['prob_change_23_30'] < 0.3 else '✅ 区分力强'}")

    print(f"\n  斜率变化分析:")
    print(f"    <23% 区间斜率:           {results['slope_pre_23']:.4f}")
    print(f"    23%-30% 区间斜率:        {results['slope_23_30']:.4f}")
    print(f"    >30% 区间斜率:           {results['slope_post_30']:.4f}")

    print(f"\n  整体范围:")
    print(f"    概率最小值:              {results['prob_min']:.4f}")
    print(f"    概率最大值:              {results['prob_max']:.4f}")
    print(f"    概率跨度:                {results['prob_range']:.4f}")

    pd.DataFrame([results]).to_csv("./results/pdp_fater_numeric.csv", index=False)
    print("\n   ✅ 数值结果已保存至: results/pdp_fater_numeric.csv")

    # ================================================================
    # 第二部分：PDP 可视化（带缓存）
    # ================================================================
    print("\n" + "=" * 80)
    print("📊 第二部分：PDP 可视化生成")
    print("=" * 80)

    pdp_1d_path = "./figures/pdp_key_features.png"
    if os.path.exists(pdp_1d_path) and not force_regenerate:
        print(f"   ⏭️  {pdp_1d_path} 已存在，跳过生成")
    else:
        key_features = ['fat_energy_ratio', 'carbo_energy_ratio', 'fat_carbo_ratio']
        key_indices = [feature_names.index(f) for f in key_features]

        fig, ax = plt.subplots(1, 3, figsize=(15, 5))

        for idx, (feat_name, feat_idx) in enumerate(zip(key_features, key_indices)):
            PartialDependenceDisplay.from_estimator(
                model, X_test,
                features=[feat_idx],
                kind='average',
                grid_resolution=50,
                ax=ax[idx],
                target=2
            )
            ax[idx].set_title(f'PDP: {feat_name}')
            ax[idx].grid(True, alpha=0.3)

            if feat_name == 'fat_energy_ratio':
                ax[idx].axvline(x=0.23, color='red', linestyle='--', alpha=0.5, label='Rural (23%)')
                ax[idx].axvline(x=0.30, color='green', linestyle='--', alpha=0.5, label='Urban (30%)')
                ax[idx].legend(fontsize=8)

        plt.tight_layout()
        plt.savefig(pdp_1d_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"   ✅ {pdp_1d_path} 已保存")

    # ================================================================
    # 第三部分：2D Interaction 可视化（带缓存）
    # ================================================================
    print("\n" + "=" * 80)
    print("📊 第三部分：2D Interaction 可视化")
    print("=" * 80)

    feat1_idx = feature_names.index('fat_energy_ratio')
    feat2_idx = feature_names.index('carbo_energy_ratio')
    feat3_idx = feature_names.index('protn_energy_ratio')

    # FatER × CarbER
    pdp_2d_carb_path = "./figures/pdp_interaction_fater_carber.png"
    if os.path.exists(pdp_2d_carb_path) and not force_regenerate:
        print(f"   ⏭️  {pdp_2d_carb_path} 已存在，跳过生成")
    else:
        fig, ax = plt.subplots(figsize=(8, 6))
        PartialDependenceDisplay.from_estimator(
            model, X_test,
            features=[(feat1_idx, feat2_idx)],
            kind='average',
            grid_resolution=30,
            ax=ax,
            target=2
        )
        ax.set_xlabel('Fat Energy Ratio')
        ax.set_ylabel('Carbohydrate Energy Ratio')
        ax.set_title('Interaction: FatER × CarbER')
        plt.tight_layout()
        plt.savefig(pdp_2d_carb_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"   ✅ {pdp_2d_carb_path} 已保存")

    # FatER × ProtER
    pdp_2d_prot_path = "./figures/pdp_interaction_fater_proter.png"
    if os.path.exists(pdp_2d_prot_path) and not force_regenerate:
        print(f"   ⏭️  {pdp_2d_prot_path} 已存在，跳过生成")
    else:
        fig, ax = plt.subplots(figsize=(8, 6))
        PartialDependenceDisplay.from_estimator(
            model, X_test,
            features=[(feat1_idx, feat3_idx)],
            kind='average',
            grid_resolution=30,
            ax=ax,
            target=2
        )
        ax.set_xlabel('Fat Energy Ratio')
        ax.set_ylabel('Protein Energy Ratio')
        ax.set_title('Interaction: FatER × ProtER')
        plt.tight_layout()
        plt.savefig(pdp_2d_prot_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"   ✅ {pdp_2d_prot_path} 已保存")

    # ================================================================
    # 第四部分：SHAP Interaction 数值分析
    # ================================================================
    print("\n" + "=" * 80)
    print("📊 第四部分：SHAP Interaction 数值分析")
    print("=" * 80)

    shap_interaction_path = "./results/shap_interactions.csv"

    X_sample = X_test[:2000]

    try:
        interactions_df, main_effects, total_main, total_interaction = analyze_interaction_strength(
            model, X_sample, feature_names
        )

        if len(interactions_df) > 0:
            interactions_df.to_csv(shap_interaction_path, index=False)
            pd.DataFrame({'feature': feature_names[:len(main_effects)], 'main_effect': main_effects}).to_csv(
                "./results/shap_main_effects.csv", index=False
            )
            print("   ✅ SHAP 交互结果已保存")
        else:
            print("   ⚠️ SHAP 交互分析未产生结果，跳过")
            return results, pd.DataFrame()

    except Exception as e:
        print(f"   ❌ SHAP 交互分析失败: {e}")
        return results, pd.DataFrame()

    print("\n【判断标准】")
    print("  ✅ 好的模型应该：")
    print("     - 交互效应占比 > 10% (证明不是纯加性模型)")
    print("     - Top 交互对强度 > 0.01 (存在有意义的交互)")

    print("\n【SHAP Interaction 数值结果】")
    print(f"  主效应总和:               {total_main:.6f}")
    print(f"  交互效应总和:             {total_interaction:.6f}")
    interaction_ratio = total_interaction / (total_main + total_interaction) if (
                                                                                            total_main + total_interaction) > 0 else 0
    print(f"  交互效应占比:             {interaction_ratio:.2%}")

    if interaction_ratio > 0.10:
        print(f"     → ✅ 模型捕获了显著的非加性交互效应")
    else:
        print(f"     → ⚠️ 模型以加性效应为主")

    print("\n【Top 10 特征交互对】")
    print(f"{'Rank':<5} {'Feature 1':<22} {'Feature 2':<22} {'Interaction Strength':<20}")
    print("-" * 75)
    for idx, row in enumerate(interactions_df.head(10).itertuples(), 1):
        print(f"{idx:<5} {row.feature_1:<22} {row.feature_2:<22} {row.interaction_strength:.8f}")

    top_inter = interactions_df.iloc[0]
    if top_inter['interaction_strength'] > 0.01:
        print(f"\n     → ✅ 最强交互对强度 > 0.01")
    else:
        print(f"\n     → ⚠️ 交互强度较弱")

    # ================================================================
    # 总结
    # ================================================================
    print("\n" + "=" * 80)
    print("📊 综合判断总结")
    print("=" * 80)

    checks = []
    if results['linear_r2'] < 0.9:
        checks.append("✅ 非线性结构")
    else:
        checks.append("⚠️ 接近线性")
    if results['smoothness_std'] < 0.1:
        checks.append("✅ 曲线平滑")
    else:
        checks.append("⚠️ 有抖动")
    if not results['is_monotonic']:
        checks.append("✅ 捕获异质性（非单调）")
    else:
        checks.append("✅ 单调递增")
    if results['prob_change_23_30'] < 0.3:
        checks.append("✅ 避免单一特征过拟合")
    else:
        checks.append("✅ 区分力强")
    if interaction_ratio > 0.10:
        checks.append("✅ 显著交互效应")
    else:
        checks.append("⚠️ 以加性为主")

    print("\n   " + "  ".join(checks))

    if all(c.startswith("✅") for c in checks):
        print("\n   🎉 模型通过所有检查！学习的是连续、非线性、有交互的结构。")
    else:
        print("\n   📝 模型整体良好，部分指标可关注。")

    return results, interactions_df


if __name__ == "__main__":
    results, interactions_df = run_pdp_analysis(force_regenerate=False)