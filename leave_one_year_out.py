#!/usr/bin/env python3
"""
Leave-One-Year-Out Validation + 生成 Figure 4
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, f1_score, cohen_kappa_score
from sklearn.preprocessing import StandardScaler
import warnings
import os

warnings.filterwarnings('ignore')


def save_figure(fig, filename_base, dpi=300):
    """保存图片为 PNG 和 TIFF 格式"""
    os.makedirs("./figures", exist_ok=True)
    fig.savefig(f"./figures/{filename_base}.png", dpi=dpi, bbox_inches='tight')
    fig.savefig(f"./figures/{filename_base}.tiff", dpi=dpi, bbox_inches='tight',
                format='tiff', pil_kwargs={'compression': 'tiff_lzw'})
    print(f"   ✅ {filename_base}.png 和 {filename_base}.tiff 已保存")


def load_full_data():
    """加载完整数据（未分割）"""
    import pyreadstat
    df, _ = pyreadstat.read_sas7bdat("./data/c12diet.sas7bdat")
    df.columns = df.columns.str.upper()
    df = df[['T2', 'T1', 'WAVE', 'D3KCAL', 'D3CARBO', 'D3FAT', 'D3PROTN']].dropna()
    df = df[(df.D3KCAL > 500) & (df.D3KCAL < 5000)]

    # 特征工程
    df['fat_energy_ratio'] = df.D3FAT * 9 / df.D3KCAL
    df['carbo_energy_ratio'] = df.D3CARBO * 4 / df.D3KCAL
    df['protn_energy_ratio'] = df.D3PROTN * 4 / df.D3KCAL
    df['fat_carbo_ratio'] = df.D3FAT / (df.D3CARBO + 1e-6)
    df['Year'] = df['WAVE'].astype(int)
    df['Province'] = df['T1'].astype(int)

    # 标签
    df['label'] = 0
    df.loc[df['T2'] == 1, 'label'] = 2
    df.loc[(df['fat_energy_ratio'] >= 0.23) & (df['fat_energy_ratio'] <= 0.30), 'label'] = 1

    return df


def run_leave_one_year_out():
    print("=" * 80)
    print("🧪 Leave-One-Year-Out Validation")
    print("=" * 80)

    print("\n[1/2] 加载数据...")
    df = load_full_data()
    years = sorted(df['Year'].unique())
    print(f"   ✅ 总样本: {len(df)}")
    print(f"   年份: {years}")

    feature_cols = ['fat_energy_ratio', 'carbo_energy_ratio', 'protn_energy_ratio',
                    'fat_carbo_ratio', 'Year', 'Province']

    results = []

    print("\n[2/2] 运行 Leave-One-Year-Out...")

    for test_year in years:
        train_mask = df['Year'] != test_year
        test_mask = df['Year'] == test_year

        X_train = df.loc[train_mask, feature_cols].values
        y_train = df.loc[train_mask, 'label'].values
        X_test = df.loc[test_mask, feature_cols].values
        y_test = df.loc[test_mask, 'label'].values

        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        n_neg = (y_train == 0).sum()
        n_pos = (y_train == 1).sum()

        model = XGBClassifier(
            n_estimators=300, max_depth=4, learning_rate=0.1,
            scale_pos_weight=np.sqrt(n_neg / n_pos) if n_pos > 0 else 1,
            min_child_weight=5, gamma=0.2, reg_alpha=0.5, reg_lambda=1.2,
            objective='multi:softmax', num_class=3,
            random_state=42, n_jobs=1
        )
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)

        # 模拟填补评估
        np.random.seed(42)
        mask = np.random.choice([True, False], size=len(y_test), p=[0.3, 0.7])
        y_masked = y_test.copy()
        y_masked[mask] = -1

        y_proba = model.predict_proba(X_test)
        y_imp = y_masked.copy()
        y_imp[mask] = np.argmax(y_proba[mask], axis=1)

        acc_imp = accuracy_score(y_test[mask], y_imp[mask])
        f1_imp = f1_score(y_test[mask], y_imp[mask], average='macro')
        kappa_imp = cohen_kappa_score(y_test[mask], y_imp[mask])

        results.append({
            'Test_Year': test_year,
            'N_Train': len(X_train),
            'N_Test': len(X_test),
            'Imputation_Accuracy': acc_imp,
            'Imputation_F1': f1_imp,
            'Imputation_Kappa': kappa_imp
        })

        print(f"   {test_year}: N={len(X_test):5d}, Acc={acc_imp:.4f}, F1={f1_imp:.4f}, Kappa={kappa_imp:.4f}")

    df_results = pd.DataFrame(results)

    # 保存 CSV
    os.makedirs("./results", exist_ok=True)
    df_results.to_csv("./results/leave_one_year_out.csv", index=False)

    # ========== 生成 Figure 4 ==========
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    years_str = df_results['Test_Year'].astype(str).values
    acc = df_results['Imputation_Accuracy'].values
    f1 = df_results['Imputation_F1'].values

    # (a) Accuracy
    ax = axes[0]
    ax.bar(years_str, acc, color='#2E86AB', alpha=0.8, width=0.6)
    ax.axhline(y=acc.mean(), color='#A23B72', linestyle='--', linewidth=1.5,
               label=f'Mean = {acc.mean():.3f}')
    ax.set_xlabel('Survey Year', fontsize=12)
    ax.set_ylabel('Accuracy', fontsize=12)
    ax.set_title('(a) Recovery Accuracy by Held-Out Year', fontsize=14, fontweight='bold')
    ax.set_ylim(0.65, 0.85)
    ax.legend(fontsize=10, loc='lower right')
    ax.grid(True, alpha=0.3, axis='y')

    # (b) Macro-F1
    ax = axes[1]
    ax.bar(years_str, f1, color='#F18F01', alpha=0.8, width=0.6)
    ax.axhline(y=f1.mean(), color='#A23B72', linestyle='--', linewidth=1.5,
               label=f'Mean = {f1.mean():.3f}')
    ax.set_xlabel('Survey Year', fontsize=12)
    ax.set_ylabel('Macro-F1', fontsize=12)
    ax.set_title('(b) Macro-F1 by Held-Out Year', fontsize=14, fontweight='bold')
    ax.set_ylim(0.60, 0.85)
    ax.legend(fontsize=10, loc='lower right')
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    save_figure(fig, "Fig4_temporal_validation", dpi=300)
    plt.close()

    # 打印汇总
    print("\n" + "=" * 80)
    print("📊 Leave-One-Year-Out 汇总")
    print("=" * 80)
    print(f"   平均 Accuracy: {acc.mean():.4f} ± {acc.std():.4f}")
    print(f"   平均 Macro-F1: {f1.mean():.4f} ± {f1.std():.4f}")
    print(f"   平均 Kappa:    {df_results['Imputation_Kappa'].mean():.4f} ± {df_results['Imputation_Kappa'].std():.4f}")

    print("\n✅ 结果已保存至: results/leave_one_year_out.csv")
    print("✅ 图片已保存至: figures/Fig4_temporal_validation.tiff")

    return df_results


if __name__ == "__main__":
    df_results = run_leave_one_year_out()