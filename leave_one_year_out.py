#!/usr/bin/env python3
"""
Leave-One-Year-Out Validation
比 temporal split 更细粒度的时间泛化验证
"""
import numpy as np
import pandas as pd
import joblib
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, f1_score, cohen_kappa_score
from sklearn.preprocessing import StandardScaler
from collections import Counter
import warnings

warnings.filterwarnings('ignore')


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
    print("=" * 70)
    print("🧪 Leave-One-Year-Out Validation")
    print("=" * 70)

    print("\n[1/2] 加载数据...")
    df = load_full_data()
    years = sorted(df['Year'].unique())
    print(f"   ✅ 总样本: {len(df)}")
    print(f"   年份: {years}")

    feature_cols = ['fat_energy_ratio', 'carbo_energy_ratio', 'protn_energy_ratio',
                    'fat_carbo_ratio', 'Province']

    results = []

    print("\n[2/2] 运行 Leave-One-Year-Out...")

    for test_year in years:
        # 分割
        train_mask = df['Year'] != test_year
        test_mask = df['Year'] == test_year

        X_train = df.loc[train_mask, feature_cols].values
        y_train = df.loc[train_mask, 'label'].values
        X_test = df.loc[test_mask, feature_cols].values
        y_test = df.loc[test_mask, 'label'].values

        # 标准化
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        # 训练
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

        # 评估
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

    # 汇总
    df_results = pd.DataFrame(results)

    print("\n" + "=" * 70)
    print("📊 Leave-One-Year-Out 汇总")
    print("=" * 70)
    print(
        f"   平均 Accuracy: {df_results['Imputation_Accuracy'].mean():.4f} ± {df_results['Imputation_Accuracy'].std():.4f}")
    print(f"   平均 Macro-F1: {df_results['Imputation_F1'].mean():.4f} ± {df_results['Imputation_F1'].std():.4f}")
    print(f"   平均 Kappa:    {df_results['Imputation_Kappa'].mean():.4f} ± {df_results['Imputation_Kappa'].std():.4f}")

    # 保存
    os.makedirs("./results", exist_ok=True)
    df_results.to_csv("./results/leave_one_year_out.csv", index=False)
    print("\n✅ 结果已保存至: results/leave_one_year_out.csv")

    return df_results


if __name__ == "__main__":
    import os

    df_results = run_leave_one_year_out()