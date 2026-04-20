#!/usr/bin/env python3
"""
空间外部验证：留一省交叉验证
"""
import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler
from model import PROVINCE_SHORT
import warnings

warnings.filterwarnings('ignore')


def run_spatial_validation():
    print("=" * 70)
    print("🧪 空间外部验证 (Leave-One-Province-Out)")
    print("=" * 70)

    # 加载数据
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

    # 3分类标签
    df['label'] = 0
    df.loc[df['T2'] == 1, 'label'] = 2
    df.loc[(df['fat_energy_ratio'] >= 0.23) & (df['fat_energy_ratio'] <= 0.30), 'label'] = 1

    # 选择样本量足够的省份
    province_counts = df['Province'].value_counts()
    major_provinces = province_counts[province_counts >= 500].index.tolist()

    results = []
    feature_cols = ['fat_energy_ratio', 'carbo_energy_ratio', 'protn_energy_ratio',
                    'fat_carbo_ratio', 'Year', 'Province']

    print(f"\n📊 参与验证的省份: {len(major_provinces)} 个")

    for test_prov in major_provinces:
        prov_name = PROVINCE_SHORT.get(test_prov, f"Province{test_prov}")

        # 分割
        train_mask = df['Province'] != test_prov
        test_mask = df['Province'] == test_prov

        X_train = df.loc[train_mask, feature_cols].values
        y_train = df.loc[train_mask, 'label'].values
        X_test = df.loc[test_mask, feature_cols].values
        y_test = df.loc[test_mask, 'label'].values

        if len(X_test) < 30:
            continue

        # 标准化
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        # 训练
        model = XGBClassifier(
            n_estimators=300, max_depth=4, learning_rate=0.1,
            objective='multi:softmax', num_class=3, random_state=42, n_jobs=1
        )
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        results.append({
            'Province': prov_name,
            'n_test': len(X_test),
            'Accuracy': accuracy_score(y_test, y_pred),
            'Macro_F1': f1_score(y_test, y_pred, average='macro')
        })

        print(
            f"   {prov_name:<12}: n={len(X_test):4d}, Acc={results[-1]['Accuracy']:.3f}, F1={results[-1]['Macro_F1']:.3f}")

    # 汇总
    df_results = pd.DataFrame(results)
    print("\n" + "=" * 70)
    print("📊 空间外部验证汇总")
    print("=" * 70)
    print(f"   平均 Accuracy: {df_results['Accuracy'].mean():.3f} ± {df_results['Accuracy'].std():.3f}")
    print(f"   平均 Macro-F1: {df_results['Macro_F1'].mean():.3f} ± {df_results['Macro_F1'].std():.3f}")

    df_results.to_csv("./results/spatial_validation.csv", index=False)
    print("\n✅ 结果已保存至: results/spatial_validation.csv")


if __name__ == "__main__":
    run_spatial_validation()