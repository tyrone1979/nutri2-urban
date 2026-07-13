#!/usr/bin/env python3
"""
时间外部验证：用1991-2006训练，预测2009-2011
"""
import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler
import warnings

warnings.filterwarnings('ignore')

from data_pipeline import engineer_columns
from features import FEATURE_COLS


def run_temporal_validation():
    print("=" * 70)
    print("🧪 时间外部验证 (Temporal Validation)")
    print("=" * 70)

    # 加载原始数据（未标准化）
    import pyreadstat
    df, _ = pyreadstat.read_sas7bdat("./data/c12diet.sas7bdat")
    df.columns = df.columns.str.upper()
    df = df[['T2', 'T1', 'WAVE', 'D3KCAL', 'D3CARBO', 'D3FAT', 'D3PROTN']].dropna()
    df = df[(df.D3KCAL > 500) & (df.D3KCAL < 5000)]
    df = engineer_columns(df)

    train_mask = df['Year'] <= 2006
    test_mask = df['Year'] >= 2009

    cols = FEATURE_COLS
    X_train = df.loc[train_mask, cols].values
    y_train = df.loc[train_mask, 'label'].values
    X_test = df.loc[test_mask, cols].values
    y_test = df.loc[test_mask, 'label'].values

    # 标准化
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    print(f"\n📊 数据分割:")
    print(f"   训练集 (1991-2006): {len(X_train)} 样本")
    print(f"   测试集 (2009-2011): {len(X_test)} 样本")

    # 训练模型
    model = XGBClassifier(
        n_estimators=600, max_depth=4, learning_rate=0.05,
        subsample=0.9, colsample_bytree=0.9, min_child_weight=5,
        gamma=0.2, reg_alpha=0.5, reg_lambda=1.2,
        objective='multi:softmax', num_class=3, random_state=42, n_jobs=1
    )

    print("\n⏳ 训练中...")
    model.fit(X_train, y_train)

    # 评估
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='macro')

    print("\n" + "=" * 70)
    print("📊 时间外部验证结果")
    print("=" * 70)
    print(f"   Accuracy:  {acc:.4f}")
    print(f"   Macro-F1:  {f1:.4f}")

    print("\n📊 与原始测试集性能对比:")
    print("   (重跑 main.py 后更新 baseline 对比数值)")

    # 保存
    result = {'temporal_acc': acc, 'temporal_f1': f1,
              'n_train': len(X_train), 'n_test': len(X_test)}
    pd.DataFrame([result]).to_csv("./results/temporal_validation.csv", index=False)
    print("\n✅ 结果已保存至: results/temporal_validation.csv")


if __name__ == "__main__":
    run_temporal_validation()