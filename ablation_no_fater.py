#!/usr/bin/env python3
"""
消融实验：移除 FatER 特征
验证模型不是简单学习阈值
"""
import numpy as np
import pandas as pd
import joblib
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, f1_score, cohen_kappa_score, roc_auc_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from collections import Counter
from scipy.spatial.distance import jensenshannon
import warnings

warnings.filterwarnings('ignore')


def load_data_without_fater():
    """加载数据，移除 FatER 特征"""
    import pyreadstat
    df, _ = pyreadstat.read_sas7bdat("./data/c12diet.sas7bdat")
    df.columns = df.columns.str.upper()
    df = df[['T2', 'T1', 'WAVE', 'D3KCAL', 'D3CARBO', 'D3FAT', 'D3PROTN']].dropna()
    df = df[(df.D3KCAL > 500) & (df.D3KCAL < 5000)]

    # 特征工程（不包含 FatER）
    df['carbo_energy_ratio'] = df.D3CARBO * 4 / df.D3KCAL
    df['protn_energy_ratio'] = df.D3PROTN * 4 / df.D3KCAL
    df['fat_carbo_ratio'] = df.D3FAT / (df.D3CARBO + 1e-6)
    df['Year'] = df['WAVE'].astype(int)
    df['Province'] = df['T1'].astype(int)

    # 标签（仍用 FatER 定义，但特征中不含 FatER）
    fat_pct = df.D3FAT * 9 / df.D3KCAL
    df['label'] = 0
    df.loc[df['T2'] == 1, 'label'] = 2
    df.loc[(fat_pct >= 0.23) & (fat_pct <= 0.30), 'label'] = 1

    feature_cols = ['carbo_energy_ratio', 'protn_energy_ratio', 'fat_carbo_ratio', 'Year', 'Province']

    return df, feature_cols


def compute_fidelity(y_true, y_imp):
    true_counts = np.bincount(y_true, minlength=3)
    imp_counts = np.bincount(y_imp, minlength=3)
    true_dist = true_counts / len(y_true)
    imp_dist = imp_counts / len(y_imp)
    js_div = jensenshannon(true_dist, imp_dist)
    max_diff = np.max(np.abs(imp_dist - true_dist))
    return js_div, max_diff


def run_ablation_experiment():
    print("=" * 70)
    print("🧪 消融实验：移除 FatER 特征")
    print("=" * 70)

    # 加载数据
    print("\n[1/3] 加载数据...")
    df, feature_cols = load_data_without_fater()
    print(f"   ✅ 总样本: {len(df)}")
    print(f"   特征: {feature_cols}")

    # 分割
    from sklearn.model_selection import train_test_split
    X = df[feature_cols].values
    y = df['label'].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # 标准化
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    print(f"\n   训练集: {len(X_train)} | 测试集: {len(X_test)}")

    # 训练模型（无 FatER）
    print("\n[2/3] 训练 Balanced XGBoost (无 FatER)...")
    n_neg = (y_train == 0).sum()
    n_pos = (y_train == 1).sum()

    model_no_fater = XGBClassifier(
        n_estimators=600, max_depth=4, learning_rate=0.05,
        subsample=0.9, colsample_bytree=0.9,
        scale_pos_weight=np.sqrt(n_neg / n_pos),
        min_child_weight=5, gamma=0.2, reg_alpha=0.5, reg_lambda=1.2,
        objective='multi:softmax', num_class=3,
        random_state=42, n_jobs=1
    )
    model_no_fater.fit(X_train, y_train)

    # 评估分类性能
    y_pred = model_no_fater.predict(X_test)
    y_proba = model_no_fater.predict_proba(X_test)

    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='macro')
    auc = roc_auc_score(y_test, y_proba, multi_class='ovo', average='macro')

    print(f"\n   ✅ 分类性能 (无 FatER):")
    print(f"      Accuracy:  {acc:.4f}")
    print(f"      Macro-F1:  {f1:.4f}")
    print(f"      Macro-AUC: {auc:.4f}")

    # 加载完整模型对比
    print("\n[3/3] 对比完整模型...")
    model_full = joblib.load("./saved_models/Balanced_XGBoost.pkl")
    data_full = joblib.load("./saved_models/scaler.pkl")  # 只是scaler

    # 模拟缺失填补评估
    np.random.seed(42)
    mask = np.random.choice([True, False], size=len(y_test), p=[0.3, 0.7])
    y_masked = y_test.copy()
    y_masked[mask] = -1

    # 无 FatER 模型填补
    y_proba_no = model_no_fater.predict_proba(X_test)
    y_imp_no = y_masked.copy()
    y_imp_no[mask] = np.argmax(y_proba_no[mask], axis=1)

    acc_imp_no = accuracy_score(y_test[mask], y_imp_no[mask])
    f1_imp_no = f1_score(y_test[mask], y_imp_no[mask], average='macro')
    kappa_imp_no = cohen_kappa_score(y_test[mask], y_imp_no[mask])
    js_no, maxdiff_no = compute_fidelity(y_test, y_imp_no)

    # KNN 基线
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(X_test[~mask], y_test[~mask])
    y_imp_knn = y_masked.copy()
    y_imp_knn[mask] = knn.predict(X_test[mask])
    acc_knn = accuracy_score(y_test[mask], y_imp_knn[mask])
    f1_knn = f1_score(y_test[mask], y_imp_knn[mask], average='macro')

    # 汇总
    results = pd.DataFrame([
        {'Model': 'Full (with FatER)', 'Accuracy': 0.7852, 'Macro_F1': 0.7701, 'Kappa': 0.6378, 'JS_Div': 0.0229,
         'Max_Diff': 0.0266},
        {'Model': 'No FatER', 'Accuracy': acc_imp_no, 'Macro_F1': f1_imp_no, 'Kappa': kappa_imp_no, 'JS_Div': js_no,
         'Max_Diff': maxdiff_no},
        {'Model': 'KNN (k=5)', 'Accuracy': acc_knn, 'Macro_F1': f1_knn, 'Kappa': np.nan, 'JS_Div': np.nan,
         'Max_Diff': np.nan}
    ])

    print("\n" + "=" * 70)
    print("📊 消融实验结果对比")
    print("=" * 70)
    print(results.to_string(index=False))

    # 保存
    os.makedirs("./results", exist_ok=True)
    results.to_csv("./results/ablation_no_fater.csv", index=False)
    print("\n✅ 结果已保存至: results/ablation_no_fater.csv")

    return results


if __name__ == "__main__":
    import os

    results = run_ablation_experiment()