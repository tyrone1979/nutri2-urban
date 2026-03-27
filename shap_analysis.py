import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
import os
import warnings
warnings.filterwarnings('ignore')

import shap

# ===================== SHAP 分析类（适配最新版 shap 0.51.0）=====================
class RegionSHAPAnalyzer:
    def __init__(self, model_dir="./model", fig_dir="./figures"):
        self.model_dir = model_dir
        self.fig_dir = fig_dir
        os.makedirs(fig_dir, exist_ok=True)

    def load_xgb_model(self):
        """ 加载你训练好的 4区域 XGBoost 模型 """
        model_path = os.path.join(self.model_dir, "XGBoost.pkl")
        model = joblib.load(model_path)
        return model

    def get_ratio_features(self):
        """ 🔥 只读取【供能比特征】，不包含任何营养素总量 """
        df = pd.read_sas("./data/c12diet.sas7bdat")
        df.columns = df.columns.str.upper()
        df = df[['D3KCAL', 'D3CARBO', 'D3FAT', 'D3PROTN', 'T1']].dropna()
        df = df[(df['D3KCAL'] > 500) & (df['D3KCAL'] < 5000)]

        # 计算供能比
        df['fat_energy_ratio'] = (df['D3FAT'] * 9) / df['D3KCAL']
        df['carbo_energy_ratio'] = (df['D3CARBO'] * 4) / df['D3KCAL']
        df['protn_energy_ratio'] = (df['D3PROTN'] * 4) / df['D3KCAL']
        df['fat_carbo_ratio'] = df['D3FAT'] / (df['D3CARBO'] + 1e-6)

        # 只保留这4个特征
        X = df[['fat_energy_ratio', 'carbo_energy_ratio', 'protn_energy_ratio', 'fat_carbo_ratio']]
        return X

    def run_shap(self):
        model = self.load_xgb_model()
        X = self.get_ratio_features()

        # ===================== 🔥 最新 SHAP 0.51 正确用法 =====================
        explainer = shap.Explainer(model, X)
        shap_values = explainer(X)

        # 1. 画 SHAP 摘要图
        plt.figure(figsize=(8, 5))
        shap.summary_plot(shap_values, X, show=False)
        plt.savefig(os.path.join(self.fig_dir, "shap_region_ratios.png"), dpi=300, bbox_inches='tight')
        plt.close()

        # 2. 画特征重要性条形图
        plt.figure(figsize=(8, 5))
        shap.summary_plot(shap_values, X, plot_type="bar", show=False)
        plt.savefig(os.path.join(self.fig_dir, "shap_region_importance.png"), dpi=300, bbox_inches='tight')
        plt.close()

        print("✅ SHAP 分析完成！图片已保存到：./figures/")

# ===================== 运行 =====================
if __name__ == "__main__":
    analyzer = RegionSHAPAnalyzer()
    analyzer.run_shap()