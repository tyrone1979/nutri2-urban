import shap
import matplotlib.pyplot as plt
import seaborn as sns
# 其他导入
import pandas as pd
import numpy as np
import os
FIGURE_DIR = "./figures"

# ===================== SHAP Analyzer（3分类修复版）=====================
class SHAPAnalyzer:
    def __init__(self, model, X_test, feature_names):
        self.model = model
        self.X_test = pd.DataFrame(X_test, columns=feature_names)
        self.explainer = None
        self.shap_values = None  # 3分类会是 [n_sample, n_feat, 3]

    def run(self):
        print("\n[4/4] 生成 SHAP 特征重要性图...")
        os.makedirs(FIGURE_DIR, exist_ok=True)

        self.explainer = shap.TreeExplainer(self.model)
        self.shap_values = self.explainer.shap_values(self.X_test)

        # ===================== 打印 SHAP 值统计 =====================
        print("\n" + "=" * 70)
        print("📊 SHAP Value Analysis")
        print("=" * 70)

        # 处理多分类 SHAP 值
        if isinstance(self.shap_values, list) or len(self.shap_values.shape) == 3:
            # 多分类情况
            n_classes = self.shap_values.shape[2] if len(self.shap_values.shape) == 3 else len(self.shap_values)
            print(f"\n多分类检测: {n_classes} 个类别 (0=Rural, 1=Transitional, 2=Urban)")

            for class_idx in range(n_classes):
                # 获取当前类别的 SHAP 值
                if len(self.shap_values.shape) == 3:
                    sv_class = self.shap_values[:, :, class_idx]
                else:
                    sv_class = self.shap_values[class_idx]

                # 计算平均绝对 SHAP 值（特征重要性）
                mean_abs_shap = np.abs(sv_class).mean(axis=0)

                print(f"\n{'=' * 50}")
                print(
                    f"Class {class_idx} {'(Rural)' if class_idx == 0 else '(Transitional)' if class_idx == 1 else '(Urban)'}")
                print(f"{'=' * 50}")
                print(f"{'Feature':<25} {'Mean |SHAP|':<15} {'Direction':<15}")
                print(f"{'-' * 50}")

                for name, value in zip(self.X_test.columns, mean_abs_shap):
                    # 计算平均 SHAP 值（带符号）来判断方向
                    mean_shap = sv_class.mean(axis=0)[list(self.X_test.columns).index(name)]
                    direction = "Positive → Urban" if mean_shap > 0 else "Negative → Rural"
                    print(f"{name:<25} {value:.6f}        {direction}")

                # 打印总体统计
                print(f"\n总平均 |SHAP|: {mean_abs_shap.mean():.6f}")
                print(f"最重要的特征: {self.X_test.columns[np.argmax(mean_abs_shap)]} ({mean_abs_shap.max():.6f})")

            # 🔥 取第 2 类（城市）的 SHAP 值画图（保持原有逻辑）
            sv = self.shap_values[:, :, 2] if len(self.shap_values.shape) == 3 else self.shap_values[2]
            print(f"\n✅ 使用 Class 2 (Urban) 的 SHAP 值生成可视化")

        else:
            # 二分类情况
            mean_abs_shap = np.abs(self.shap_values).mean(axis=0)
            print(f"\n{'Feature':<25} {'Mean |SHAP|':<15} {'Direction':<15}")
            print(f"{'-' * 50}")
            for name, value in zip(self.X_test.columns, mean_abs_shap):
                mean_shap = self.shap_values.mean(axis=0)[list(self.X_test.columns).index(name)]
                direction = "Positive → Urban" if mean_shap > 0 else "Negative → Rural"
                print(f"{name:<25} {value:.6f}        {direction}")
            sv = self.shap_values

        print("=" * 70)

        # 1. 标准SHAP summary图
        plt.figure(figsize=(10, 6))
        shap.summary_plot(sv, self.X_test, show=False)
        plt.tight_layout()
        plt.savefig(f"{FIGURE_DIR}/FigS1_shap_summary.tiff", dpi=300, bbox_inches='tight',
                format='tiff', pil_kwargs={'compression': 'tiff_lzw'})
        plt.close()
        print("\n   ✅ shap_summary.tiff 已保存")

        return self

if __name__=="__main__":
    from model import DataPipeline
    import joblib

    data = DataPipeline().load()
    model = joblib.load("./saved_models/Balanced_XGBoost.pkl")

    SHAPAnalyzer(
        model,
        data.X_test,
        data.feature_names
    ).run()