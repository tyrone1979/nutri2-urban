import shap
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# ====================== SHAP 分析封装类 ======================
class SHAPAnalyzer:
    """
    用于城乡膳食结构分类的 SHAP 可解释性分析
    自动计算 SHAP 值 → 绘图 → 保存 → 输出解释
    """

    def __init__(self, model, X_test, feature_names=None):
        """
        :param model: 训练好的模型（XGBoost 最佳）
        :param X_test: 测试集（numpy array）
        :param feature_names: 特征名列表
        """
        self.model = model
        self.X_test = X_test
        self.feature_names = feature_names or [
            "Fat Energy Ratio",
            "Carbo Energy Ratio",
            "Protein Energy Ratio",
            "Fat/Carbo Ratio"
        ]

        # 转 DataFrame 便于 SHAP 绘图
        self.X_test_df = pd.DataFrame(
            self.X_test,
            columns=self.feature_names
        )

        # SHAP 解释器
        self.explainer = shap.TreeExplainer(self.model)
        self.shap_values = None

    def compute_shap_values(self):
        """计算 SHAP 值"""
        self.shap_values = self.explainer.shap_values(self.X_test_df)
        return self

    def plot_summary(self, save_path="./figures/shap_summary.png"):
        """SHAP 总览图（特征重要性）"""
        plt.figure(figsize=(8, 5))
        shap.summary_plot(
            self.shap_values,
            self.X_test_df,
            show=False
        )
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"✅ SHAP 总览图已保存：{save_path}")

    def plot_force_single(self, sample_idx=0, save_path="./figures/shap_force.png"):
        """单样本 force plot"""
        shap.force_plot(
            self.explainer.expected_value,
            self.shap_values[sample_idx, :],
            self.X_test_df.iloc[sample_idx, :],
            matplotlib=True,
            show=False
        )
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"✅ SHAP 单样本图已保存：{save_path}")

    def plot_dependence(self, feature_idx=0, save_path="./figures/shap_dependence.png"):
        """SHAP 依赖图（特征影响趋势）"""
        shap.dependence_plot(
            feature_idx,
            self.shap_values,
            self.X_test_df,
            show=False
        )
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"✅ SHAP 依赖图已保存：{save_path}")

    def run_all(self):
        """一键运行所有 SHAP 分析"""
        self.compute_shap_values()
        self.plot_summary()
        self.plot_force_single()
        self.plot_dependence(feature_idx=0)
        print("\n🎉 SHAP 全部分析完成！")

# ====================== 调用 SHAP 分析 ======================
if __name__ == "__main__":
    # 1. 训练完你的模型（xgb）
    # ...（你的训练代码）

    # 2. 初始化 SHAP 分析器
    shap_analyzer = SHAPAnalyzer(
        model=xgb,                  # 你的模型
        X_test=X_test,              # 测试集
        feature_names=None          # 自动使用标准膳食特征名
    )

    # 3. 一键运行所有分析 + 出图
    shap_analyzer.run_all()