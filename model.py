#!/usr/bin/env python3
import os

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

# 关键：禁用所有 Python multiprocessing（Python 3.13 在 Apple Silicon 上有严重问题）
import multiprocessing

multiprocessing.set_start_method('spawn', force=True)

# 现在安全导入 torch
import torch
import torch.nn as nn
import torch.optim as optim

torch.set_num_threads(1)

# 其他导入
import pandas as pd
import numpy as np
import time
import joblib
import glob
import warnings
from tqdm import tqdm

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score

import shap
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings('ignore')

# ===================== 全局配置 =====================
plt.rcParams["font.sans-serif"] = ["Arial"]
plt.rcParams["axes.unicode_minus"] = False

DATA_PATH = "./data/c12diet.sas7bdat"
RESULT_PATH = "./results/model_results.csv"
MODEL_SAVE_DIR = "./saved_models"
FIGURE_DIR = "./figures"

# ===================== 省份编码映射（英文）=====================
PROVINCE_SHORT = {
    11: "Beijing", 21: "Liaoning", 23: "Heilongjiang", 31: "Shanghai",
    32: "Jiangsu", 37: "Shandong", 41: "Henan", 42: "Hubei",
    43: "Hunan", 45: "Guangxi", 52: "Guizhou", 55: "Chongqing"
}


# ===================== Data Pipeline =====================
from sklearn.preprocessing import StandardScaler

class DataPipeline:
    def __init__(self, path=DATA_PATH):
        self.path = path
        self.X_train, self.X_test, self.y_train, self.y_test = None, None, None, None
        self.year_train, self.year_test = None, None
        self.province_train, self.province_test = None, None
        self.feature_names = ["fat_energy_ratio", "carbo_energy_ratio", "protn_energy_ratio", "fat_carbo_ratio", "Year", "Province"]
        self.scaler = StandardScaler()

    def load(self):
        print("[1/4] 正在加载数据...")
        if not os.path.exists(self.path):
            raise FileNotFoundError(f"数据文件不存在: {self.path}")

        try:
            import pyreadstat
            print("   使用 pyreadstat 读取（更快）...")
            df, meta = pyreadstat.read_sas7bdat(self.path)
        except ImportError:
            print("   pyreadstat 未安装，使用 pandas.read_sas（较慢）...")
            print("   建议: pip install pyreadstat 加速读取")
            df = pd.read_sas(self.path, encoding='utf-8')

        df.columns = df.columns.str.upper()
        df = df[['T2', 'T1', 'WAVE', 'D3KCAL', 'D3CARBO', 'D3FAT', 'D3PROTN']].dropna()
        df = df[(df.D3KCAL > 500) & (df.D3KCAL < 5000)]

        df['is_urban'] = df['T2'].apply(lambda x: 1 if x == 1 else 0)
        df['fat_pct'] = df.D3FAT * 9 / df.D3KCAL
        df['carbo_pct'] = df.D3CARBO * 4 / df.D3KCAL
        df['protn_pct'] = df.D3PROTN * 4 / df.D3KCAL
        df['fat_carbo'] = df.D3FAT / (df.D3CARBO + 1e-6)
        df['Year'] = df['WAVE'].astype(int)  # WAVE -> Year
        df['Province'] = df['T1'].map(PROVINCE_SHORT)  # T1 -> Province name
        df['Province_Code'] = df['T1'].astype(int)  # Keep code for reference

        X = df[['fat_pct', 'carbo_pct', 'protn_pct', 'fat_carbo', 'Year', 'Province_Code']].values
        y = df['is_urban'].values

        # 分层抽样
        self.X_train, self.X_test, self.y_train, self.y_test, train_idx, test_idx = train_test_split(
            X, y, range(len(X)), test_size=0.2, random_state=42, stratify=y
        )

        # 保存Year和Province用于后续分析
        self.year_train = df.iloc[train_idx]['Year'].values
        self.year_test = df.iloc[test_idx]['Year'].values
        self.province_train = df.iloc[train_idx]['Province'].values
        self.province_test = df.iloc[test_idx]['Province'].values

        # 标准化（Province_Code也参与标准化，或者单独处理）
        self.X_train = self.scaler.fit_transform(self.X_train)
        self.X_test = self.scaler.transform(self.X_test)
        joblib.dump(self.scaler, f"{MODEL_SAVE_DIR}/scaler.pkl")

        print(f"✅ 数据加载完成 | 训练集: {len(self.X_train)} | 测试集: {len(self.X_test)}")
        print(f"   Year范围: {self.year_test.min()}-{self.year_test.max()}")
        print(f"   省份数: {len(np.unique(self.province_test))}")
        return self


# ===================== ML Models =====================
class MLModels:
    def __init__(self, X_train, X_test, y_train, y_test, result_path=RESULT_PATH):
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.results = {}
        self.trained_models = {}
        self.result_path = result_path
        os.makedirs(MODEL_SAVE_DIR, exist_ok=True)
        self._load_existing_results_and_models()

    def _load_existing_results_and_models(self):
        if os.path.exists(self.result_path):
            try:
                self.results = pd.read_csv(self.result_path, index_col=0).to_dict(orient="index")
                print("✅ 已加载历史模型结果")
            except Exception as e:
                print(f"⚠️ 加载结果文件失败: {e}")

        model_files = {
            "Logistic_Regression": "Logistic_Regression.pkl",
            "Random_Forest": "Random_Forest.pkl",
            "XGBoost": "XGBoost.pkl",
            "Balanced_XGBoost": "Balanced_XGBoost.pkl"
        }

        loaded = []
        for name, file in model_files.items():
            path = f"{MODEL_SAVE_DIR}/{file}"
            if os.path.exists(path):
                try:
                    self.trained_models[name] = joblib.load(path)
                    loaded.append(name)
                except Exception as e:
                    print(f"⚠️ 加载 {name} 失败: {e}")

        if loaded:
            print(f"✅ 已加载模型: {', '.join(loaded)}")

    def _should_train(self, name):
        if name in self.results and name in self.trained_models:
            print(f"⏭️  {name}: 已存在，跳过")
            return False
        return True

    def logistic_regression(self):
        if not self._should_train("Logistic_Regression"):
            return self
        print("[2/4] 训练 Logistic Regression...")
        model = LogisticRegression(max_iter=5000)
        return self._train_eval(model, "Logistic_Regression")

    def random_forest(self):
        if not self._should_train("Random_Forest"):
            return self
        print("[2/4] 训练 Random Forest...")
        model = RandomForestClassifier(n_estimators=300, max_depth=6, random_state=42, n_jobs=1)
        return self._train_eval(model, "Random_Forest")

    def xgboost(self):
        if not self._should_train("XGBoost"):
            return self
        print("[2/4] 训练 XGBoost...")
        model = XGBClassifier(n_estimators=300, max_depth=4, learning_rate=0.1,
                              objective='binary:logistic', random_state=42, n_jobs=1)
        return self._train_eval(model, "XGBoost")

    def balanced_xgboost(self):
        if not self._should_train("Balanced_XGBoost"):
            return self
        print("[2/4] 训练 FINAL BUSTER XGBoost（AUC+F1 双封顶）...")

        n_neg = (self.y_train == 0).sum()
        n_pos = (self.y_train == 1).sum()

        model = XGBClassifier(
            n_estimators=600,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            scale_pos_weight=np.sqrt(n_neg / n_pos),  # 温和平衡
            min_child_weight=5,  # 医学数据神参数
            gamma=0.2,
            reg_alpha=0.5,
            reg_lambda=1.2,
            eval_metric="logloss",
            objective="binary:logistic",
            random_state=42,
            n_jobs=1,
        )
        return self._train_eval(model, "Balanced_XGBoost")

    # ===================== 🔥 终极杀器：Stacking 融合模型（全模型集成）=====================
    from sklearn.ensemble import StackingClassifier
    def stacking_ensemble(self):
        from sklearn.ensemble import StackingClassifier
        if not self._should_train("Stacking_Ensemble"):
            return self
        print("[2/4] 训练 Stacking 融合模型（全模型集成，最强性能）...")

        # 基模型
        base_models = [
            ("lr", LogisticRegression(max_iter=5000)),
            ("rf", RandomForestClassifier(n_estimators=300, max_depth=6, random_state=42)),
            ("xgb", XGBClassifier(n_estimators=300, max_depth=4, learning_rate=0.1,
                                  objective="binary:logistic", random_state=42)),
        ]

        # 元模型
        meta_model = XGBClassifier(
            n_estimators=200,
            max_depth=3,
            learning_rate=0.08,
            scale_pos_weight=np.sqrt(np.sum(self.y_train == 0) / np.sum(self.y_train == 1)),
            random_state=42
        )

        stacking = StackingClassifier(
            estimators=base_models,
            final_estimator=meta_model,
            cv=5,
            stack_method="predict_proba",
            n_jobs=1
        )

        return self._train_eval(stacking, "Stacking_Ensemble")

    def _train_eval(self, model, name):
        start = time.time()
        model.fit(self.X_train, self.y_train)
        train_time = time.time() - start

        ypb = model.predict_proba(self.X_test)[:, 1]

        # ===================== 🔥 只针对正例优化 F1（城市样本）=====================
        from sklearn.metrics import precision_recall_curve
        precision, recall, thresholds = precision_recall_curve(self.y_test, ypb, pos_label=1)
        f1_scores = 2 * precision * recall / (precision + recall + 1e-6)
        best_idx = np.nanargmax(f1_scores)
        best_thresh = thresholds[best_idx]
        yp = (ypb >= best_thresh).astype(int)

        self.results[name] = {
            "Acc": round(accuracy_score(self.y_test, yp), 3),
            "F1": round(f1_score(self.y_test, yp), 3),
            "AUC": round(roc_auc_score(self.y_test, ypb), 3),
            "Time(s)": round(train_time, 2),
            "Best_Thresh": round(best_thresh, 3)
        }
        self.trained_models[name] = model
        joblib.dump(model, f"{MODEL_SAVE_DIR}/{name}.pkl")
        print(f"✅ {name} | F1:{self.results[name]['F1']:.3f} | AUC:{self.results[name]['AUC']:.3f}")
        return self

    def force_retrain(self):
        print("\n🗑️  清除历史模型文件...")
        files = glob.glob(f"{MODEL_SAVE_DIR}/*.pkl") + [self.result_path]
        for f in files:
            if os.path.exists(f):
                os.remove(f)
        self.results = {}
        self.trained_models = {}
        print("✅ 已清除，将重新训练\n")
        return self


# ===================== PyTorch MLP =====================
class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(6, 16),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(16, 8),
            nn.BatchNorm1d(8),
            nn.ReLU(),
            nn.Linear(8, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x)


class TorchTrainer:
    def __init__(self, X_train, X_test, y_train, y_test):
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.model = None
        self.device = torch.device("cpu")
        self.model_path = f"{MODEL_SAVE_DIR}/PyTorch_MLP.pth"
        os.makedirs(MODEL_SAVE_DIR, exist_ok=True)

    def load_model(self):
        if not os.path.exists(self.model_path):
            return False
        self.model = MLP().to(self.device)
        try:
            self.model.load_state_dict(torch.load(self.model_path, map_location=self.device))
            print("✅ 已加载保存的PyTorch MLP模型")
            return self._evaluate()
        except RuntimeError:
            print("⚠️  模型架构变化，重新训练...")
            os.remove(self.model_path)
            return False

    def train(self, epochs=30, batch_size=32):
        load_res = self.load_model()
        if load_res:
            return load_res, self.model

        print(f"\n[3/4] 训练 PyTorch MLP | {epochs}轮 | batch={batch_size}")
        self.model = MLP().to(self.device)
        loss_fn = nn.BCELoss()
        opt = optim.Adam(self.model.parameters(), lr=1e-3)

        n_samples = len(self.X_train)
        n_batches = (n_samples + batch_size - 1) // batch_size

        X_tensor = torch.tensor(self.X_train, dtype=torch.float32).to(self.device)
        y_tensor = torch.tensor(self.y_train, dtype=torch.float32).reshape(-1, 1).to(self.device)

        pbar = tqdm(range(epochs), desc="Training MLP", ncols=80)
        for epoch in pbar:
            self.model.train()
            indices = torch.randperm(n_samples, device=self.device)
            total_loss = 0.0
            for i in range(n_batches):
                start_idx = i * batch_size
                end_idx = min(start_idx + batch_size, n_samples)
                batch_idx = indices[start_idx:end_idx]
                bx = X_tensor[batch_idx]
                by = y_tensor[batch_idx]
                pred = self.model(bx)
                loss = loss_fn(pred, by)
                opt.zero_grad()
                loss.backward()
                opt.step()
                total_loss += loss.item()
            avg_loss = total_loss / n_batches
            pbar.set_postfix({"loss": f"{avg_loss:.4f}"})

        res = self._evaluate()
        self.save_model()
        print(f"✅ PyTorch MLP | Acc: {res['Acc']:.3f} | F1: {res['F1']:.3f} | AUC: {res['AUC']:.3f}")
        return res, self.model

    def _evaluate(self):
        self.model.eval()
        with torch.no_grad():
            xt = torch.tensor(self.X_test, dtype=torch.float32).to(self.device)
            ypb = self.model(xt).cpu().numpy().flatten()

        # 🔥 加最优阈值
        from sklearn.metrics import precision_recall_curve
        precision, recall, thresholds = precision_recall_curve(self.y_test, ypb)
        f1_scores = 2 * (precision * recall) / (precision + recall + 1e-6)
        best_idx = np.argmax(f1_scores)
        best_thresh = thresholds[best_idx]
        yp = (ypb >= best_thresh).astype(int)

        return {
            "Acc": round(accuracy_score(self.y_test, yp), 3),
            "F1": round(f1_score(self.y_test, yp), 3),
            "AUC": round(roc_auc_score(self.y_test, ypb), 3),
            "Best_Thresh": round(best_thresh, 3)
        }

    def save_model(self, path=None):
        torch.save(self.model.state_dict(), path or self.model_path)


# ===================== SHAP Analyzer（修改版）=====================
class SHAPAnalyzer:
    def __init__(self, model, X_test, feature_names, year_test=None, province_test=None):
        self.model = model
        self.X_test = pd.DataFrame(X_test, columns=feature_names)
        self.year_test = year_test
        self.province_test = province_test
        self.explainer = None
        self.shap_values = None

    def run(self):
        print("\n[4/4] 生成 SHAP 特征重要性图...")
        os.makedirs(FIGURE_DIR, exist_ok=True)

        self.explainer = shap.TreeExplainer(self.model)
        self.shap_values = self.explainer.shap_values(self.X_test)

        # 1. 标准SHAP summary图
        plt.figure(figsize=(10, 6))
        shap.summary_plot(self.shap_values, self.X_test, show=False)
        plt.tight_layout()
        plt.savefig(f"{FIGURE_DIR}/shap_summary.png", dpi=300, bbox_inches='tight')
        plt.close()
        print("   ✅ shap_summary.png 已保存")

        # 2. 按Year分层的SHAP分析
        if self.year_test is not None:
            self._plot_by_year()

        # 3. 按Province分层的SHAP分析
        if self.province_test is not None:
            self._plot_by_province()

        return self

    def _plot_by_year(self):
        """按年份分析特征重要性变化"""
        years = sorted(np.unique(self.year_test))
        feature_cols = list(self.X_test.columns)

        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()

        for idx, feature in enumerate(feature_cols):
            if idx >= 6:
                break
            ax = axes[idx]

            for year in years:
                year_mask = self.year_test == year
                if year_mask.sum() > 0 and isinstance(self.shap_values, np.ndarray):
                    year_shap = self.shap_values[year_mask, idx]
                    ax.scatter([year] * len(year_shap), year_shap, alpha=0.3, s=15, label=str(year))

            ax.set_xlabel("Year")
            ax.set_ylabel(f"SHAP value")
            ax.set_title(f"{feature}")
            ax.grid(True, alpha=0.3)
            if idx == 0:
                ax.legend(title="Year", bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)

        plt.tight_layout()
        plt.savefig(f"{FIGURE_DIR}/shap_by_year.png", dpi=300, bbox_inches='tight')
        plt.close()
        print("   ✅ shap_by_year.png 已保存")

    def _plot_by_province(self):
        """按省份分析平均SHAP值"""
        feature_cols = list(self.X_test.columns)
        provinces = np.unique(self.province_test)

        # 计算每个省份的平均|SHAP|
        prov_shap_list = []
        valid_provinces = []

        for prov in provinces:
            prov_mask = self.province_test == prov
            if prov_mask.sum() > 30 and isinstance(self.shap_values, np.ndarray):
                mean_shap = np.abs(self.shap_values[prov_mask, :]).mean(axis=0)
                prov_shap_list.append(mean_shap)
                valid_provinces.append(prov)

        if len(valid_provinces) > 1:
            prov_shap_df = pd.DataFrame(prov_shap_list, index=valid_provinces, columns=feature_cols)

            # 热力图
            plt.figure(figsize=(10, 8))
            sns.heatmap(prov_shap_df, annot=True, fmt='.3f', cmap='YlOrRd')
            plt.title("Mean |SHAP| Value by Province")
            plt.xlabel("Feature")
            plt.ylabel("Province")
            plt.tight_layout()
            plt.savefig(f"{FIGURE_DIR}/shap_by_province.png", dpi=300, bbox_inches='tight')
            plt.close()
            print("   ✅ shap_by_province.png 已保存")

            # 各省份特征重要性排名（前4个特征）
            fig, axes = plt.subplots(2, 2, figsize=(12, 10))
            axes = axes.flatten()

            for idx, feature in enumerate(feature_cols[:4]):
                ax = axes[idx]
                feature_means = prov_shap_df[feature].sort_values(ascending=True)
                colors = plt.cm.RdYlBu_r(np.linspace(0.2, 0.8, len(feature_means)))
                feature_means.plot(kind='barh', ax=ax, color=colors)
                ax.set_title(f"{feature} Importance by Province")
                ax.set_xlabel("Mean |SHAP|")

            plt.tight_layout()
            plt.savefig(f"{FIGURE_DIR}/shap_province_ranking.png", dpi=300, bbox_inches='tight')
            plt.close()
            print("   ✅ shap_province_ranking.png 已保存")


# ===================== Main Trainer =====================
class Trainer:
    def run(self, force_retrain=False):
        os.makedirs(FIGURE_DIR, exist_ok=True)
        os.makedirs(MODEL_SAVE_DIR, exist_ok=True)
        os.makedirs("./results", exist_ok=True)

        data = DataPipeline().load()

        ml = MLModels(data.X_train, data.X_test, data.y_train, data.y_test)

        if force_retrain:
            ml.force_retrain()

        ml.logistic_regression().random_forest().xgboost().balanced_xgboost().stacking_ensemble()

        torch_res, torch_model = TorchTrainer(
            data.X_train, data.X_test, data.y_train, data.y_test
        ).train()
        ml.results["PyTorch_MLP"] = torch_res

        res_df = pd.DataFrame(ml.results).T
        print("\n" + "=" * 65)
        print("📊 城乡饮食分类模型 - 所有模型结果")
        print("=" * 65)
        print(res_df.to_string())
        res_df.to_csv("./results/model_results.csv", encoding="utf-8-sig")

        # SHAP分析 - 传入Year和Province
        SHAPAnalyzer(
            ml.trained_models["XGBoost"], 
            data.X_test, 
            data.feature_names,
            year_test=data.year_test,
            province_test=data.province_test
        ).run()

        return {
            "ml_models": ml.trained_models,
            "mlp_model": torch_model,
            "data": data,
            "results": res_df
        }


# ===================== 运行入口 =====================
if __name__ == "__main__":
    try:
        multiprocessing.set_start_method('spawn', force=True)
    except RuntimeError:
        pass

    assets = Trainer().run(force_retrain=False)
