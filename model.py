#!/usr/bin/env python3
# ===================== 必须在最开头！Apple Silicon + Python 3.13 防卡死 =====================
import os

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

# 🔥 关键：禁用所有 Python multiprocessing（Python 3.13 在 Apple Silicon 上有严重问题）
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

warnings.filterwarnings('ignore')

# ===================== 全局配置 =====================
plt.rcParams["font.sans-serif"] = ["Arial"]
plt.rcParams["axes.unicode_minus"] = False

DATA_PATH = "./data/c12diet.sas7bdat"
RESULT_PATH = "./results/model_results.csv"
MODEL_SAVE_DIR = "./saved_models"
FIGURE_DIR = "./figures"


# ===================== Data Pipeline（优化版：使用 pyreadstat 替代 pandas）=====================
class DataPipeline:
    def __init__(self, path=DATA_PATH):
        self.path = path
        self.X_train, self.X_test, self.y_train, self.y_test = None, None, None, None
        self.feature_names = ["fat_energy_ratio", "carbo_energy_ratio", "protn_energy_ratio", "fat_carbo_ratio"]

    def load(self):
        print("[1/5] 正在加载数据...")
        if not os.path.exists(self.path):
            raise FileNotFoundError(f"数据文件不存在: {self.path}")

        # 🔥 优化：尝试使用 pyreadstat（更快），失败则回退到 pandas
        try:
            import pyreadstat
            print("   使用 pyreadstat 读取（更快）...")
            df, meta = pyreadstat.read_sas7bdat(self.path)
        except ImportError:
            print("   pyreadstat 未安装，使用 pandas.read_sas（较慢）...")
            print("   💡 建议: pip install pyreadstat 加速读取")
            df = pd.read_sas(self.path, encoding='utf-8')

        df.columns = df.columns.str.upper()
        df = df[['T2', 'D3KCAL', 'D3CARBO', 'D3FAT', 'D3PROTN']].dropna()
        df = df[(df.D3KCAL > 500) & (df.D3KCAL < 5000)]

        df['is_urban'] = df['T2'].apply(lambda x: 1 if x == 1 else 0)
        df['fat_pct'] = df.D3FAT * 9 / df.D3KCAL
        df['carbo_pct'] = df.D3CARBO * 4 / df.D3KCAL
        df['protn_pct'] = df.D3PROTN * 4 / df.D3KCAL
        df['fat_carbo'] = df.D3FAT / (df.D3CARBO + 1e-6)

        X = df[['fat_pct', 'carbo_pct', 'protn_pct', 'fat_carbo']].values
        y = df['is_urban'].values

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        print(f"✅ 数据加载完成 | 训练集: {len(self.X_train)} | 测试集: {len(self.X_test)}\n")
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
        """加载历史结果和对应的模型文件"""
        if os.path.exists(self.result_path):
            try:
                self.results = pd.read_csv(self.result_path, index_col=0).to_dict(orient="index")
                print("✅ 已加载历史模型结果")
            except Exception as e:
                print(f"⚠️ 加载结果文件失败: {e}")

        model_files = {
            "Logistic_Regression": "Logistic_Regression.pkl",
            "Random_Forest": "Random_Forest.pkl",
            "XGBoost": "XGBoost.pkl"
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
            print("💡 删除 ./saved_models/ 下的 .pkl 文件可强制重新训练\n")

    def _should_train(self, name):
        if name in self.results and name in self.trained_models:
            print(f"⏭️  {name}: 已存在，跳过")
            return False
        return True

    def logistic_regression(self):
        if not self._should_train("Logistic_Regression"):
            return self
        print("[2/5] 训练 Logistic Regression...")
        model = LogisticRegression(max_iter=5000)
        return self._train_eval(model, "Logistic_Regression")

    def random_forest(self):
        if not self._should_train("Random_Forest"):
            return self
        print("[2/5] 训练 Random Forest...")
        model = RandomForestClassifier(n_estimators=300, max_depth=6, random_state=42, n_jobs=1)
        return self._train_eval(model, "Random_Forest")

    def xgboost(self):
        if not self._should_train("XGBoost"):
            return self
        print("[2/5] 训练 XGBoost...")
        model = XGBClassifier(n_estimators=300, max_depth=4, learning_rate=0.1,
                              objective='binary:logistic', random_state=42, n_jobs=1)
        return self._train_eval(model, "XGBoost")

    def _train_eval(self, model, name):
        start = time.time()
        model.fit(self.X_train, self.y_train)
        train_time = time.time() - start

        yp = model.predict(self.X_test)
        ypb = model.predict_proba(self.X_test)[:, 1]

        self.results[name] = {
            "Acc": round(accuracy_score(self.y_test, yp), 3),
            "F1": round(f1_score(self.y_test, yp), 3),
            "AUC": round(roc_auc_score(self.y_test, ypb), 3),
            "Time(s)": round(train_time, 2)
        }
        self.trained_models[name] = model
        joblib.dump(model, f"{MODEL_SAVE_DIR}/{name}.pkl")
        print(f"✅ {name} 完成 | Acc: {self.results[name]['Acc']:.3f} | 耗时: {train_time:.1f}s")
        return self

    def force_retrain(self):
        """强制重新训练"""
        print("\n🗑️  清除历史模型文件...")
        files = glob.glob(f"{MODEL_SAVE_DIR}/*.pkl") + [self.result_path]
        for f in files:
            if os.path.exists(f):
                os.remove(f)
                print(f"   删除: {os.path.basename(f)}")
        self.results = {}
        self.trained_models = {}
        print("✅ 已清除，将重新训练\n")
        return self


# ===================== PyTorch MLP（Apple Silicon + Python 3.13 终极修复版）=====================
class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(4, 16),
            nn.ReLU(),
            nn.Linear(16, 8),
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
        if os.path.exists(self.model_path):
            print(f"\n[3/5] 发现已保存的PyTorch MLP模型，正在加载...")
            self.model = MLP().to(self.device)
            self.model.load_state_dict(torch.load(self.model_path, map_location=self.device))
            print("✅ PyTorch MLP模型加载完成")
            return self._evaluate()
        return False

    def train(self, epochs=30, batch_size=32):
        load_res = self.load_model()
        if load_res:
            return load_res, self.model

        print(f"\n[3/5] 开始训练 PyTorch MLP | 轮次: {epochs} | 批次: {batch_size}")
        print(f"📱 设备: {self.device} | 线程: {torch.get_num_threads()}")
        print("⚠️  使用手动batch循环（禁用DataLoader防止卡死）\n")

        self.model = MLP().to(self.device)
        loss_fn = nn.BCELoss()
        opt = optim.Adam(self.model.parameters(), lr=1e-3)

        # 🔥 终极修复：完全不用 DataLoader，纯手动 batch（避免 Python 3.13 multiprocessing bug）
        n_samples = len(self.X_train)
        n_batches = (n_samples + batch_size - 1) // batch_size

        # 预转换张量（只转换一次）
        X_tensor = torch.tensor(self.X_train, dtype=torch.float32).to(self.device)
        y_tensor = torch.tensor(self.y_train, dtype=torch.float32).reshape(-1, 1).to(self.device)

        print(f"📊 样本数: {n_samples} | 每轮 batches: {n_batches}")

        # 外层 epoch 进度条
        for epoch in range(epochs):
            self.model.train()

            # 手动 shuffle（PyTorch 原生，无 numpy 依赖）
            indices = torch.randperm(n_samples, device=self.device)
            total_loss = 0.0

            # 内层 batch 进度条（使用简单 print 避免 tqdm 在 PyCharm 中的刷新问题）
            print(f"\n🔄 Epoch {epoch + 1}/{epochs} ", end="")

            for i in range(n_batches):
                start_idx = i * batch_size
                end_idx = min(start_idx + batch_size, n_samples)

                # 直接切片
                batch_idx = indices[start_idx:end_idx]
                bx = X_tensor[batch_idx]
                by = y_tensor[batch_idx]

                # 训练步骤
                pred = self.model(bx)
                loss = loss_fn(pred, by)

                opt.zero_grad()
                loss.backward()
                opt.step()

                total_loss += loss.item()

                # 进度指示（每10个batch显示一个点）
                if (i + 1) % max(1, n_batches // 10) == 0:
                    print(".", end="", flush=True)

            avg_loss = total_loss / n_batches
            print(f" | Loss: {avg_loss:.4f}")

        res = self._evaluate()
        self.save_model()
        print(f"\n🎉 PyTorch MLP 完成 | Acc: {res['Acc']:.3f} | F1: {res['F1']:.3f} | AUC: {res['AUC']:.3f}")
        return res, self.model

    def _evaluate(self):
        self.model.eval()
        with torch.no_grad():
            xt = torch.tensor(self.X_test, dtype=torch.float32).to(self.device)
            ypb = self.model(xt).cpu().numpy().flatten()
        yp = (ypb > 0.5).astype(int)
        return {
            "Acc": round(accuracy_score(self.y_test, yp), 3),
            "F1": round(f1_score(self.y_test, yp), 3),
            "AUC": round(roc_auc_score(self.y_test, ypb), 3)
        }

    def save_model(self, path=None):
        torch.save(self.model.state_dict(), path or self.model_path)


# ===================== SHAP Analyzer =====================
class SHAPAnalyzer:
    def __init__(self, model, X_test, feature_names):
        self.model = model
        self.X_test = pd.DataFrame(X_test, columns=feature_names)
        self.explainer = shap.TreeExplainer(model)
        self.shap_values = None

    def run(self):
        print("\n[4/5] 生成 SHAP 特征重要性图...")
        self.shap_values = self.explainer.shap_values(self.X_test)
        shap.summary_plot(self.shap_values, self.X_test, show=False)
        plt.savefig(f"{FIGURE_DIR}/shap_summary.png", dpi=300, bbox_inches='tight')
        plt.close()
        print("✅ SHAP 图已保存")


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

        ml.logistic_regression().random_forest().xgboost()

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

        SHAPAnalyzer(ml.trained_models["XGBoost"], data.X_test, data.feature_names).run()

        print("\n[5/5] 保存所有模型...")
        joblib.dump(ml.trained_models, f"{MODEL_SAVE_DIR}/all_ml_models.pkl")
        print("🎉 全部完成！\n")

        return {
            "ml_models": ml.trained_models,
            "mlp_model": torch_model,
            "data": data,
            "results": res_df
        }


# ===================== 运行入口 =====================
if __name__ == "__main__":
    # 再次确保 spawn 模式
    try:
        multiprocessing.set_start_method('spawn', force=True)
    except RuntimeError:
        pass

    # 运行训练
    assets = Trainer().run(force_retrain=False)