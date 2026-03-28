import pandas as pd
import numpy as np
import time
import joblib  # 用于保存模型
import os

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

import shap
import matplotlib.pyplot as plt

# ===================== 全局配置 =====================
plt.rcParams["font.sans-serif"] = ["Arial"]
plt.rcParams["axes.unicode_minus"] = False

# 路径配置
DATA_PATH = "./data/c12diet.sas7bdat"
RESULT_PATH = "./results/model_results.csv"
MODEL_SAVE_DIR = "./saved_models"  # 模型保存目录
FIGURE_DIR = "./figures"

# ===================== Data Loader =====================
class DataPipeline:
    def __init__(self, path=DATA_PATH):
        self.path = path
        self.X_train, self.X_test, self.y_train, self.y_test = None, None, None, None
        self.feature_names = ["fat_energy_ratio", "carbo_energy_ratio", "protn_energy_ratio", "fat_carbo_ratio"]

    def load(self):
        print("[1/5] 正在加载数据...")
        df = pd.read_sas(self.path)
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
        print("✅ 数据加载与预处理完成\n")
        return self


# ===================== ML Models（带模型保存 + 进度） =====================
class MLModels:
    def __init__(self, X_train, X_test, y_train, y_test, result_path=RESULT_PATH):
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.results = {}
        self.trained_models = {}  # 保存所有训练好的模型
        self.result_path = result_path

        # 自动创建目录
        os.makedirs(MODEL_SAVE_DIR, exist_ok=True)

        # 加载历史结果
        if os.path.exists(self.result_path):
            self.results = pd.read_csv(self.result_path, index_col=0).to_dict(orient="index")
            print("✅ 已加载历史模型结果，跳过重复训练\n")

    def logistic_regression(self):
        if "Logistic_Regression" in self.results:
            return self
        print("[2/5] 正在训练 Logistic Regression...")
        model = LogisticRegression(max_iter=5000)
        return self._train_eval(model, "Logistic_Regression")

    def random_forest(self):
        if "Random_Forest" in self.results:
            return self
        print("[2/5] 正在训练 Random Forest...")
        model = RandomForestClassifier(n_estimators=300, max_depth=6, random_state=42)
        return self._train_eval(model, "Random_Forest")

    def xgboost(self):
        if "XGBoost" in self.results:
            return self
        print("[2/5] 正在训练 XGBoost...")
        model = XGBClassifier(n_estimators=300, max_depth=4, learning_rate=0.1,
                              objective='binary:logistic', random_state=42)
        return self._train_eval(model, "XGBoost")

    def _train_eval(self, model, name):
        # 训练
        model.fit(self.X_train, self.y_train)

        # 预测评估
        yp = model.predict(self.X_test)
        ypb = model.predict_proba(self.X_test)[:, 1]

        # 保存指标
        self.results[name] = {
            "Acc": round(accuracy_score(self.y_test, yp), 3),
            "F1": round(f1_score(self.y_test, yp), 3),
            "AUC": round(roc_auc_score(self.y_test, ypb), 3)
        }

        # 保存模型到内存 + 本地文件
        self.trained_models[name] = model
        joblib.dump(model, f"{MODEL_SAVE_DIR}/{name}.pkl")
        print(f"✅ {name} 训练完成 | Acc: {self.results[name]['Acc']}")
        return self

    def load_saved_models(self):
        """加载本地保存的所有模型（复用）"""
        model_names = ["Logistic_Regression", "Random_Forest", "XGBoost"]
        for name in model_names:
            path = f"{MODEL_SAVE_DIR}/{name}.pkl"
            if os.path.exists(path):
                self.trained_models[name] = joblib.load(path)
        print("✅ 所有本地模型加载完成")
        return self.trained_models


# ===================== PyTorch MLP（带进度） =====================
class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(4, 16), nn.ReLU(),
            nn.Linear(16, 8), nn.ReLU(),
            nn.Linear(8, 1), nn.Sigmoid()
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
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def train(self, epochs=30):
        print(f"\n[3/5] 开始训练 PyTorch MLP (设备: {self.device})")
        class DS(Dataset):
            def __init__(self, x, y):
                self.x = torch.tensor(x, dtype=torch.float32)
                self.y = torch.tensor(y, dtype=torch.float32).reshape(-1, 1)
            def __len__(self): return len(self.x)
            def __getitem__(self, i): return self.x[i], self.y[i]

        loader = DataLoader(DS(self.X_train, self.y_train), batch_size=32, shuffle=True)
        self.model = MLP().to(self.device)
        loss_fn = nn.BCELoss()
        opt = optim.Adam(self.model.parameters(), lr=1e-3)

        for epoch in range(epochs):
            self.model.train()
            total_loss = 0
            for bx, by in loader:
                bx, by = bx.to(self.device), by.to(self.device)
                loss = loss_fn(self.model(bx), by)
                opt.zero_grad()
                loss.backward()
                opt.step()
                total_loss += loss.item()

            if (epoch + 1) % 5 == 0:
                print(f"  Epoch {epoch+1:2d}/{epochs} | Loss: {total_loss:.4f}")

        # 评估
        self.model.eval()
        with torch.no_grad():
            ypb = self.model(torch.tensor(self.X_test, dtype=torch.float32).to(self.device)).cpu().numpy().flatten()
        yp = (ypb > 0.5).astype(int)

        res = {
            "Acc": round(accuracy_score(self.y_test, yp), 3),
            "F1": round(f1_score(self.y_test, yp), 3),
            "AUC": round(roc_auc_score(self.y_test, ypb), 3)
        }
        print(f"✅ PyTorch MLP 训练完成 | Acc: {res['Acc']}\n")
        return res, self.model

    def save_model(self, path=f"{MODEL_SAVE_DIR}/PyTorch_MLP.pth"):
        torch.save(self.model.state_dict(), path)


# ===================== SHAP Analyzer =====================
class SHAPAnalyzer:
    def __init__(self, model, X_test, feature_names):
        self.model = model
        self.X_test = pd.DataFrame(X_test, columns=feature_names)
        self.explainer = shap.TreeExplainer(model)
        self.shap_values = None

    def run(self):
        print("[4/5] 正在生成 SHAP 特征重要性图...")
        self.shap_values = self.explainer.shap_values(self.X_test)
        shap.summary_plot(self.shap_values, self.X_test, show=False)
        plt.savefig(f"{FIGURE_DIR}/shap_summary.png", dpi=300, bbox_inches='tight')
        plt.close()
        print("✅ SHAP 图已保存\n")


# ===================== Main Trainer =====================
class Trainer:
    def run(self):
        # 创建目录
        os.makedirs(FIGURE_DIR, exist_ok=True)
        os.makedirs(MODEL_SAVE_DIR, exist_ok=True)
        os.makedirs("./results", exist_ok=True)

        # 1. 数据
        data = DataPipeline().load()

        # 2. 机器学习模型
        ml = MLModels(data.X_train, data.X_test, data.y_train, data.y_test)
        ml.logistic_regression().random_forest().xgboost()

        # 3. 深度学习模型
        torch_res, torch_model = TorchTrainer(
            data.X_train, data.X_test, data.y_train, data.y_test
        ).train()
        ml.results["PyTorch_MLP"] = torch_res

        # 输出结果
        res_df = pd.DataFrame(ml.results).T
        print("\n" + "=" * 60)
        print("📊 城乡饮食分类模型 - 所有模型结果")
        print("=" * 60)
        print(res_df)
        res_df.to_csv("./results/model_results.csv", encoding="utf-8-sig")

        # 4. SHAP分析
        SHAPAnalyzer(ml.trained_models["XGBoost"], data.X_test, data.feature_names).run()

        # 5. 保存全部模型
        print("[5/5] 保存所有训练好的模型到本地...")
        joblib.dump(ml.trained_models, f"{MODEL_SAVE_DIR}/all_ml_models.pkl")
        print("✅ 所有模型已保存至：./saved_models/\n")
        print("🎉 全部任务运行完成！")

        # 返回所有训练好的模型，方便后续直接使用
        return {
            "ml_models": ml.trained_models,
            "mlp_model": torch_model,
            "data": data,
            "results": res_df
        }


# ===================== 运行入口 =====================
if __name__ == "__main__":
    # 运行训练，返回所有模型
    assets = Trainer().run()

    # ============= 后续直接使用已训练好的模型 =============
    # 示例：直接调用XGBoost模型预测
    # xgb_model = assets["ml_models"]["XGBoost"]
    # y_pred = xgb_model.predict(assets["data"].X_test)