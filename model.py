import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score
import torch
import torch.nn as nn
import torch.optim as optim
import shap
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader

# ===================== Data Loader =====================
class DataPipeline:
    def __init__(self, path="./data/c12diet.sas7bdat"):
        self.path = path
        self.X_train, self.X_test, self.y_train, self.y_test = None, None, None, None
        self.feature_names = [
            "Fat_Energy_Ratio",
            "Carbo_Energy_Ratio",
            "Protein_Energy_Ratio",
            "Fat_Carbo_Ratio"
        ]

    def load(self):
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
        return self


# ===================== ML Models =====================
class MLModels:
    def __init__(self, X_train, X_test, y_train, y_test, result_path="./results/model_results.csv"):
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.results = {}
        self.trained_models = {}
        self.result_path = result_path

        # 如果结果已存在，直接加载
        if os.path.exists(self.result_path):
            self.results = pd.read_csv(self.result_path, index_col=0).to_dict(orient="index")
            print("✅ 已加载历史结果，跳过训练")

    def logistic_regression(self):
        if "Logistic_Regression" in self.results:
            return self
        model = LogisticRegression(max_iter=5000)
        return self._train_eval(model, "Logistic_Regression")

    def random_forest(self):
        if "Random_Forest" in self.results:
            return self
        model = RandomForestClassifier(n_estimators=300, max_depth=6)
        return self._train_eval(model, "Random_Forest")

    def xgboost(self):
        if "XGBoost" in self.results:
            return self
        model = XGBClassifier(n_estimators=300, max_depth=4, learning_rate=0.1, objective='binary:logistic')
        return self._train_eval(model, "XGBoost")

    def _train_eval(self, model, name):
        model.fit(self.X_train, self.y_train)
        yp = model.predict(self.X_test)
        ypb = model.predict_proba(self.X_test)[:, 1]
        self.results[name] = {
            "Acc": round(accuracy_score(self.y_test, yp), 3),
            "F1": round(f1_score(self.y_test, yp), 3),
            "AUC": round(roc_auc_score(self.y_test, ypb), 3)
        }
        self.trained_models[name] = model
        return self


# ===================== PyTorch MLP =====================
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

    def train(self, epochs=30):
        class DS(Dataset):
            def __init__(self, x, y):
                self.x = torch.tensor(x, dtype=torch.float32)
                self.y = torch.tensor(y, dtype=torch.float32).reshape(-1, 1)

            def __len__(self): return len(self.x)
            def __getitem__(self, i): return self.x[i], self.y[i]

        loader = DataLoader(DS(self.X_train, self.y_train), batch_size=32, shuffle=True)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = MLP().to(device)
        loss_fn = nn.BCELoss()
        opt = optim.Adam(model.parameters(), lr=1e-3)

        for _ in range(epochs):
            for bx, by in loader:
                bx, by = bx.to(device), by.to(device)
                loss = loss_fn(model(bx), by)
                opt.zero_grad()
                loss.backward()
                opt.step()

        model.eval()
        with torch.no_grad():
            ypb = model(torch.tensor(self.X_test, dtype=torch.float32)).cpu().numpy().flatten()
        yp = (ypb > 0.5).astype(int)

        return {
            "Acc": round(accuracy_score(self.y_test, yp), 3),
            "F1": round(f1_score(self.y_test, yp), 3),
            "AUC": round(roc_auc_score(self.y_test, ypb), 3)
        }


# ===================== SHAP Analyzer =====================
class SHAPAnalyzer:
    def __init__(self, model, X_test, feature_names):
        self.model = model
        self.X_test = pd.DataFrame(X_test, columns=feature_names)
        self.explainer = shap.TreeExplainer(model)
        self.shap_values = None

    def run(self):
        self.shap_values = self.explainer.shap_values(self.X_test)
        shap.summary_plot(self.shap_values, self.X_test, show=False)
        plt.savefig("./figures/shap_summary.png", dpi=300, bbox_inches='tight')
        plt.close()
        print("✅ SHAP 图已保存至 ./figures/shap_summary.png")


# ===================== Main Trainer =====================
class Trainer:
    def run(self):
        import os
        os.makedirs("./results", exist_ok=True)
        os.makedirs("./figures", exist_ok=True)

        data = DataPipeline().load()
        ml = MLModels(data.X_train, data.X_test, data.y_train, data.y_test)
        ml.logistic_regression().random_forest().xgboost()

        torch_res = TorchTrainer(data.X_train, data.X_test, data.y_train, data.y_test).train()
        ml.results["PyTorch_MLP"] = torch_res

        res_df = pd.DataFrame(ml.results).T
        print("\n" + "=" * 60)
        print("📊 城乡分类模型结果")
        print("=" * 60)
        print(res_df)
        res_df.to_csv("./results/model_results.csv", encoding="utf-8-sig")

        # SHAP for XGBoost
        SHAPAnalyzer(ml.trained_models["XGBoost"], data.X_test, data.feature_names).run()


if __name__ == "__main__":
    Trainer().run()