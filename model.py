import pandas as pd
import numpy as np

from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score
import torch
import torch.nn as nn
import torch.optim as optim

# ===================== 1. 数据加载类 =====================
class DataPipeline:
    def __init__(self, path="./data/c12diet.sas7bdat"):
        self.path = path
        self.df = None
        self.X_train, self.X_test, self.y_train, self.y_test = None, None, None, None

    def load(self):
        df = pd.read_sas(self.path)
        df.columns = df.columns.str.upper()
        df = df[['T2', 'D3KCAL', 'D3CARBO', 'D3FAT', 'D3PROTN']].dropna()
        df = df[(df.D3KCAL > 500) & (df.D3KCAL < 5000)]

        # 城乡标签
        df['is_urban'] = df['T2'].apply(lambda x: 1 if x == 1 else 0)

        # 4个供能比
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


# ===================== 2. 传统机器学习模型类 =====================
class MLModels:
    def __init__(self, X_train, X_test, y_train, y_test):
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.results = {}

    def logistic_regression(self):
        m = LogisticRegression(max_iter=5000)
        return self._train_eval(m, "Logistic Regression")

    def random_forest(self):
        m = RandomForestClassifier(300, max_depth=6)
        return self._train_eval(m, "Random Forest")

    def xgboost(self):
        m = XGBClassifier(300, max_depth=4, learning_rate=0.1)
        return self._train_eval(m, "XGBoost")

    def _train_eval(self, model, name):
        model.fit(self.X_train, self.y_train)
        yp = model.predict(self.X_test)
        ypb = model.predict_proba(self.X_test)[:, 1]
        self.results[name] = {
            "Acc": round(accuracy_score(self.y_test, yp), 3),
            "F1": round(f1_score(self.y_test, yp), 3),
            "AUC": round(roc_auc_score(self.y_test, ypb), 3)
        }
        return self


# ===================== 3. PyTorch MLP 模型类 =====================
class TorchMLP(nn.Module):
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
        class Ds(Dataset):
            def __init__(self, x, y):
                self.x = torch.tensor(x, dtype=torch.float32)
                self.y = torch.tensor(y, dtype=torch.float32).reshape(-1, 1)

            def __len__(self):
                return len(self.x)

            def __getitem__(self, i):
                return self.x[i], self.y[i]

        train_loader = DataLoader(Ds(self.X_train, self.y_train), batch_size=32, shuffle=True)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = TorchMLP().to(device)
        loss_fn = nn.BCELoss()
        opt = optim.Adam(model.parameters(), lr=1e-3)

        model.train()
        for _ in range(epochs):
            for bx, by in train_loader:
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


# ===================== 4. 主执行类 =====================
class Trainer:
    def run(self):
        # 1. 加载数据
        data = DataPipeline().load()

        # 2. 传统模型
        ml = MLModels(data.X_train, data.X_test, data.y_train, data.y_test)
        ml.logistic_regression().random_forest().xgboost()

        # 3. 深度学习
        torch_res = TorchTrainer(data.X_train, data.X_test, data.y_train, data.y_test).train()
        ml.results["PyTorch MLP"] = torch_res

        # 输出
        print("\n" + "=" * 60)
        print("📊 城乡分类模型对比 Urban vs Rural")
        print("=" * 60)
        res_df = pd.DataFrame(ml.results).T
        print(res_df)
        res_df.to_csv("./results/model_comparison.csv", index=True, encoding="utf-8-sig")


if __name__ == "__main__":
    Trainer().run()