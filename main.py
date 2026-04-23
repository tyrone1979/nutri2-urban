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

        df['fat_pct'] = df.D3FAT * 9 / df.D3KCAL
        df['carbo_pct'] = df.D3CARBO * 4 / df.D3KCAL
        df['protn_pct'] = df.D3PROTN * 4 / df.D3KCAL
        df['fat_carbo'] = df.D3FAT / (df.D3CARBO + 1e-6)
        df['Year'] = df['WAVE'].astype(int)
        df['Province'] = df['T1'].astype(int)

        # ===================== 3分类标签 =====================
        df['label'] = 0  # 农村
        df.loc[df['T2'] == 1, 'label'] = 2  # 城市
        # 中间过渡类
        mask_mid = (df['fat_pct'] >= 0.23) & (df['fat_pct'] <= 0.30)
        df.loc[mask_mid, 'label'] = 1

        X = df[['fat_pct', 'carbo_pct', 'protn_pct', 'fat_carbo', 'Year', 'Province']].values
        y = df['label'].values  # 不再用 is_urban

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
        # ===================== Statistics of class counts by province =====================
        print("\n" + "=" * 60)
        print("📊 Dataset Class Statistics (0=Rural, 1=Urban/Rural, 2=Urban)")
        print("=" * 60)

        # Overall count
        total_counts = df['label'].value_counts().sort_index()
        print("Overall class counts:")
        print(f"  Rural (0)          : {total_counts.get(0, 0)}")
        print(f"  Urban/Rural (1)    : {total_counts.get(1, 0)}")
        print(f"  Urban (2)          : {total_counts.get(2, 0)}")
        print(f"  Total              : {len(df)}")

        # By province
        print("\nClass distribution by province:")
        prov_counts = df.groupby('Province')['label'].value_counts().unstack(fill_value=0)
        for code in sorted(prov_counts.index):
            prov_name = PROVINCE_SHORT.get(code, f"Province{code}")
            rural = prov_counts.loc[code, 0] if 0 in prov_counts.columns else 0
            mixed = prov_counts.loc[code, 1] if 1 in prov_counts.columns else 0
            urban = prov_counts.loc[code, 2] if 2 in prov_counts.columns else 0
            print(f"  {prov_name:12s} | Rural:{rural:4d}  Urban/Rural:{mixed:4d}  Urban:{urban:4d}")

        print("=" * 60)
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
                              objective='multi:softmax', num_class=3,random_state=42, n_jobs=1)
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


    def _train_eval(self, model, name):
        start = time.time()
        model.fit(self.X_train, self.y_train)
        train_time = time.time() - start

        ypb = model.predict_proba(self.X_test)
        yp = model.predict(self.X_test)

        self.results[name] = {
            "Acc": round(accuracy_score(self.y_test, yp), 3),
            "F1": round(f1_score(self.y_test, yp, average='weighted'), 3),
            "AUC": round(roc_auc_score(self.y_test, ypb, multi_class='ovo', average='macro'), 3),
            "Time(s)": round(train_time, 2)
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


class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(6, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, 3)
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
            return self._evaluate()
        except RuntimeError:
            if os.path.exists(self.model_path):
                os.remove(self.model_path)
            return False

    def train(self, epochs=150, batch_size=1024):
        load_res = self.load_model()
        if load_res:
            return load_res, self.model

        print(f"\n[3/4] 训练 PyTorch MLP 3分类 | {epochs}轮 | batch={batch_size}")
        self.model = MLP().to(self.device)
        loss_fn = nn.CrossEntropyLoss()
        opt = optim.Adam(self.model.parameters(), lr=1e-3)

        X_tensor = torch.tensor(self.X_train, dtype=torch.float32).to(self.device)
        y_tensor = torch.tensor(self.y_train, dtype=torch.long).to(self.device)

        n_samples = len(X_tensor)
        pbar = tqdm(range(epochs), desc="Training MLP", ncols=80)

        # 🔥 开始计时
        start_time = time.time()

        for epoch in pbar:
            self.model.train()
            perm = torch.randperm(n_samples).to(self.device)
            total_loss = 0

            for i in range(0, n_samples, batch_size):
                idx = perm[i:i+batch_size]
                bx, by = X_tensor[idx], y_tensor[idx]
                pred = self.model(bx)
                loss = loss_fn(pred, by)
                opt.zero_grad()
                loss.backward()
                opt.step()
                total_loss += loss.item()

            avg_loss = total_loss / (n_samples // batch_size + 1)
            pbar.set_postfix(loss=f"{avg_loss:.3f}")

        # 🔥 计算总时间
        train_time = time.time() - start_time

        res = self._evaluate()
        res["Time(s)"] = round(train_time, 2)
        self.save_model()

        print(f"✅ MLP 3分类 | Acc:{res['Acc']:.3f} F1(w):{res['F1']:.3f} AUC:{res['AUC']:.3f} Time:{res['Time(s)']:.2f}s")
        return res, self.model

    def _evaluate(self):
        self.model.eval()
        with torch.no_grad():
            xt = torch.tensor(self.X_test, dtype=torch.float32).to(self.device)
            logits = self.model(xt)
            ypb = torch.softmax(logits, dim=1).cpu().numpy()
            yp = logits.argmax(1).cpu().numpy()

        return {
            "Acc": round(accuracy_score(self.y_test, yp), 3),
            "F1": round(f1_score(self.y_test, yp, average='weighted'), 3),
            "AUC": round(roc_auc_score(self.y_test, ypb, multi_class='ovo', average='macro'), 3),
            "Time(s)": 0.0
        }

    def save_model(self, path=None):
        torch.save(self.model.state_dict(), path or self.model_path)


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

        ml.logistic_regression().random_forest().xgboost().balanced_xgboost()

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
