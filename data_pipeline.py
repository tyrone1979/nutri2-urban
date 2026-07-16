import os

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from features import FEATURE_COLS, FEATURE_NAMES, NUTRIENT_ALL_COLS, assign_context_labels

DATA_PATH = "./data/c12diet.sas7bdat"
MODEL_SAVE_DIR = "./saved_models"

PROVINCE_SHORT = {
    11: "Beijing", 21: "Liaoning", 23: "Heilongjiang", 31: "Shanghai",
    32: "Jiangsu", 37: "Shandong", 41: "Henan", 42: "Hubei",
    43: "Hunan", 45: "Guangxi", 52: "Guizhou", 55: "Chongqing"
}


def engineer_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["fat_pct"] = df.D3FAT * 9 / df.D3KCAL
    df["carbo_pct"] = df.D3CARBO * 4 / df.D3KCAL
    df["protn_pct"] = df.D3PROTN * 4 / df.D3KCAL
    df["fat_carbo"] = df.D3FAT / (df.D3CARBO + 1e-6)
    df["Year"] = df.WAVE.astype(int)
    df["Province"] = df.T1.astype(int)
    df["label"] = assign_context_labels(df["T2"].astype(int), df["fat_pct"])
    return df


class DataPipeline:
    """Load CHNS data. Plan B: y from T2 + transitional FatER band; X = 4 macros + Year + Province."""

    def __init__(self, path=DATA_PATH, feature_cols=None):
        self.path = path
        self.feature_cols = feature_cols or FEATURE_COLS
        self.feature_names = [
            FEATURE_NAMES[FEATURE_COLS.index(c)] if c in FEATURE_COLS else c
            for c in self.feature_cols
        ]
        self.X_train = self.X_test = self.y_train = self.y_test = None
        self.year_train = self.year_test = None
        self.province_train = self.province_test = None
        self.nutrients_test = None  # all 4 macros for downstream (incl. FatER)
        self.scaler = StandardScaler()

    def load(self, verbose=True):
        if verbose:
            print("[1/4] 正在加载数据...", flush=True)
        if not os.path.exists(self.path):
            raise FileNotFoundError(f"数据文件不存在: {self.path}")

        try:
            import pyreadstat
            if verbose:
                print("   使用 pyreadstat 读取（更快）...", flush=True)
            df, _ = pyreadstat.read_sas7bdat(self.path)
        except ImportError:
            df = pd.read_sas(self.path, encoding="utf-8")

        df.columns = df.columns.str.upper()
        df = df[["T2", "T1", "WAVE", "D3KCAL", "D3CARBO", "D3FAT", "D3PROTN"]].dropna()
        df = df[(df.D3KCAL > 500) & (df.D3KCAL < 5000)]
        df = engineer_columns(df)

        X = df[self.feature_cols].values
        y = df["label"].values
        nutrients = df[NUTRIENT_ALL_COLS].values

        indices = np.arange(len(X))
        (self.X_train, self.X_test, self.y_train, self.y_test,
         _nutrients_train, self.nutrients_test,
         train_idx, test_idx) = train_test_split(
            X, y, nutrients, indices,
            test_size=0.2, random_state=42, stratify=y,
        )

        self.year_train = df.iloc[train_idx]["Year"].values
        self.year_test = df.iloc[test_idx]["Year"].values
        self.province_train = df.iloc[train_idx]["Province"].values
        self.province_test = df.iloc[test_idx]["Province"].values

        self.X_train = self.scaler.fit_transform(self.X_train)
        self.X_test = self.scaler.transform(self.X_test)
        os.makedirs(MODEL_SAVE_DIR, exist_ok=True)
        joblib.dump(self.scaler, f"{MODEL_SAVE_DIR}/scaler.pkl")

        if verbose:
            print(f"Data loaded | train: {len(self.X_train)} | test: {len(self.X_test)}", flush=True)
            print(f"   Features: {self.feature_names}", flush=True)
            total = df["label"].value_counts().sort_index()
            print(f"   Rural/Trans/Urban: {total.get(0,0)}/{total.get(1,0)}/{total.get(2,0)}", flush=True)
        return self
