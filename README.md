# An Explainable AI Framework with SHAP-Enhanced Ensemble Learning for Intelligent Urban-Rural Dietary Pattern Recognition

---

## Overview

This repository implements an **explainable AI framework** for urban-rural dietary pattern classification using the China Health and Nutrition Survey (CHNS) dataset. The framework integrates ensemble learning with SHAP-based interpretability to provide both accurate predictions and actionable insights for population health analytics.

### Key Achievements

- **High-performance classification**: XGBoost achieves **0.782 accuracy** and **0.915 AUC** on three-class dietary pattern recognition
- **Multi-model ensemble**: Logistic Regression, Random Forest, XGBoost, Balanced XGBoost, and Multi-Layer Perceptron
- **SHAP-based interpretability**: Global feature importance, local explanations, temporal/spatial stratification
- **Industrial-grade pipeline**: Scalable preprocessing, reproducible results, production-ready model serialization

---

## Table of Contents

- [Installation](#installation)
- [Dataset](#dataset)
- [Framework Architecture](#framework-architecture)
- [Usage](#usage)
- [Model Performance](#model-performance)
- [SHAP Interpretability](#shap-interpretability)
- [Project Structure](#project-structure)
- [Results](#results)
- [Citation](#citation)
- [License](#license)

---

## Installation

### Prerequisites

- Python 3.8+
- pip / conda

### Setup

```bash
# Clone repository
git clone https://github.com/your-repo/dietary-pattern-xai.git
cd dietary-pattern-xai

# Install dependencies
pip install -r requirements.txt
```

### Requirements

```
pandas>=1.3.0
numpy>=1.21.0
scikit-learn>=1.0.0
xgboost>=1.5.0
torch>=1.10.0
shap>=0.40.0
matplotlib>=3.4.0
seaborn>=0.11.0
joblib>=1.1.0
pyreadstat>=1.1.0
tqdm>=4.62.0
```

---

## Dataset

### Source

**China Health and Nutrition Survey (CHNS)** – `c12diet.sas7bdat`

| Attribute | Value |
|-----------|-------|
| Survey period | 1991–2011 (8 waves) |
| Sample size | 101,926 (after QC) |
| Geographic coverage | 12 provinces |
| Format | SAS7BDAT |

### Quality Control

```python
# Energy intake filtering (500-5000 kcal)
df = df[(df.D3KCAL > 500) & (df.D3KCAL < 5000)]
```

### Feature Engineering

Six features are constructed for classification:

| Feature | Formula | Description |
|---------|---------|-------------|
| **FatER** | (Fat × 9) / Total Energy | Fat energy ratio |
| **CarbER** | (Carb × 4) / Total Energy | Carbohydrate energy ratio |
| **ProtER** | (Protein × 4) / Total Energy | Protein energy ratio |
| **FCR** | Fat / (Carb + 1e-6) | Fat-to-carbohydrate ratio |
| **Year** | WAVE | Survey year (standardized) |
| **Province** | T1 | Geographic region (encoded) |

### Target Variable (3-Class)

| Class | Label | Fat Energy Ratio | Sample Count | Percentage |
|-------|-------|------------------|--------------|------------|
| 0 | Rural | < 23% | 52,257 | 51.3% |
| 1 | Transitional | 23–30% | 22,982 | 22.5% |
| 2 | Urban | > 30% | 26,687 | 26.2% |

### Data Split

| Split | Samples | Percentage |
|-------|---------|------------|
| Training | 81,540 | 80% |
| Testing | 20,386 | 20% |

---

## Framework Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                    Data Ingestion Layer                             │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────────────┐  │
│  │ CHNS Data   │→ │ Quality     │→ │ Feature Engineering         │  │
│  │ (SAS)       │  │ Control     │  │ + Standardization           │  │
│  └─────────────┘  └─────────────┘  └─────────────────────────────┘  │
├─────────────────────────────────────────────────────────────────────┤
│                    Model Training Layer                             │
│  ┌───────────────────────────────────────────────────────────────┐  │
│  │ Ensemble Learning (5 Models)                                  │  │
│  │ LR | RF | XGB | Balanced XGB | MLP                            │  │
│  └───────────────────────────────────────────────────────────────┘  │
│  ┌───────────────────────────────────────────────────────────────┐  │
│  │ Hyperparameter Optimization | Class Balancing                 │  │
│  └───────────────────────────────────────────────────────────────┘  │
├─────────────────────────────────────────────────────────────────────┤
│                    Evaluation Layer                                 │
│  ┌───────────────────────────────────────────────────────────────┐  │
│  │ Accuracy | Weighted F1 | Macro AUC                           │  │
│  │ Stratified 80/20 Split | Cross-validation                    │  │
│  └───────────────────────────────────────────────────────────────┘  │
├─────────────────────────────────────────────────────────────────────┤
│                 SHAP Interpretability Layer                         │
│  ┌───────────────────────────────────────────────────────────────┐  │
│  │ Global Feature Importance                                     │  │
│  │ Local Prediction Explanations                                 │  │
│  │ Temporal Stratification (1991-2011)                           │  │
│  │ Spatial Stratification (12 Provinces)                         │  │
│  └───────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Usage

### Quick Start

```python
from model import Trainer

# Run full pipeline
trainer = Trainer()
results = trainer.run(force_retrain=False)
```

### Force Retrain

```python
# Clear existing models and retrain
results = trainer.run(force_retrain=True)
```

### Training Individual Models

```python
from model import DataPipeline, MLModels

# Load and preprocess data
data = DataPipeline().load()

# Train specific models
ml = MLModels(data.X_train, data.X_test, data.y_train, data.y_test)
ml.logistic_regression()      # Train LR
ml.random_forest()            # Train RF
ml.xgboost()                  # Train XGBoost
ml.balanced_xgboost()         # Train Balanced XGBoost

# View results
print(ml.results)
```

### SHAP Analysis

```python
from model import SHAPAnalyzer

# Generate SHAP plots (global, temporal, spatial)
shap = SHAPAnalyzer(
    model=ml.trained_models["XGBoost"],
    X_test=data.X_test,
    feature_names=data.feature_names,
    year_test=data.year_test,
    province_test=data.province_test
).run()
```

### Load Trained Models

```python
import joblib
import torch
from model import MLP

# Load scikit-learn models
model = joblib.load("./saved_models/XGBoost.pkl")
predictions = model.predict(data.X_test)

# Load PyTorch MLP
model = MLP()
model.load_state_dict(torch.load("./saved_models/PyTorch_MLP.pth"))
model.eval()
```

### Command Line

```bash
# Run full pipeline
python model.py

# Force retrain (clear existing models)
python model.py --force-retrain
```

---

## Model Performance

### Evaluation Metrics

| Metric | Description |
|--------|-------------|
| **Accuracy** | Overall classification correctness |
| **Weighted F1** | Harmonic mean of precision/recall (class-weighted) |
| **Macro AUC** | Area under ROC curve averaged across three classes |
| **Time(s)** | Training time in seconds |

### Results

| Model | Accuracy | Weighted F1 | Macro AUC | Training Time |
|-------|----------|-------------|-----------|---------------|
| Logistic Regression | 0.679 | 0.661 | 0.842 | 0.21s |
| Random Forest | 0.774 | 0.755 | 0.909 | 13.16s |
| XGBoost | 0.781 | 0.768 | 0.915 | 1.36s |
| Balanced XGBoost | **0.782** | **0.770** | **0.915** | 2.98s |
| PyTorch MLP | 0.773 | 0.764 | 0.909 | 45.20s |

### Key Insights

| Finding | Implication |
|---------|-------------|
| **XGBoost achieves best AUC (0.915)** | Gradient boosting captures complex nonlinear patterns in dietary data |
| **Balanced XGBoost improves F1 score** | Class-weight adjustment effectively handles imbalanced classes |
| **Random Forest achieves comparable performance** | Ensemble methods robust for tabular health data |
| **All models significantly outperform random baseline (0.333)** | Dietary features are highly discriminative |

---

## SHAP Interpretability

### Global Feature Importance (Urban Class)

Based on SHAP analysis of XGBoost model for Urban class (Class 2):

| Rank | Feature | Mean \|SHAP\| | Directional Effect |
|------|---------|---------------|-------------------|
| 1 | **Fat Energy Ratio** | **0.554** | Negative → Rural (higher fat predicts rural? *) |
| 2 | Carbohydrate Energy Ratio | 0.148 | Negative → Rural |
| 3 | Protein Energy Ratio | 0.144 | Negative → Rural |
| 4 | Province | 0.140 | Negative → Rural |
| 5 | Year | 0.100 | **Positive → Urban** |
| 6 | Fat-to-Carb Ratio | 0.095 | Negative → Rural |

*Note: Negative SHAP values indicate feature contributes to Rural classification, while positive contributes to Urban classification.*

**Key Findings:**
- **Fat Energy Ratio** is the most discriminative feature (mean |SHAP| = 0.554)
- **Year** is the only feature with positive contribution to Urban classification
- All other features show negative contributions (predicting Rural)

### Temporal Stratification (1991–2011)

SHAP values evolution for Urban class across 8 survey waves:

| Year | N | FatER | CarbER | ProtER | FCR | Year | Province |
|------|---|-------|--------|--------|-----|------|---------|
| 1991 | 2,646 | 0.467 | 0.134 | 0.124 | 0.095 | 0.130 | 0.148 |
| 1993 | 2,492 | 0.508 | 0.147 | 0.109 | 0.103 | 0.107 | 0.150 |
| 1997 | 2,509 | 0.537 | 0.145 | 0.133 | 0.107 | 0.122 | 0.148 |
| 2000 | 2,727 | 0.551 | 0.159 | 0.150 | 0.098 | 0.037 | 0.137 |
| 2004 | 2,332 | 0.581 | 0.160 | 0.132 | 0.094 | 0.034 | 0.134 |
| 2006 | 2,282 | 0.557 | 0.155 | 0.141 | 0.088 | 0.033 | 0.135 |
| 2009 | 2,326 | 0.604 | 0.142 | 0.164 | 0.090 | 0.131 | 0.135 |
| 2011 | 3,072 | **0.624** | 0.145 | **0.191** | 0.084 | **0.181** | 0.133 |

**Trend Analysis (1991 → 2011):**

| Feature | 1991 | 2011 | Change | Trend |
|---------|------|------|--------|-------|
| Fat Energy Ratio | 0.467 | 0.624 | **↑33.7%** | Increasing importance |
| Protein Energy Ratio | 0.124 | 0.191 | **↑53.7%** | Rapidly increasing importance |
| Year | 0.130 | 0.181 | **↑39.7%** | Growing temporal effect |
| Carbohydrate Energy Ratio | 0.134 | 0.145 | ↑8.4% | Slight increase |
| Fat-to-Carb Ratio | 0.095 | 0.084 | ↓10.9% | Declining importance |
| Province | 0.148 | 0.133 | ↓10.0% | Slight decline |

**Key Temporal Insights:**
- **Fat Energy Ratio** importance increased by 33.7% over 20 years
- **Protein Energy Ratio** importance surged by 53.7%, indicating growing urban-rural protein gap
- **Year** feature importance grew by 39.7%, reflecting accelerating dietary transition
- Provincial differences slightly diminished over time (↓10.0%)

### Spatial Stratification (12 Provinces)

Mean |SHAP| values by province for Urban class:

| Province | N | FatER | CarbER | ProtER | FCR | Year | Province | Top Feature |
|----------|---|-------|--------|-------|-----|------|---------|-------------|
| Beijing | 279 | **0.696** | 0.128 | 0.253 | 0.086 | 0.114 | 0.213 | FatER |
| Guangxi | 2,660 | **0.659** | 0.134 | 0.134 | 0.101 | 0.110 | 0.059 | FatER |
| Shanghai | 281 | **0.618** | 0.171 | 0.274 | 0.067 | 0.196 | 0.132 | FatER |
| Hunan | 2,248 | **0.606** | 0.131 | 0.163 | 0.086 | 0.106 | 0.101 | FatER |
| Jiangsu | 2,236 | **0.592** | 0.163 | 0.148 | 0.089 | 0.090 | 0.181 | FatER |
| Heilongjiang | 1,465 | **0.577** | 0.150 | 0.180 | 0.113 | 0.101 | 0.122 | FatER |
| Liaoning | 1,731 | **0.566** | 0.125 | 0.163 | 0.105 | 0.113 | 0.099 | FatER |
| Chongqing | 265 | **0.535** | 0.140 | 0.242 | 0.093 | 0.163 | 0.095 | FatER |
| Hubei | 2,196 | **0.526** | 0.176 | 0.158 | 0.090 | 0.093 | 0.125 | FatER |
| Guizhou | 2,474 | **0.492** | 0.123 | 0.110 | 0.091 | 0.090 | 0.100 | FatER |
| Shandong | 2,131 | **0.553** | 0.152 | 0.117 | 0.092 | 0.085 | **0.300** | FatER |
| Henan | 2,420 | **0.401** | **0.180** | 0.108 | 0.099 | 0.091 | 0.178 | FatER |

**Key Spatial Insights:**
- **Fat Energy Ratio** is the dominant feature in **all 12 provinces**
- **Beijing** shows the strongest FatER effect (0.696) — highest urban dietary pattern
- **Henan** shows the weakest FatER effect (0.401) — strongest rural dietary tradition
- **Shandong** has unusually high Province effect (0.300), indicating unique regional dietary characteristics
- **Shanghai** shows strong Year effect (0.196), reflecting rapid urbanization

### National Average |SHAP| (Urban Class)

| Feature | Mean |SHAP| |
|---------|---------------|
| Fat Energy Ratio | **0.554** |
| Carbohydrate Energy Ratio | 0.148 |
| Protein Energy Ratio | 0.144 |
| Province | 0.140 |
| Year | 0.100 |
| Fat-to-Carb Ratio | 0.095 |

### Class-Specific SHAP Analysis

| Feature | Class 0 (Rural) | Class 1 (Transitional) | Class 2 (Urban) |
|---------|-----------------|------------------------|-----------------|
| Fat Energy Ratio | **0.672** | **7.291** | **0.554** |
| Carbohydrate Ratio | 0.132 | 0.205 | 0.148 |
| Protein Ratio | 0.147 | 0.268 | 0.144 |
| Fat-to-Carb Ratio | 0.127 | 1.042 | 0.095 |
| Year | 0.067 | 0.075 | 0.100 |
| Province | 0.111 | 0.080 | 0.140 |

**Key Class-Specific Insights:**
- **Transitional class (Class 1)** shows extremely high SHAP values (7.29 for FatER), indicating this class is the most difficult to classify and represents a true mixed dietary pattern
- **Fat Energy Ratio** is the dominant feature across all three classes
- **Year** and **Province** are more important for Urban class classification

---

## Project Structure

```
dietary-pattern-xai/
│
├── data/
│   └── c12diet.sas7bdat              # CHNS dataset (101,926 samples)
│
├── figures/                           # Generated visualizations
│   ├── shap_summary.png               # Global SHAP summary plot
│   ├── shap_by_year.png               # Temporal stratification plot
│   └── shap_by_province.png           # Spatial SHAP heatmap
│
├── results/
│   └── model_results.csv              # Model performance metrics
│
├── saved_models/                      # Trained model files
│   ├── Logistic_Regression.pkl        # LR model (0.679 acc, 0.842 AUC)
│   ├── Random_Forest.pkl              # RF model (0.774 acc, 0.909 AUC)
│   ├── XGBoost.pkl                    # XGB model (0.781 acc, 0.915 AUC)
│   ├── Balanced_XGBoost.pkl           # Balanced XGB (0.782 acc, 0.915 AUC)
│   ├── PyTorch_MLP.pth                # MLP model (0.773 acc, 0.909 AUC)
│   └── scaler.pkl                     # StandardScaler for inference
│
├── model.py                           # Main training pipeline
├── analysis.py                        # Statistical analysis (ANOVA, trends)
├── requirements.txt                   # Python dependencies
└── README.md                          # This file
```

---

## Results

### Model Performance Summary

| Metric | Best Model | Value |
|--------|------------|-------|
| Accuracy | Balanced XGBoost | **0.782** |
| Weighted F1 | Balanced XGBoost | **0.770** |
| Macro AUC | XGBoost / Balanced XGBoost | **0.915** |
| Fastest Training | Logistic Regression | 0.21s |
| Most Robust | Balanced XGBoost | Best F1 score |

### Key Findings

1. **Fat Energy Ratio is the most discriminative feature** – Mean |SHAP| = 0.554 (Urban class)
2. **Temporal trends show increasing importance of fat and protein** – FatER importance ↑33.7%, ProtER ↑53.7% from 1991-2011
3. **Year effect is growing** – Year feature importance ↑39.7%, reflecting accelerating dietary transition
4. **Regional heterogeneity persists** – Beijing shows strongest urban pattern (FatER=0.696), Henan shows strongest rural pattern (FatER=0.401)
5. **Transitional class is most complex** – Extremely high SHAP values (7.29 for FatER) indicate true mixed dietary patterns
6. **Ensemble methods outperform linear models** – XGBoost and Random Forest achieve > 0.909 AUC

### Generated Outputs

| File | Description |
|------|-------------|
| `results/model_results.csv` | Model accuracy, F1, AUC, training time |
| `figures/shap_summary.png` | Global feature importance plot |
| `figures/shap_by_year.png` | SHAP values stratified by year (1991-2011) |
| `figures/shap_by_province.png` | Regional SHAP heatmap (12 provinces) |
| `saved_models/*.pkl` | Serialized trained models |

### Reproducibility

- **Random seed**: Fixed at 42 across all models
- **Single-thread configuration**: Ensures deterministic results
- **Stratified split**: Preserves class distribution in train/test sets
- **Model serialization**: All models saved for future inference

---



## License

MIT License – see [LICENSE](LICENSE) file for details.

---


## Acknowledgments

- China Health and Nutrition Survey (CHNS) for providing the dataset
- Carolina Population Center, University of North Carolina at Chapel Hill
- National Institute of Nutrition and Health, Chinese Center for Disease Control and Prevention

---

**Version**: 1.0  
**Last Updated**: March 2026  
**Status**: Production-ready
```
