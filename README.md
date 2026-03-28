# Urban-Rural Dietary Structure Disparity in China: A Statistical and Machine Learning Analysis

## Overview

This repository contains the source code and implementation details for the research paper **Urban-Rural Dietary Structure Disparity in China: A Statistical and Machine Learning Analysis**. The study leverages the China Health and Nutrition Survey (CHNS) dataset to explore dietary differences between urban and rural populations in China, using statistical analysis and machine learning classification models. The core objectives are to quantify dietary disparities, evaluate the discriminative power of dietary features for urban-rural classification, and identify key influential dietary factors via model interpretability techniques.

## Research Background

China has experienced profound nutritional transition driven by rapid urbanization and economic growth, leading to significant differences in dietary patterns between urban and rural populations. Understanding these disparities is critical for formulating targeted nutritional policies and addressing diet-related non-communicable diseases (NCDs). This study focuses on the structural composition of diets (macronutrient energy ratios) rather than absolute nutrient intakes, applying statistical and machine learning methods to validate and interpret dietary differences.

## Dataset

**Source**  
The study uses data from the China Health and Nutrition Survey (CHNS), a longitudinal household survey conducted collaboratively by the Carolina Population Center (University of North Carolina at Chapel Hill) and the National Institute of Nutrition and Health (Chinese Center for Disease Control and Prevention). Specifically, we employ the `c12diet` dataset, which provides three-day average nutrient intake estimates derived from 24-hour dietary recalls.

**Dataset Details**  
- Survey period: 1991–2011 (8 survey waves, as 1989 wave was excluded due to data quality concerns)  
- Total observations (after quality control): **98,247** (42.0% urban, 58.0% rural)  
- Key variables: Total energy intake (kcal), carbohydrate intake (g), fat intake (g), protein intake (g), and urban-rural classification (T2: 1 = urban, 2 = rural)

**Quality Control Criteria**  
- Exclude observations with missing values for key nutrient variables.  
- Restrict total daily energy intake to **500–5,000 kcal** to eliminate implausible records.

## Feature Construction

Four dietary features were constructed to characterize dietary structure (independent of total energy intake):

1. **Fat Energy Ratio (FatER)**: Proportion of total energy derived from fat (9 kcal/g).  
   $$\text{FatER} = \frac{\text{D3FAT} \times 9}{\text{D3KCAL}}$$

2. **Carbohydrate Energy Ratio (CarbER)**: Proportion of total energy derived from carbohydrates (4 kcal/g).  
   $$\text{CarbER} = \frac{\text{D3CARBO} \times 4}{\text{D3KCAL}}$$

3. **Protein Energy Ratio (ProtER)**: Proportion of total energy derived from protein (4 kcal/g).  
   $$\text{ProtER} = \frac{\text{D3PROTN} \times 4}{\text{D3KCAL}}$$

4. **Fat-to-Carbohydrate Ratio (FCR)**: Balance between fat and carbohydrate intake.  
   $$\text{FCR} = \frac{\text{D3FAT}}{\text{D3CARBO} + \epsilon}$$  
   (where $$\epsilon = 10^{-6}$$ to avoid division by zero).

## Methodology

The study integrates statistical analysis and machine learning, with the following key components:

### 1. Statistical Analysis
One-way Analysis of Variance (ANOVA) was used to test the significance of dietary differences between urban and rural groups. The F-statistic was calculated to compare between-group and within-group variance, with a p-value < 0.001 considered statistically significant. Analyses were conducted at both national and provincial levels, and temporal trends were examined across survey years.

### 2. Machine Learning Classification
A binary classification task was designed to predict urban/rural status using the four dietary features. The dataset was split into training (80%) and testing (20%) sets with stratification to preserve class distribution. Four models were trained and evaluated:

- **Logistic Regression (LR)**: Linear model with maximum 5,000 iterations for convergence.  
- **Random Forest (RF)**: Ensemble model with 300 trees and maximum depth of 6.  
- **XGBoost (XGB)**: Gradient-boosted tree model with 300 boosting rounds, max depth 4, and learning rate 0.1.  
- **Multi-Layer Perceptron (MLP)**: Neural network (PyTorch) with input layer (4 features), two hidden layers (16, 8 neurons with ReLU activation), and output layer (sigmoid activation for binary classification). Trained for 30 epochs with Adam optimizer (lr = 1e-3) and BCE loss.

### 3. Model Evaluation Metrics
- **Accuracy**: Proportion of correctly classified instances.  
- **AUC (Area Under ROC Curve)**: Primary metric (robust to class imbalance), measuring the model’s ability to distinguish between urban and rural classes.  
- **F1 Score**: Harmonic mean of precision and recall, useful for imbalanced datasets.

### 4. Model Interpretability with SHAP
SHapley Additive exPlanations (SHAP) was used to interpret the XGBoost model. SHAP values quantify the contribution of each feature to individual predictions, enabling identification of key discriminative features and their directional influence (positive = urban, negative = rural). SHAP analyses were also stratified by year and province to explore temporal and regional heterogeneity.

## Key Results

### Statistical Results (ANOVA)
All four dietary features showed highly significant differences between urban and rural groups (p < 0.001). The carbohydrate energy ratio (CarbER) was the most discriminative feature nationally (F = 9434.07). Provincial analyses revealed significant urban-rural differences in all 12 major provinces for most features, with Jiangsu and Henan showing the largest effect sizes (F > 1000). Temporal analysis showed that the urban-rural gap widened over time for FatER and FCR, while narrowing for CarbER.

| Feature | F-statistic | p-value | Significance |
|---------|-------------|--------|--------------|
| Fat Energy Ratio | 6976.89 | < 0.001 | *** |
| Carbohydrate Energy Ratio | 9434.07 | < 0.001 | *** |
| Protein Energy Ratio | 4969.10 | < 0.001 | *** |
| Fat-to-Carb Ratio | 6277.49 | < 0.001 | *** |

### Model Performance
All models achieved comparable performance with AUC values ranging from **0.696 to 0.709**. XGBoost slightly outperformed others in AUC (0.733) and F1 score (0.458), while Random Forest achieved the highest accuracy (0.703). The results indicate that dietary structure alone can moderately discriminate between urban and rural populations, supporting the validity of the constructed features.

| Model | Accuracy | F1 Score | AUC | Training Time (s) |
|-------|----------|---------|-----|-------------------|
| Logistic Regression | 0.699 | 0.393 | 0.702 | 0.03 |
| Random Forest | 0.703 | 0.413 | 0.709 | 12.65 |
| XGBoost | 0.712 | 0.458 | 0.733 | 0.41 |
| PyTorch MLP | 0.701 | 0.376 | 0.712 | — |

### SHAP Interpretation
The SHAP analysis identified **Carbohydrate Energy Ratio (CarbER)** and **Fat Energy Ratio (FatER)** as the most influential features:
- Higher CarbER → Negative SHAP values (predicts rural status).  
- Higher FatER → Positive SHAP values (predicts urban status).  
- Protein Energy Ratio and Fat-to-Carb Ratio also contributed but to a lesser extent.

Stratified SHAP analyses revealed:
- **By Year**: The importance of CarbER and FatER remained stable over time, but their marginal effects shifted, reflecting dietary convergence in some features.  
- **By Province**: SHAP values varied across provinces, with Henan and Jiangsu showing the strongest feature effects, indicating regional heterogeneity in dietary drivers.

## Repository Structure

```
├── data/
│   └── c12diet.sas7bdat          # CHNS dietary dataset
├── figures/                      # Generated plots (SHAP, trends, heatmaps)
├── results/                      # Model performance and statistical output
├── saved_models/                 # Trained models (.pkl for ML, .pth for MLP)
├── analysis.py                   # Statistical analysis script
├── model.py                      # Machine learning training pipeline
├── requirements.txt              # Python dependencies
└── README.md                     # This file
```

## Code Structure

The code is organized into modular classes for reusability and clarity:

- **`analysis.py`** – Performs descriptive statistics, ANOVA, and generates visualizations (heatmaps, bar charts, temporal trends, provincial comparisons).  
- **`model.py`** – Contains the full machine learning pipeline:  
  - `DataPipeline`: Loads and preprocesses data, constructs features, and splits data.  
  - `MLModels`: Trains and evaluates LR, RF, and XGBoost models.  
  - `TorchTrainer`: Trains the MLP neural network.  
  - `SHAPAnalyzer`: Computes SHAP values and generates interpretability plots.  
  - `Trainer`: Orchestrates the entire workflow.

## Requirements

Install the required packages using pip or conda:

```bash
pip install -r requirements.txt
```

Key dependencies:
- pandas, numpy, scikit-learn, xgboost, torch, shap, matplotlib, seaborn, joblib, pyreadstat

## Usage

### 1. Prepare the Dataset
Place the `c12diet.sas7bdat` file in the `./data` directory (create the directory if it does not exist).

### 2. Run Statistical Analysis
```bash
python analysis.py
```
This will generate:
- Summary statistics and ANOVA results in `./results/urban_rural_analysis.txt`
- Visualization files in `./figures/` (heatmaps, bar charts, provincial comparisons, temporal trends)

### 3. Run Machine Learning Training
```bash
python model.py
```
This will:
- Train all models (LR, RF, XGBoost, MLP) and save them to `./saved_models/`
- Output model performance metrics to `./results/model_results.csv`
- Generate SHAP plots in `./figures/` (`shap_summary.png`, `shap_by_year.png`, `shap_by_province.png`, `shap_province_ranking.png`)

### 4. Reuse Trained Models
You can load pre-trained models for further analysis or prediction:

```python
import joblib
model = joblib.load("./saved_models/XGBoost.pkl")
predictions = model.predict(data.X_test)
```

## License
This project is licensed under the MIT License – see the [LICENSE](LICENSE) file for details.
