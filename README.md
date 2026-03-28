Urban-Rural Dietary Structure Disparity in China: A Statistical and Machine Learning Analysis

Overview

This repository contains the source code and implementation details for the research paper Urban-Rural Dietary Structure Disparity in China: A Statistical and Machine Learning Analysis. The study leverages the China Health and Nutrition Survey (CHNS) dataset to explore dietary differences between urban and rural populations in China, using statistical analysis and machine learning classification models. The core objectives are to quantify dietary disparities, evaluate the discriminative power of dietary features for urban-rural classification, and identify key influential dietary factors via model interpretability techniques.

Research Background

China has experienced profound nutritional transition driven by rapid urbanization and economic growth, leading to significant differences in dietary patterns between urban and rural populations. Understanding these disparities is critical for formulating targeted nutritional policies and addressing diet-related non-communicable diseases (NCDs). This study focuses on the structural composition of diets (macronutrient energy ratios) rather than absolute nutrient intakes, applying statistical and machine learning methods to validate and interpret dietary differences.

Dataset

Source

The study uses data from the China Health and Nutrition Survey (CHNS), a longitudinal household survey conducted collaboratively by the Carolina Population Center (University of North Carolina at Chapel Hill) and the National Institute of Nutrition and Health (Chinese Center for Disease Control and Prevention). Specifically, we employ the c12diet dataset, which provides three-day average nutrient intake estimates derived from 24-hour dietary recalls.

Dataset Details

- Survey period: 1989–2011 (9 survey waves)

- Total observations (after quality control): 98,247 (42.0% urban, 58.0% rural)

- Key variables: Total energy intake (kcal), carbohydrate intake (g), fat intake (g), protein intake (g), and urban-rural classification (T2: 1 = urban, 2 = rural)

Quality Control Criteria

- Exclude observations with missing values for key nutrient variables.

- Restrict total daily energy intake to 500–5,000 kcal to eliminate implausible records.

Feature Construction

Four dietary features were constructed to characterize dietary structure (independent of total energy intake):

1. Fat Energy Ratio (FatER): Proportion of total energy derived from fat (9 kcal/g). Formula:$$\text{FatER} = \frac{\text{D3FAT} \times 9}{\text{D3KCAL}}$$

2. Carbohydrate Energy Ratio (CarbER): Proportion of total energy derived from carbohydrates (4 kcal/g). Formula: $$\text{CarbER} = \frac{\text{D3CARBO} \times 4}{\text{D3KCAL}}$$

3. Protein Energy Ratio (ProtER): Proportion of total energy derived from protein (4 kcal/g). Formula: $$\text{ProtER} = \frac{\text{D3PROTN} \times 4}{\text{D3KCAL}}$$

4. Fat-to-Carbohydrate Ratio (FCR): Balance between fat and carbohydrate intake. Formula: $$\text{FCR} = \frac{\text{D3FAT}}{\text{D3CARBO} + \epsilon}$$ (where $$\epsilon = 10^{-6}$$ to avoid division by zero).

Methodology

The study integrates statistical analysis and machine learning, with the following key components:

1. Statistical Analysis

One-way Analysis of Variance (ANOVA) was used to test the significance of dietary differences between urban and rural groups. The F-statistic was calculated to compare between-group and within-group variance, with a p-value < 0.001 considered statistically significant.

2. Machine Learning Classification

A binary classification task was designed to predict urban/rural status using the four dietary features. The dataset was split into training (80%) and testing (20%) sets with stratification to preserve class distribution. Four models were trained and evaluated:

- Logistic Regression (LR): Linear model with maximum 5,000 iterations for convergence.

- Random Forest (RF): Ensemble model with 300 trees and maximum depth of 6.

- XGBoost (XGB): Gradient-boosted tree model with 300 boosting rounds, max depth 4, and learning rate 0.1.

- Multi-Layer Perceptron (MLP): Neural network (PyTorch) with input layer (4 features), two hidden layers (16, 8 neurons with ReLU activation), and output layer (sigmoid activation for binary classification). Trained for 30 epochs with Adam optimizer (lr = 1e-3) and BCE loss.

3. Model Evaluation Metrics

- Accuracy: Proportion of correctly classified instances.

- AUC (Area Under ROC Curve): Primary metric (robust to class imbalance), measuring the model’s ability to distinguish between urban and rural classes.

4. Model Interpretability with SHAP

SHapley Additive exPlanations (SHAP) was used to interpret the XGBoost model. SHAP values quantify the contribution of each feature to individual predictions, enabling identification of key discriminative features and their directional influence (positive = urban, negative = rural).

Code Structure

The code is organized into modular classes for reusability and clarity. Below is a breakdown of key components:

1. DataPipeline Class

Loads the CHNS c12diet dataset, performs preprocessing (handling missing values, filtering energy intake), constructs dietary features, and splits data into training/testing sets.

2. MLModels Class

Trains and evaluates the three traditional machine learning models (LR, RF, XGBoost), saves trained models to local storage, and loads historical results to avoid redundant training.

3. MLP and TorchTrainer Classes

Defines the MLP neural network architecture and handles training/evaluation using PyTorch, including dataset preparation, device configuration (CPU/GPU), and loss optimization.

4. SHAPAnalyzer Class

Computes SHAP values for the XGBoost model, generates a summary plot to visualize feature importance and directional influence, and saves the plot to the ./figures directory.

5. Trainer Class

Orchestrates the entire workflow: data loading, model training (ML and MLP), result aggregation, SHAP analysis, and saving of results/models.

Requirements

Install the required packages using pip or conda:


# Required packages
pandas>=2.0.0
numpy==1.26.4  # Compatible with all dependencies
scikit-learn>=1.3.0
xgboost>=2.0.0
torch>=2.0.0
shap>=0.45.0
matplotlib>=3.7.0
joblib>=1.3.0
sas7bdat>=2.2.0  # For reading SAS dataset
    

Usage

1. Prepare the Dataset

Place the c12diet.sas7bdat file in the ./data directory (create the directory if it does not exist).

2. Run the Training Pipeline

Execute the main script to start the entire workflow (data loading, model training, SHAP analysis, result saving):


python analysis.py


3. Outputs

After running the script, the following directories and files will be generated:

- ./results/: Contains model_results.csv with performance metrics (Accuracy, AUC) for all models.

- ./saved_models/: Saves trained models (LR, RF, XGBoost, MLP) as .pkl (ML models) and .pth (MLP model) files.

- ./figures/: Contains the SHAP summary plot (shap_summary.png) visualizing feature importance.

4. Reuse Trained Models

The MLModels.load_saved_models() method can be used to load pre-trained models for further analysis or prediction:


# Example: Load trained models
from analysis import MLModels, DataPipeline

# Load data
data = DataPipeline().load()

# Load saved models
ml = MLModels(data.X_train, data.X_test, data.y_train, data.y_test)
trained_models = ml.load_saved_models()

# Use XGBoost model for prediction
xgb_model = trained_models["XGBoost"]
y_pred = xgb_model.predict(data.X_test)
    

Key Results

Statistical Results (ANOVA)

All four dietary features showed highly significant differences between urban and rural groups (p < 0.001), with the carbohydrate energy ratio (CarbER) being the most discriminative feature (F-statistic = 9434.07).

Model Performance

All models achieved comparable performance with AUC values ranging from 0.696 to 0.700:

- Random Forest: AUC = 0.700, Accuracy = 0.699

- XGBoost: AUC = 0.698, Accuracy = 0.697

- PyTorch MLP: AUC = 0.697, Accuracy = 0.698

- Logistic Regression: AUC = 0.696, Accuracy = 0.698

SHAP Interpretation

The SHAP analysis identified Carbohydrate Energy Ratio (CarbER) and Fat Energy Ratio (FatER) as the most influential features:
    Higher CarbER → Negative SHAP values (predicts rural status).Higher FatER → Positive SHAP values (predicts urban status).CitationIf you use this code or dataset in your research, please cite the following paper:
