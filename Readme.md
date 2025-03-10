# Insurance Loss Prediction and Fraud Detection

This repository contains the implementation of **Insurance Loss Prediction and Fraud Detection** using **XGBoost, LightGBM, and Random Forest Regressor** models. The project leverages machine learning techniques for risk assessment and fraud detection in the insurance sector.

## Table of Contents

- [Insurance Loss Prediction and Fraud Detection](#insurance-loss-prediction-and-fraud-detection)
  - [Table of Contents](#table-of-contents)
  - [Introduction](#introduction)
  - [Dataset](#dataset)
  - [Models Used](#models-used)
  - [Installation](#installation)
  - [Project Structure](#project-structure)
  - [Usage](#usage)
    - [1. Data Preprocessing](#1-data-preprocessing)
    - [2. Model Training](#2-model-training)
    - [3. Model Evaluation](#3-model-evaluation)
    - [4. Hyperparameter Tuning](#4-hyperparameter-tuning)
  - [Results](#results)
  - [Future Improvements](#future-improvements)
  - [License](#license)
    - [Author](#author)

## Introduction
Insurance companies rely on machine learning to predict claim severity and detect fraudulent claims. This project implements **regression and classification models** for:
- **Insurance Loss Prediction** – Estimating claim severity based on past records.
- **Fraud Detection** – Identifying fraudulent claims using anomaly detection methods.

## Dataset
The project uses the **Allstate Claims Severity** dataset from Kaggle:
- Link: [Allstate Claims Severity](https://www.kaggle.com/c/allstate-claims-severity/data)
- The dataset contains **132 features** (categorical and numerical) and a target variable (**loss**).

## Models Used
The following models were implemented and evaluated:
1. **XGBoost** – Best performing model with the lowest RMSE and MAE.
2. **LightGBM** – Faster training but requires careful hyperparameter tuning to avoid overfitting.
3. **Random Forest Regressor** – Robust but slightly less accurate than boosting models.

## Installation
Ensure you have **Python 3.10+** installed. Run the following command to install dependencies:

```bash
pip install -r requirements.txt
```

Alternatively, install required libraries manually:
```bash
pip install xgboost lightgbm catboost scikit-learn pandas numpy matplotlib seaborn optuna category_encoders
```

## Project Structure
```
├── data/                         # Contains dataset (CSV format)
├── models/                       # Trained model files
├── notebooks/                    # Jupyter notebooks for experimentation
├── scripts/                      # Python scripts for preprocessing, training, and evaluation
│   ├── preprocess.py             # Data preprocessing (encoding, scaling, feature selection)
│   ├── train.py                  # Model training (XGBoost, LightGBM, Random Forest)
│   ├── evaluate.py               # Model evaluation (RMSE, MAE, R²)
│   ├── hyperparameter_tuning.py  # Hyperparameter tuning using Optuna
├── README.md                     # Project documentation
├── requirements.txt               # Dependencies
└── LICENSE                        # License file
```

## Usage
### 1. Data Preprocessing
```bash
python scripts/preprocess.py
```
This step includes:
- Encoding categorical features
- Scaling numerical data
- Outlier removal
- Feature engineering (Polynomial Features, PCA)

### 2. Model Training
Train the models using:
```bash
python scripts/train.py
```
This trains **XGBoost, LightGBM, and Random Forest** models on the preprocessed data.

### 3. Model Evaluation
Evaluate model performance:
```bash
python scripts/evaluate.py
```
This step calculates **RMSE, MAE, and R² scores** for comparison.

### 4. Hyperparameter Tuning
To optimize the models using **Optuna**:
```bash
python scripts/hyperparameter_tuning.py
```

## Results
| Model                  | RMSE  | MAE   | R² Score |
|------------------------|-------|-------|----------|
| **XGBoost**           | 1451  | 1030  | 0.630    |
| **LightGBM**          | 1828  | 1177  | 0.604    |
| **Random Forest**     | 1917  | 1281  | 0.564    |

XGBoost achieved the best results in terms of **lowest RMSE and highest R² score**, making it the recommended model for deployment.

## Future Improvements
- **Experiment with deep learning models** (e.g., LSTMs for sequential data analysis).
- **Improve feature selection** using SHAP or other explainability techniques.
- **Deploy the model** as a web API using Flask or FastAPI.

## License
This project is licensed under the **MIT License**. See `LICENSE` for details.

---
### Author
**Nirusan Hariharan**  
University of Westminster | IIT Sri Lanka

