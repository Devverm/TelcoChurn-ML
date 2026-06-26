# TelcoChurn ML

A supervised machine learning project that predicts whether a telecom customer will churn (cancel their service), using the IBM Telco Customer Churn dataset. The pipeline covers end-to-end steps — from exploratory data analysis to a deployable predictive system.

---

## Table of Contents

- [Problem Statement](#problem-statement)
- [Dataset](#dataset)
- [Project Structure](#project-structure)
- [Workflow](#workflow)
- [Tech Stack](#tech-stack)
- [Installation](#installation)
- [Usage](#usage)
- [Model Performance](#model-performance)
- [Outputs](#outputs)
- [Future Work](#future-work)

---

## Problem Statement

Customer churn is one of the most costly challenges in the telecom industry. Acquiring a new customer costs significantly more than retaining an existing one. This project builds a classification model to identify customers at high risk of churning, enabling proactive retention strategies.

---

## Key Insights

### Dataset & Class Distribution
- The dataset contains **7,043 customers** with a **26.5% churn rate** (1,869 churned vs 5,174 retained), confirming significant class imbalance that required SMOTE correction.
- `TotalCharges` had **11 whitespace entries** masquerading as missing values — replaced with `0.0` before type conversion.

### EDA — What Drives Churn?

**Contract Type** is the single strongest churn signal:

| Contract | Churn Rate |
|---|---|
| Month-to-month | **42.7%** |
| One year | 11.3% |
| Two year | 2.8% |

Customers on month-to-month contracts churn at **15× the rate** of two-year contract holders.

**Tenure** — churned customers have far shorter lifetimes:

| Churn | Avg Tenure |
|---|---|
| No | 37.6 months |
| Yes | **18.0 months** |

Customers who stay longer are significantly more loyal; the first 12–18 months are the highest-risk window.

**Monthly Charges** — churned customers pay more on average:

| Churn | Avg Monthly Charge |
|---|---|
| No | $61.27 |
| Yes | **$74.44** |

Higher bills correlate with higher churn, suggesting price sensitivity as a key factor.

**Internet Service Type**:

| Service | Churn Rate |
|---|---|
| Fiber optic | **41.9%** |
| DSL | 19.0% |
| No internet | 7.4% |

Fiber optic customers churn at more than twice the rate of DSL users — possibly due to increased competition or higher pricing in that segment.

**Payment Method**:

| Payment Method | Churn Rate |
|---|---|
| Electronic check | **45.3%** |
| Mailed check | 19.1% |
| Bank transfer (auto) | 16.7% |
| Credit card (auto) | 15.2% |

Electronic check users churn at nearly **3× the rate** of automatic payment users, suggesting a link between payment friction and disengagement.

**Demographics**:
- **Senior citizens** churn at **41.7%** vs 23.6% for non-seniors — nearly double the rate.
- Customers **without a partner** churn at 33.0% vs 19.7% for those with a partner.
- Customers **without dependents** churn at 31.3% vs 15.5% for those with dependents.
- **Paperless billing** users churn at 33.6% vs 16.3% for paper billing users.

### Model Performance

**5-Fold Cross-Validation Accuracy (on SMOTE-balanced training data):**

| Model | CV Accuracy |
|---|---|
| Decision Tree | 77.78% ± 6.61% |
| XGBoost | 83.13% ± 8.32% |
| **Random Forest** | **84.08% ± 7.47%** |

**Random Forest — Test Set Results:**

| Metric | No Churn | Churn | Overall |
|---|---|---|---|
| Precision | 0.85 | 0.58 | — |
| Recall | 0.85 | 0.59 | — |
| F1-Score | 0.85 | 0.58 | — |
| **Accuracy** | — | — | **77.86%** |

The model performs well on the majority class (No Churn) but has room to improve on minority class recall — a known trade-off when dealing with imbalanced churn data.

### Top 10 Most Important Features (Random Forest)

| Rank | Feature | Importance |
|---|---|---|
| 1 | TotalCharges | 14.16% |
| 2 | MonthlyCharges | 13.65% |
| 3 | Contract | 12.66% |
| 4 | tenure | 12.18% |
| 5 | OnlineSecurity | 8.67% |
| 6 | TechSupport | 7.36% |
| 7 | PaymentMethod | 4.43% |
| 8 | OnlineBackup | 3.80% |
| 9 | Dependents | 3.04% |
| 10 | InternetService | 3.02% |

Billing-related features (`TotalCharges`, `MonthlyCharges`) and `tenure` dominate — consistent with the EDA findings. `OnlineSecurity` and `TechSupport` being in the top 6 suggests that value-added services play a meaningful role in retention.

---

## Dataset

**Source:** IBM Sample Dataset — [Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)  
**File:** `WA_Fn-UseC_-Telco-Customer-Churn.csv`

| Property | Details |
|---|---|
| Rows | 7,043 customers |
| Columns | 21 features |
| Target | `Churn` (Yes / No) |
| Class distribution | No Churn: 5,174 (73.5%) · Churn: 1,869 (26.5%) |

### Features

| Category | Features |
|---|---|
| Demographics | `gender`, `SeniorCitizen`, `Partner`, `Dependents` |
| Account info | `tenure`, `Contract`, `PaperlessBilling`, `PaymentMethod`, `MonthlyCharges`, `TotalCharges` |
| Phone services | `PhoneService`, `MultipleLines` |
| Internet services | `InternetService`, `OnlineSecurity`, `OnlineBackup`, `DeviceProtection`, `TechSupport`, `StreamingTV`, `StreamingMovies` |

---

## Project Structure

```
├── Customer_Churn_Prediction_using_ML.ipynb   # Main notebook
├── WA_Fn-UseC_-Telco-Customer-Churn.csv       # Raw dataset
├── customer_churn_model.pkl                   # Saved Random Forest model
├── encoders.pkl                               # Saved Label Encoders
└── README.md
```

---

## Workflow

### 1. Data Loading & Understanding
- Loaded the CSV into a Pandas DataFrame (7,043 rows × 21 columns)
- Dropped `customerID` (non-predictive identifier)
- Identified class imbalance: ~73.5% non-churn vs ~26.5% churn

### 2. Data Cleaning
- Detected and replaced whitespace entries in `TotalCharges` with `0.0`
- Cast `TotalCharges` from string to float
- Confirmed no null values remaining

### 3. Exploratory Data Analysis (EDA)
- **Numerical features** (`tenure`, `MonthlyCharges`, `TotalCharges`): distribution plots (histogram + KDE) with mean/median markers; box plots for outlier detection
- **Correlation heatmap** for numerical features
- **Categorical features**: count plots for all object-type columns and `SeniorCitizen`

### 4. Data Preprocessing
- Target encoded: `Yes → 1`, `No → 0`
- All categorical features encoded with `LabelEncoder`; encoders saved to `encoders.pkl` for inference reuse
- 80/20 train-test split (`random_state=42`)
- **SMOTE** (Synthetic Minority Oversampling Technique) applied to the training set to address class imbalance

### 5. Model Training
Three classifiers trained with default hyperparameters and evaluated via 5-fold cross-validation on the SMOTE-resampled training set:

| Model | CV Accuracy |
|---|---|
| Decision Tree | baseline |
| Random Forest | **highest** |
| XGBoost | competitive |

**Random Forest** selected as the final model based on cross-validation performance.

### 6. Model Evaluation
Final model evaluated on the held-out test set using:
- Accuracy Score
- Confusion Matrix
- Classification Report (Precision, Recall, F1-score)

### 7. Predictive System
A reusable inference pipeline:
1. Load saved model (`customer_churn_model.pkl`) and encoders (`encoders.pkl`)
2. Accept raw customer data as a dictionary
3. Apply the saved encoders for categorical transformation
4. Output: churn prediction (`Churn` / `No Churn`) + probability scores

---

## Tech Stack

| Library | Purpose |
|---|---|
| `pandas`, `numpy` | Data manipulation |
| `matplotlib`, `seaborn` | Visualization |
| `scikit-learn` | Preprocessing, model training, evaluation |
| `imbalanced-learn` | SMOTE oversampling |
| `xgboost` | Gradient boosting classifier |
| `pickle` | Model and encoder serialization |

---

## Installation

```bash
# Clone the repository
git clone https://github.com/<your-username>/customer-churn-prediction.git
cd customer-churn-prediction

# Install dependencies
pip install numpy pandas matplotlib seaborn scikit-learn imbalanced-learn xgboost
```

> **Note:** The notebook was originally developed on Google Colab. To run locally, update the dataset path from `/content/WA_Fn-UseC_-Telco-Customer-Churn.csv` to your local path.

---

## Usage

### Run the notebook
Open `Customer_Churn_Prediction_using_ML.ipynb` in Jupyter or Google Colab and run all cells.

### Run inference on new data
```python
import pickle
import pandas as pd

# Load model and encoders
with open("customer_churn_model.pkl", "rb") as f:
    model_data = pickle.load(f)

with open("encoders.pkl", "rb") as f:
    encoders = pickle.load(f)

loaded_model = model_data["model"]

# Provide customer data
input_data = {
    'gender': 'Female', 'SeniorCitizen': 0, 'Partner': 'Yes',
    'Dependents': 'No', 'tenure': 1, 'PhoneService': 'No',
    'MultipleLines': 'No phone service', 'InternetService': 'DSL',
    'OnlineSecurity': 'No', 'OnlineBackup': 'Yes', 'DeviceProtection': 'No',
    'TechSupport': 'No', 'StreamingTV': 'No', 'StreamingMovies': 'No',
    'Contract': 'Month-to-month', 'PaperlessBilling': 'Yes',
    'PaymentMethod': 'Electronic check', 'MonthlyCharges': 29.85,
    'TotalCharges': 29.85
}

input_df = pd.DataFrame([input_data])

for column, encoder in encoders.items():
    input_df[column] = encoder.transform(input_df[column])

prediction = loaded_model.predict(input_df)
probability = loaded_model.predict_proba(input_df)

print(f"Prediction: {'Churn' if prediction[0] == 1 else 'No Churn'}")
print(f"Probability: {probability}")
```

---

## Outputs

| File | Description |
|---|---|
| `customer_churn_model.pkl` | Trained Random Forest classifier + feature names |
| `encoders.pkl` | Dictionary of fitted `LabelEncoder` objects per categorical column |

---

## Future Work

- [ ] Hyperparameter tuning (GridSearchCV / RandomizedSearchCV)
- [ ] Explore downsampling strategies as an alternative to SMOTE
- [ ] Implement Stratified K-Fold cross-validation
- [ ] Address model overfitting (pruning, regularization)
- [ ] Try additional models (Logistic Regression, LightGBM, SVM)
- [ ] Build a web app (Streamlit / Flask) for real-time churn prediction
- [ ] Add feature importance visualizations

---
