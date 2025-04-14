# Credit Card Fraud Detection Pipeline

This project presents a complete pipeline for detecting fraudulent credit card transactions using machine learning. The dataset used is highly imbalanced, which presents unique challenges that are addressed through careful preprocessing, resampling techniques, and evaluation.

## Dataset

- **Source:** [Kaggle - Credit Card Fraud Detection](https://www.kaggle.com/mlg-ulb/creditcardfraud)
- **Size:** 284,807 transactions
- **Features:** 30 anonymized features (`V1` to `V28`), along with `Time`, `Amount`, and `Class` (target variable: 0 = Non-Fraud, 1 = Fraud)

## Project Pipeline

### 1. Library Imports
Included libraries for:
- Data processing: `pandas`, `numpy`
- Visualization: `matplotlib`, `seaborn`
- Machine learning models and utilities: `sklearn`
- Handling class imbalance: `imblearn`
- Runtime analysis: `time`

### 2. Exploratory Data Analysis (EDA)
- Inspected dataset structure and basic statistics
- Checked for null values and outliers
- Investigated class distribution and correlation between features

### 3. Data Preprocessing
- Standardized `Amount` and `Time` features
- Handled data imbalance using SMOTE (Synthetic Minority Over-sampling Technique)
- Split the data into training and testing sets (stratified to maintain class proportions)

### 4. Model Training
Trained multiple machine learning models, including:
- Logistic Regression
- Decision Tree
- Random Forest
- XGBoost
- LightGBM

### 5. Model Evaluation
- Evaluated models using accuracy, precision, recall, F1-score, ROC-AUC
- Special focus on recall due to the importance of detecting fraud
- Compared performance before and after applying SMOTE

### 6. Results
- LightGBM and XGBoost showed the highest performance in detecting fraud
- Significant improvements in recall with SMOTE applied
- ROC-AUC scores were consistently high for ensemble models

## File Structure

```
credit-card-fraud-detection/
│
├── fraud_detection_pipeline.ipynb     # Jupyter notebook with full workflow
├── creditcard.csv                     # Dataset
├── README.md                          # Project overview and documentation
```

## Conclusion

This project highlights the importance of preprocessing and proper evaluation when dealing with imbalanced classification problems. With appropriate resampling and tuning, machine learning models can effectively detect rare fraudulent transactions.
