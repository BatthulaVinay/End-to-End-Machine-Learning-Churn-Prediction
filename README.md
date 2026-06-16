# Customer Churn Prediction System

Built an end-to-end machine learning pipeline to predict customer churn and compare multiple classification approaches on an imbalanced dataset.

The project focuses on:

- Churn prediction
- Class imbalance handling
- Model comparison
- Precision–Recall tradeoffs
- Business-oriented evaluation metrics

Goal:

Identify customers likely to churn while minimizing false predictions and maximizing actionable business insights.


Customer Dataset
       │
       ▼
Data Validation
       │
       ▼
Preprocessing
(Missing Values,
Encoding,
Scaling)
       │
       ▼
Train/Test Split
       │
       ▼
Model Training
 ├─ Logistic Regression
 ├─ Logistic Regression + SMOTE
 ├─ Decision Tree
 ├─ Random Forest
 └─ XGBoost
       │
       ▼
Evaluation
(Recall, Precision,
F1, ROC-AUC, PR-AUC)
       │
       ▼
Model Comparison
       │
       ▼
Best Model Selection
(XGBoost)
