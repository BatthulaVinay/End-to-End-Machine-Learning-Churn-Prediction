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


```mermaid
flowchart TD

A[Customer Dataset] --> B[Data Validation]

B --> C[Preprocessing]
C --> C1[Missing Values]
C --> C2[Encoding]
C --> C3[Scaling]

C --> D[Train/Test Split]
D --> E[Model Training]

E --> E1[Logistic Regression]
E --> E2[Logistic Regression + SMOTE]
E --> E3[Decision Tree]
E --> E4[Random Forest]
E --> E5[XGBoost]



F --> G[Model Comparison]

G --> H[Best Model Selection<br/>XGBoost]

````

