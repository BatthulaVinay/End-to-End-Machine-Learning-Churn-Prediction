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

E1 --> F[Evaluation]
E2 --> F
E3 --> F
E4 --> F
E5 --> F

F --> G[Model Comparison]

G --> H[Best Model Selection<br/>XGBoost]

````
## Dataset

Telecom customer churn dataset.

Target Variable:
- Churn (Yes/No)

Challenges:

- Significant class imbalance
- Mixed numerical and categorical features
- Business cost of false negatives

````
## Experiments (MOST IMPORTANT)

> **Note:** These are the actual empirical results obtained during the model evaluation phase.

| Model | Precision | Recall | ROC-AUC | F1-Score |
| :--- | :---: | :---: | :---: | :---: |
| **Logistic Regression** | 0.237 | **0.928** | 0.816 | 0.378 |
| **Logistic Regression + SMOTE** | 0.250 | 0.907 | 0.811 | 0.392 |
| **Decision Tree** | 0.611 | 0.680 | 0.767 | 0.644 |
| **Random Forest** | 0.617 | 0.814 | 0.892 | 0.702 |
| **XGBoost (Best Model)** | **0.752** | 0.784 | **0.901** | **0.768** |

### 📈 Key Insights from Results

1. **Why XGBoost Wins:** It achieves the highest **F1-Score (0.768)** and **ROC-AUC (0.901)**, proving it is the most robust model for balancing false positives and false negatives.
2. **The Precision-Recall Trade-off:** While *Logistic Regression* captures the most churners (Recall: 0.928), its abysmal Precision (0.237) means it suffers from a massive rate of false alarms. 
3. **Tree-Based Dominance:** Ensemble methods (Random Forest and XGBoost) significantly outperform linear baselines on this dataset.
