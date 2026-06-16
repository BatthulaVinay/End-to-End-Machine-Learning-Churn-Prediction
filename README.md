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

```
## Dataset

Telecom customer churn dataset.

Target Variable:
- Churn (Yes/No)

Challenges:

- Significant class imbalance
- Mixed numerical and categorical features
- Business cost of false negatives

## Experiments

The objective was to maximize churn detection performance while maintaining a balance between precision and recall.

### Model Performance Comparison

| Model | Precision | Recall | ROC-AUC | F1 Score |
|---------|---------:|---------:|---------:|---------:|
| Logistic Regression | 0.237 | 0.928 | 0.816 | 0.378 |
| Logistic Regression + SMOTE | 0.250 | 0.907 | 0.811 | 0.392 |
| Decision Tree | 0.611 | 0.680 | 0.767 | 0.644 |
| Random Forest | 0.617 | 0.814 | 0.892 | 0.702 |
| XGBoost | **0.752** | 0.784 | **0.901** | **0.768** |


## Experimental Findings

### 1. Logistic Regression

Very high recall (92.8%) but extremely poor precision.

Pros:
- Rarely misses churners

Cons:
- Large number of false alarms

### 2. SMOTE

Improved precision slightly.

However:
- Minimal improvement overall
- Did not significantly improve ROC-AUC

### 3. Random Forest

Major performance jump.

- ROC-AUC increased to 0.892
- F1 increased to 0.702

### 4. XGBoost

Best overall model.

- Highest ROC-AUC
- Highest F1 Score
- Best precision-recall balance

## Failure Analysis

### False Positives

Customers predicted to churn but remained active.

Possible reasons:

- Temporary reduction in usage
- Seasonal behavior
- Insufficient behavioral features

### False Negatives

Customers predicted to stay but churned.

Possible reasons:

- External factors not present in dataset
- Competitor promotions
- Customer satisfaction issues not captured

### Class Imbalance Challenge

Initial models favored the majority class.

SMOTE was evaluated to mitigate imbalance but provided only marginal improvements.

### Model Tradeoff

Logistic Regression maximized recall but produced excessive false positives.

XGBoost achieved a more balanced business outcome.

## Best Model: XGBoost

Performance:

- Precision: 75.2%
- Recall: 78.4%
- ROC-AUC: 90.1%
- F1 Score: 76.8%

Confusion Matrix:

True Negatives: 545
False Positives: 25

False Negatives: 21
True Positives: 76

## Evaluation Methodology

Metrics Used:

- Precision
- Recall
- F1 Score
- ROC-AUC
- PR-AUC

Reason:

Accuracy alone is insufficient for imbalanced churn datasets.

Business impact is better reflected through precision-recall metrics.



