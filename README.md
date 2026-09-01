#  Loan Approval Prediction & Executive Dashboard

[![Live App](https://img.shields.io/badge/Render-Live_Demo-brightgreen?style=for-the-badge&logo=render)](https://loan-approval-prediction-yx3l.onrender.com/)
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?style=for-the-badge&logo=python)](https://www.python.org/)
[![Dash](https://img.shields.io/badge/Dash-Plotly-100000?style=for-the-badge&logo=plotly)](https://dash.plotly.com/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)

An end-to-end Machine Learning web application and interactive dashboard designed to evaluate credit risk and predict loan approval probabilities in real time.

>  **Live Demo:** [https://loan-approval-prediction-yx3l.onrender.com/](https://loan-approval-prediction-yx3l.onrender.com/)

---

##  Executive Summary

Whether funding higher education, refinancing existing debt, or launching a business venture, individuals frequently rely on financial institutions (banks, credit unions, and micro-finance lenders) for capital.

For financial institutions, approving or rejecting these applications is a critical balance between growth and risk management. Key decisions rely on multi-faceted applicant profiles—including credit history, prior defaults, income ratios, and employment stability. 

Rather than relying on manual, discretionary, or arbitrary decision-making, financial institutions require a **robust, automated algorithmic pipeline** to:
- **Minimize Default Risk:** Accurately identify high-risk applicants before credit extension.
- **Maximize Profitability:** Maintain steady approval rates for low-risk, qualified borrowers.
- **Ensure Scalability & Reliability:** Serve real-time evaluations consistently for daily high-volume applications.

---

##  Key Technical Highlights & Pipeline

- **Dataset:** Sourced from Kaggle ([Loan Approval Prediction Dataset](https://www.kaggle.com/)).
- **Imbalanced Data Handling:** Addressed class imbalance within the `loan_status` target variable during cross-validation and evaluation to ensure high sensitivity toward identifying true loan approval cases without compromising precision.
- **Model Training & Tuning:** Built using `GradientBoostingClassifier` with extensive **GridSearchCV** hyperparameter optimization.
- **Data Preprocessing & Validation:** Implemented an automated object-oriented `Model` wrapper class to handle input sanitization, data alignment, and dynamic missing value imputation at inference time.
- **Interactive Web App:** Deployed with [Dash](https://dash.plotly.com/) and Plotly on [Render](https://render.com/) (served via **Gunicorn**), offering executive visual analytics and interactive parameter inputs.

- ## Model Performance & Diagnostic Evaluation

To verify that the tuned `GradientBoostingClassifier` generalizes well to unseen data and does not suffer from overfitting, performance was evaluated across both the training ($N = 26,064$) and test ($N = 6,517$) splits.

### Performance Summary & Overfit Check

| Metric | Training Set | Test Set | Evaluation / Diagnosis |
| :--- | :---: | :---: | :--- |
| **Accuracy** | **89.0%** | **89.0%** | **Optimal Generalization:** Identical score indicates zero variance loss. |
| **ROC-AUC Score** | **0.920** | **0.920** | **Strong Discriminative Power:** Excellent class separation across all thresholds. |
| **Approval Recall (`True`)** | **76.0%** | **76.0%** | **High Sensitivity:** Captures 76% of eligible loan approvals accurately. |
| **Approval Precision (`True`)** | **73.0%** | **74.0%** | **Stable Precision:** Minimal variation (+1% on test set) for approval calls. |
| **Approval F1-Score (`True`)** | **0.75** | **0.75** | **Balanced Trade-off:** Consistent harmonic mean on minority/approval class. |

---

### Detailed Classification Reports

#### Training Set Performance
```text
              precision    recall  f1-score   support

       False       0.93      0.92      0.93     20401
        True       0.73      0.76      0.75      5663

    accuracy                           0.89     26064
   macro avg       0.83      0.84      0.84     26064
weighted avg       0.89      0.89      0.89     26064

Model ROC-AUC: 0.92

```
#### Test Set Performance
```text
              precision    recall  f1-score   support

       False       0.93      0.92      0.93     5072
        True       0.74      0.76      0.75     1445

    accuracy                           0.89     6517
   macro avg       0.83      0.84      0.84     6517
weighted avg       0.89      0.89      0.89     6517

Model ROC-AUC: 0.92
