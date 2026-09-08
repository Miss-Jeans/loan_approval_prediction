#  CREDIT OPTIMIZATION AND A/B TESTING DASHBOARD

![MongoDB](https://img.shields.io/badge/MongoDB-47A248?style=for-the-badge&logo=mongodb&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-%23F7931E.svg?style=for-the-badge&logo=scikit-learn&logoColor=white)
![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
![Dash](https://img.shields.io/badge/Dash-0081C8?style=for-the-badge&logo=plotly&logoColor=white)
![Pandas](https://img.shields.io/badge/pandas-%23150458.svg?style=for-the-badge&logo=pandas&logoColor=white)
![Render](https://img.shields.io/badge/Render-%2346E3B7.svg?style=for-the-badge&logo=render&logoColor=white)

An end-to-end Machine Learning web application and executive dashboard designed to evaluate credit risk, simulate decision-boundary threshold tuning and run A/B testing on medium-risk loan applicants in real time.
>  **Live Demo:** [https://loan-approval-prediction-yx3l.onrender.com/](https://loan-approval-prediction-yx3l.onrender.com/)

---

##  Executive Summary
Whether funding higher education, refinancing existing debt or launching a business venture, individuals frequently rely on financial institutions (banks, credit unions and microfinance lenders) for capital.

For financial institutions, approving or rejecting these applications is a critical balance between portfolio growth and risk management. Key decisions rely on multi-faceted applicant profiles—including credit history, prior defaults, income ratios and employment stability.

Rather than relying on static or discretionary decision-making, this project implements a production-grade algorithmic pipeline and experimental framework to:

* **Minimize Default Risk:** Accurately identify high-risk applicants before credit extension using a tuned Gradient Boosting Classifier.
* **Maximize Portfolio Profitability:** Re-evaluate previously rejected medium-risk borrowers using dynamic threshold tuning and financial return-on-investment (ROI) modeling.
* **Validate Decisions Statistically:** Perform dynamic Chi-Square ($\chi^2$) test-of-independence evaluations on experimental A/B variants.
* **Ensure End-to-End Scalability:** Stream documents from MongoDB through a Python processing engine to a Dash/Plotly interactive web dashboard hosted on Render.

---

##  Key Technical Highlights & Pipeline

- **Dataset:** Sourced from Kaggle ([Loan Approval Prediction Dataset](https://www.kaggle.com/)).
- **Imbalanced Data Handling:** Addressed class imbalance within the `loan_status` target variable during cross-validation and evaluation to ensure high sensitivity toward identifying true loan approval cases without compromising precision.
- **Model Training & Hyperparameter Tuning:** Built using `GradientBoostingClassifier` with extensive **GridSearchCV** hyperparameter optimization.
- **Object-Oriented Pipeline Architecture:** Implemented modular Python wrappers (`Model`, `Repository`, `Experiment`) handling automated data preprocessing, input alignment, dynamic thresholding and risk calculation.
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

#### Training Set Performance ($N=26,064$)
```text
              precision    recall  f1-score   support

       False       0.93      0.92      0.93     20401
        True       0.73      0.76      0.75      5663

    accuracy                           0.89     26064
   macro avg       0.83      0.84      0.84     26064
weighted avg       0.89      0.89      0.89     26064

Model ROC-AUC: 0.92

```
#### Test Set Performance ($N=6,517$)
```text
              precision    recall  f1-score   support

       False       0.93      0.92      0.93     5072
        True       0.74      0.76      0.75     1445

    accuracy                           0.89     6517
   macro avg       0.83      0.84      0.84     6517
weighted avg       0.89      0.89      0.89     6517

Model ROC-AUC: 0.92
```

## A/B Testing Experiment

### 1. Data Pipeline & Architecture
* **Data Ingestion (Phase I):** Raw applicant demographics and loan performance datasets were imported into **MongoDB** as document collections to enable scalable querying and persistent state management across the modeling pipeline.
* **Database Repository Layer:** A dedicated `Repository` module executes aggregation queries against MongoDB to extract candidate pools, assign experimental groups, and stream records into pandas DataFrames for feature engineering.

---

### 2. Problem Context & Risk Stratification
During predictive modeling, analysis revealed a high loan rejection rate under the baseline criteria (**{threshold} > 0.50**). Evaluating default risk across loan grades (**A through G**) established distinct risk tiers:

| Loan Grade | Default Risk (%) | Risk Category | Pipeline Action |
|:-----------| :--- | :--- | :--- |
| **A**      | $9\%$ | Low Risk | Standard Approval |
| **B**      | $16\%$ | Low Risk | Standard Approval |
| **C**      | $20\%$ | Marginal / Boundary | Standard Approval |
| **D**      | $59\%$ | Medium Risk | **Selected for A/B Testing** |
| **E**      | $64\%$ | Medium Risk | **Selected for A/B Testing** |
| **F**      | $70\%$ | High Risk | Automatic Rejection |
| **G**      | $98\%$ | High Risk | Automatic Rejection |

* **Target Experimental Population:** Medium-risk applicants in **Grades D and E** represent candidates historically rejected under strict baseline thresholds who offer potential portfolio expansion.

---

### 3. A/B Experiment Setup
Demographic records pulled from MongoDB for grades D and E were split into two experimental variants:

* **Control Group:** Vetted against baseline approval criteria ($\text{Predicted Rejection Risk} > 0.50$).
* **Treatment Group:** Vetted against an adjusted risk threshold ($\text{Predicted Rejection Risk} \le 0.15$) to reconsider previously rejected candidates.

---

### 4. Financial Optimization & ROI Analysis
Because financial institutions operate on profit preservation and capital growth, adjusting thresholds must be financially viable. The system tracks:
1. **Reconsidered Approvals:** Volume of previously rejected treatment applicants approved under the adjusted threshold ($\le 0.15$).
2. **Net ROI & Profitability:**
   $$\text{Net Expected Profit} = \text{Expected Gross Interest Income} - \text{Expected Default Loss}$$
   * **Gross Interest Income:** Portfolio volume x average interest rate on reconsidered loans.
   * **Expected Default Loss:** Modeled by weighting individual loan amounts against predicted rejection probabilities ($P(\text{reject})$).

---

### 5. Statistical Validation
A **Chi-Square Test of Independence** ($\chi^2$) is executed dynamically on the contingency table comparing Control vs. Treatment outcomes to verify if threshold adjustments yield a statistically significant difference ($p \le 0.05$).

---

## Tech Stack & Dependencies

* **Language & Frameworks:** Python 3.11+, Dash, Plotly, Flask, Gunicorn
* **Data Processing & ML:** Pandas, NumPy, Scikit-learn, Statsmodels, Category Encoders, Joblib
* **Database & Infrastructure:** MongoDB (PyMongo), PyArrow, Render Cloud Platform

---

## Local Development Setup

To run this application locally on your machine:

### Clone Repository & Set Environment Variables
```bash
git clone [https://github.com/Miss-Jeans/loan_approval_prediction.git](https://github.com/Miss-Jeans/loan_approval_prediction.git)
cd loan_approval_prediction

