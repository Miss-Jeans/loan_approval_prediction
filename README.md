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
- **Imbalanced Data Handling:** Address class imbalance (where non-default applications significantly outnumber defaults) during cross-validation and evaluation to ensure high sensitivity toward potential default cases.
- **Model Training & Tuning:** Built using `GradientBoostingClassifier` with extensive **GridSearchCV** hyperparameter optimization.
- **Data Preprocessing & Validation:** Implemented an automated object-oriented `Model` wrapper class to handle input sanitization, data alignment, and dynamic missing value imputation at inference time.
- **Interactive Web App:** Deployed with **Dash** and **Plotly** on **Render** (served via **Gunicorn**), offering executive visual analytics and interactive parameter inputs.
