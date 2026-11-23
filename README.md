# Telecom Customer Churn & Revenue Impact Analysis

**Live Streamlit App:** `[Paste your Streamlit Cloud URL here after deploying]`

## 1. Problem
In telecom retention, missing an actual churner (false negative) is costlier than reaching out to a loyal customer (false positive). Because churn is imbalanced (~`[26.5%]` of customers), accuracy alone is misleading. This project prioritizes **Recall** and **PR-AUC** so retention teams catch high-risk subscribers while ranking outreach by expected revenue at risk.

## 2. Data
- **Dataset:** BlastChar Telco Customer Churn (`WA_Fn-UseC_-Telco-Customer-Churn.csv` on [Kaggle](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)).
- **Size & Target:** `[7,043]` rows and `[21]` raw columns (`[7,021]` rows after dropping `[22]` duplicate rows post-`customerID` removal, or `[7,043]` if kept); baseline churn rate is `[~26.5%]`.
- **Note:** The raw CSV is excluded from the repository; download it from Kaggle and place it at `data/WA_Fn-UseC_-Telco-Customer-Churn.csv` to reproduce the notebook.

## 3. Approach
- **Data Cleaning:** Cast `TotalCharges` to numeric (`errors="coerce"`), found `[11]` missing entries where `tenure == 0` (new customers prior to a full billing cycle), and imputed `0`.
- **Leakage Prevention:** Split data into train and test sets (`[80/20]`, stratified) **before** fitting preprocessing transformers (`ColumnTransformer` with scaling and one-hot encoding).
- **Modeling & Threshold Tuning:** Trained Logistic Regression and XGBoost pipelines, evaluated Recall and PR-AUC, and tuned the classification threshold using training cross-validation rather than peeking at the test set.
- **Explainability:** Used SHAP to inspect global feature importance and directional effects on predicted churn risk.

## 4. Results
Replace the bracketed metrics below with your test-set scores:

| Model | Threshold | Recall | PR-AUC | F1 / ROC-AUC |
| :--- | :---: | :---: | :---: | :---: |
| Logistic Regression | `[0.50 / tuned]` | `[0.XX]` | `[0.XX]` | `[0.XX / 0.XX]` |
| XGBoost | `[tuned val]` | `[0.XX]` | `[0.XX]` | `[0.XX / 0.XX]` |

- **Confusion Matrix Takeaway:** `[Summarize false negatives vs. false positives on the test set and how threshold tuning improved churner capture.]`

## 5. Key Drivers & Recommendations
Top predictors from the SHAP summary/bar plots (predictive associations, not proven causal effects):

1. **`[Top Feature 1, e.g., Contract_Month-to-month]`:** `[How it shifts churn risk in the SHAP plot]` $\rightarrow$ `[Actionable retention offer or contract incentive]`.
2. **`[Top Feature 2, e.g., tenure]`:** `[How it shifts churn risk in the SHAP plot]` $\rightarrow$ `[Onboarding or early-lifecycle intervention]`.
3. **`[Top Feature 3, e.g., MonthlyCharges or InternetService_Fiber optic]`:** `[How it shifts churn risk in the SHAP plot]` $\rightarrow$ `[Targeted pricing, bundle, or support check]`.

## 6. Business Impact
Ranking test customers by 12-month expected value at risk (`churn_prob * MonthlyCharges * 12`) and targeting the **top 20%** cohort captures `[XX%]` of all test churners.

Simulated net savings across retention scenarios (using `1 USD = 3.64 QAR`):

| Scenario | Retention Rate | Net Savings (USD) | Net Savings (QAR) |
| :--- | :---: | :---: | :---: |
| Pessimistic | 10% | `$[X,XXX]` | `QAR [X,XXX]` |
| Realistic | 20% | `$[X,XXX]` | `QAR [X,XXX]` |
| Optimistic | 30% | `$[X,XXX]` | `QAR [X,XXX]` |

- **Assumptions:** `[1,409]` test customers (or exact `len(X_test)`), `12`-month revenue horizon, `$10` retention outreach cost per contacted customer, `10% / 20% / 30%` retained share of actual churners in the top 20% cohort, and `1 USD = 3.64 QAR`.
- **Disclaimer:** Figures reflect a simulated test-set scenario, not guaranteed or realized revenue.

## 7. Limitations
- **Dataset Scope:** Built on a public telecom benchmark dataset rather than live carrier billing, network quality, or regional subscriber data.
- **Correlation vs. Causation:** SHAP explains how features influence model predictions; interventions require A/B testing or causal evaluation to verify real lift.

## 8. Live Demo & Local Setup
- **Live App:** `[Paste your Streamlit Cloud URL here after deploying]`
- **Local Setup:**
  ```powershell
  python -m venv .venv
  .\.venv\Scripts\Activate.ps1
  pip install -r requirements.txt
  streamlit run app.py