import joblib
import pandas as pd
import streamlit as st

model = joblib.load("churn_model.pkl")
defaults = joblib.load("defaults.pkl")

st.set_page_config(page_title="Customer Churn Predictor", layout="centered")
st.title("Customer Churn Predictor")
st.caption(
    "Enter a customer profile to estimate churn probability using the trained"
    " XGBoost pipeline."
)

tenure = st.slider("Tenure (months)", min_value=0, max_value=72, value=12)
monthly = st.number_input(
    "Monthly charges (USD)", min_value=18.0, max_value=120.0, value=70.0
)
contract = st.selectbox(
    "Contract", ["Month-to-month", "One year", "Two year"]
)
internet = st.selectbox(
    "Internet service", ["DSL", "Fiber optic", "No"]
)
tech = st.selectbox(
    "Tech support", ["Yes", "No", "No internet service"]
)
payment = st.selectbox(
    "Payment method",
    [
        "Electronic check",
        "Mailed check",
        "Bank transfer (automatic)",
        "Credit card (automatic)",
    ],
)

if st.button("Predict"):
  row = defaults.copy()
  row.update({
      "tenure": tenure,
      "MonthlyCharges": monthly,
      "TotalCharges": tenure * monthly,
      "Contract": contract,
      "InternetService": internet,
      "TechSupport": tech,
      "PaymentMethod": payment,
  })
  prob = float(model.predict_proba(pd.DataFrame([row]))[0, 1])

  st.metric("Churn probability", f"{prob:.0%}")
  if prob >= 0.50:
    st.warning("High risk: Consider proactive retention outreach or contract offer.")
  else:
    st.success("Lower risk: Standard engagement.")