import joblib
import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="Qatar Telecom Churn Predictor",
    layout="centered",
)

model = joblib.load("churn_model.pkl")
defaults = joblib.load("defaults.pkl")

st.title("Customer Churn Predictor — Qatar Operator")
st.caption(
    "Estimate subscriber churn probability and retention actions using the trained XGBoost pipeline."
)

st.subheader("1. Account & Plan Details")
col1, col2 = st.columns(2)

with col1:
    tenure = st.slider("Tenure (months)", 0, 72, int(defaults.get("tenure", 12)))
    monthly_qar = st.number_input(
        "Monthly charges (QAR)",
        min_value=50.0,
        max_value=1200.0,
        value=float(defaults.get("MonthlyCharges_QAR", 280.0)),
        step=10.0,
    )
    segment = st.selectbox(
        "Customer segment",
        ["Expat", "Qatari National"],
        index=0 if defaults.get("CustomerSegment", "Expat") == "Expat" else 1,
    )

with col2:
    contract_opts = ["Month-to-month", "One year", "Two year"]
    contract = st.selectbox(
        "Contract",
        contract_opts,
        index=contract_opts.index(defaults["Contract"])
        if defaults.get("Contract") in contract_opts
        else 0,
    )
    internet_opts = ["Fiber (FTTH)", "5G Home Wireless", "Mobile Data Only"]
    internet = st.selectbox(
        "Internet service",
        internet_opts,
        index=internet_opts.index(defaults["InternetService"])
        if defaults.get("InternetService") in internet_opts
        else 0,
    )
    payment_opts = [
        "Card Auto-Pay",
        "Bank Auto-Debit",
        "Digital Wallet / In-App Pay",
        "Prepaid / Cash / Voucher",
    ]
    payment = st.selectbox(
        "Payment method",
        payment_opts,
        index=payment_opts.index(defaults["PaymentMethod"])
        if defaults.get("PaymentMethod") in payment_opts
        else 0,
    )

st.subheader("2. Usage, Bundles & Network Experience")
col3, col4 = st.columns(2)

with col3:
    usage_change = st.slider(
        "Data usage change vs. last month (%)",
        min_value=-80,
        max_value=80,
        value=int(round(float(defaults.get("DataUsageChangePct", -5.0)))),
        help="Negative values indicate lower data consumption than the previous month.",
    )
    bundles = st.slider(
        "Bundled services count",
        min_value=0,
        max_value=4,
        value=int(round(float(defaults.get("BundledServicesCount", 1)))),
    )

with col4:
    ticket_opts = [0, 1, 2, 3, 4]
    default_tickets = int(round(float(defaults.get("SupportTickets30d", 0))))
    tickets = st.selectbox(
        "Support tickets (last 30 days)",
        ticket_opts,
        index=ticket_opts.index(default_tickets) if default_tickets in ticket_opts else 0,
    )
    net_opts = ["Normal", "Moderate Drops", "High Congestion"]
    net_quality = st.selectbox(
        "Network experience",
        net_opts,
        index=net_opts.index(defaults["NetworkQuality"])
        if defaults.get("NetworkQuality") in net_opts
        else 0,
    )

if st.button("Predict", type="primary"):
    row = defaults.copy()
    row.update(
        {
            "tenure": int(tenure),
            "MonthlyCharges_QAR": float(monthly_qar),
            "TotalCharges_QAR": float(tenure * monthly_qar),
            "Contract": contract,
            "InternetService": internet,
            "PaymentMethod": payment,
            "CustomerSegment": segment,
            "DataUsageChangePct": float(usage_change),
            "SupportTickets30d": int(tickets),
            "BundledServicesCount": int(bundles),
            "NetworkQuality": net_quality,
        }
    )

    input_df = pd.DataFrame([row])[list(defaults.keys())]
    prob = float(model.predict_proba(input_df)[0, 1])

    st.divider()
    st.metric("Churn probability", f"{prob:.0%}")

    if prob >= 0.65:
        st.error("High Churn Risk — Prioritize proactive retention outreach.")
        if usage_change <= -20:
            st.info("Suggested play: Bonus local data or roaming pack to re-engage active usage.")
        elif net_quality != "Normal" or tickets >= 2:
            st.info("Suggested play: Priority technical review and service quality check.")
        else:
            st.info("Suggested play: Contract renewal incentive or bundled add-on offer.")
    elif prob >= 0.35:
        st.warning("Moderate Churn Risk — Monitor usage and encourage auto-pay.")
    else:
        st.success("Low Churn Risk — Customer profile appears stable.")