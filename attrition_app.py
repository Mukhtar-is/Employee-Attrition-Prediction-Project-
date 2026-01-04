import streamlit as st
import pandas as pd
import numpy as np
import joblib
import time

from streamlit_cookies_manager  import EncryptedCookieManager


# -----------------------------
# Authentication setup
# -----------------------------
cookies = EncryptedCookieManager(
    prefix="hr_app",
    password=st.secrets["cookie"]["password"]
)
if not cookies.ready():
    st.stop()

SESSION_TIMEOUT = 60 * 30  # 30 minutes

def login():
    if "logged_in" not in st.session_state:
        st.session_state.logged_in = False

    # Check cookie
    if cookies.get("login_time"):
        login_time = float(cookies.get("login_time"))
        if time.time() - login_time < SESSION_TIMEOUT:
            st.session_state.logged_in = True

    if st.session_state.logged_in:
        return True

    st.title("🔐 Secure Login")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    if st.button("Login"):
        users = dict(st.secrets.get("auth", {}).get("users", {}))
        if username in users and password == users[username]:
            st.session_state.logged_in = True
            cookies["login_time"] = str(time.time())
            cookies.save()
            st.success("Login successful. Refreshing...")
            st.rerun()
        else:
            st.error("Invalid credentials")
    return False

if not login():
    st.stop()


# Example logout button
if st.sidebar.button("Logout"):
    st.session_state.logged_in = False
    cookies["login_time"] = ""  # clear cookie
    cookies.save()
    st.rerun()


# -----------------------------
# App setup
# -----------------------------
st.set_page_config(page_title="Employee Attrition Predictor", page_icon="💼", layout="wide")

st.title("Employee Attrition Predictor")
st.caption("Predict the likelihood of employee turnover and get clear business friendly interpretations.")


# -----------------------------
# Load trained model (cached)
# -----------------------------
@st.cache_resource
def load_model(path="hr-analytics-pca-forest.pkl"):
    return joblib.load(path)

model = load_model()

# -----------------------------
# Helper: predict with interpretation
# -----------------------------
def predict_employee(example_df: pd.DataFrame):
    # Ensure columns order matches model expectations (uses DataFrame columns)
    pred = model.predict(example_df)[0]
    proba = float(model.predict_proba(example_df)[:, 1][0])  # probability of attrition (class 1)
    return pred, proba

def interpret_probability(p: float):
    # Risk tiers and interpretations
    if p >= 0.70:
        tier = "High risk"
        color = "red"
        action = (
            "- Review workload, role fit, and project alignment\n"
            "- Consider compensation adjustment or career development plan\n"
            "- Schedule manager check-in and engagement actions"
        )
    elif p >= 0.40:
        tier = "Moderate risk"
        color = "orange"
        action = (
            "- Monitor closely over the next quarter\n"
            "- Discuss development opportunities and recognition\n"
            "- Address any blockers (workload, support, tools)"
        )
    else:
        tier = "Low risk"
        color = "green"
        action = (
            "- Maintain engagement and recognition cadence\n"
            "- Keep an eye on workload balance and satisfaction trends"
        )
    return tier, color, action

# -----------------------------
# Sidebar: inputs
# -----------------------------
st.sidebar.header("Employee profile")

average_montly_hours = st.sidebar.slider("Average monthly hours", 60, 360, 148, step=1)
number_project = st.sidebar.slider("Number of projects", 1, 10, 2, step=1)
time_spend_company = st.sidebar.slider("Years at company", 1, 15, 3, step=1)

salary_high = st.sidebar.selectbox("Salary high (binary)", [0, 1], index=0)
Work_accident = st.sidebar.selectbox("Work accident (binary)", [0, 1], index=0)

# PCA components (use ranges that make sense for standardized components)
first_principle_component = st.sidebar.slider("First principal component", -3.0, 3.0, 0.536108, step=0.01)
second_principle_component = st.sidebar.slider("Second principal component", -3.0, 3.0, -0.562070, step=0.01)
third_principle_component = st.sidebar.slider("Third principal component", -3.0, 3.0, 0.672928, step=0.01)

# Construct input row with the exact feature names used in training
input_row = pd.DataFrame([{
    "average_montly_hours": average_montly_hours,
    "number_project": number_project,
    "time_spend_company": time_spend_company,
    "salary_high": salary_high,
    "Work_accident": Work_accident,
    "first_principle_component": first_principle_component,
    "second_principle_component": second_principle_component,
    "third_principle_component": third_principle_component
}])

# -----------------------------
# Prediction section
# -----------------------------
left, right = st.columns([1, 1])

with left:
    st.subheader("Prediction")
    if st.button("Predict attrition"):
        pred, proba = predict_employee(input_row)

        # Probability display
        st.metric(label="Attrition probability", value=f"{proba:.2f}")
        st.progress(min(100, max(0, int(proba * 100))))

        # Result message
        if pred == 1:
            st.error("⚠️ Predicted: Employee likely to leave (class = 1)")
        else:
            st.success("✅ Predicted: Employee likely to stay (class = 0)")

        # Interpretation
        tier, color, action = interpret_probability(proba)
        st.markdown(f"**Risk tier:** {tier}")
        st.markdown("**Recommended actions:**")
        st.markdown(action)

with right:
    st.subheader("Feature summary")
    st.dataframe(input_row.T.rename(columns={0: "value"}))

# -----------------------------
# Optional: Batch prediction
# -----------------------------
st.markdown("---")
st.subheader("Batch predictions (optional)")
st.caption("Upload a CSV with the same feature columns to score multiple employees at once.")

uploaded = st.file_uploader("Upload CSV file", type=["csv"])
if uploaded is not None:
    try:
        batch_df = pd.read_csv(uploaded)
        missing = set(input_row.columns) - set(batch_df.columns)
        extra = set(batch_df.columns) - set(input_row.columns)

        if missing:
            st.error(f"Missing columns: {sorted(missing)}")
        else:
            # Align columns and predict
            batch_df = batch_df[input_row.columns]
            preds = model.predict(batch_df)
            probas = model.predict_proba(batch_df)[:, 1]
            out = batch_df.copy()
            out["prediction"] = preds
            out["attrition_probability"] = probas
            st.success("Batch prediction completed.")
            st.dataframe(out.head(20))

            # Simple summary
            st.markdown("**Summary:**")
            st.write(f"- Total employees scored: {len(out)}")
            st.write(f"- Predicted leavers: {int((out['prediction'] == 1).sum())}")
            st.write(f"- Average attrition probability: {out['attrition_probability'].mean():.2f}")

            # Download results
            csv = out.to_csv(index=False).encode("utf-8")
            st.download_button("Download results as CSV", data=csv, file_name="attrition_predictions.csv", mime="text/csv")

    except Exception as e:
        st.error(f"Error processing file: {e}")

# -----------------------------
# Model info and reliability notes
# -----------------------------
st.markdown("---")
st.subheader("Model notes")
st.markdown(
    "- This app uses your trained Random Forest model saved in hr-analytics-pca-forest.pkl."
)
st.markdown(
    "- The model expects the exact feature names shown above. If your training pipeline included scaling or PCA, that preprocessing should be inside the pickle."
)
st.markdown(
    "- Risk tiers are based on probability thresholds (≥0.70 high, 0.40–0.69 moderate, <0.40 low). Adjust them to match your HR preferences."
)

# Optional: show feature importances if available
if hasattr(model, "feature_importances_"):
    try:
        st.subheader("Feature importance")
        importances = pd.Series(model.feature_importances_, index=input_row.columns).sort_values(ascending=False)
        st.bar_chart(importances)
    except Exception:
        pass




