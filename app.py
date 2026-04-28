import streamlit as st
import pandas as pd
import numpy as np
import joblib

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Heart Disease Predictor",
    page_icon="❤️",
    layout="centered"
)

st.title("❤️ Heart Disease Prediction")
st.markdown("Enter patient details below to predict heart disease risk.")
st.divider()

# ── Hardcoded feature names (no JSON file needed) ─────────────────────────────
FEATURE_NAMES = [
    'age', 'sex', 'cp', 'trestbps', 'chol', 'fbs',
    'restecg', 'thalach', 'exang', 'oldpeak', 'slope',
    'ca', 'thal', 'age_group', 'hr_reserve',
    'st_slope_interaction', 'angina_flag'
]

# ── Load model and scaler ─────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    model  = joblib.load("heart_disease_model.pkl")
    scaler = joblib.load("heart_disease_scaler.pkl")
    return model, scaler

model, scaler = load_model()

# ── Input form ────────────────────────────────────────────────────────────────
col1, col2 = st.columns(2)

with col1:
    age      = st.slider("Age", 20, 80, 55)
    sex      = st.selectbox("Sex", [1, 0],
                            format_func=lambda x: "Male" if x == 1 else "Female")
    cp       = st.selectbox("Chest Pain Type", [0, 1, 2, 3],
                            format_func=lambda x: [
                                "Typical Angina",
                                "Atypical Angina",
                                "Non-Anginal Pain",
                                "Asymptomatic"
                            ][x])
    trestbps = st.slider("Resting Blood Pressure (mmHg)", 90, 200, 130)
    chol     = st.slider("Cholesterol (mg/dl)", 100, 600, 240)
    fbs      = st.selectbox("Fasting Blood Sugar > 120 mg/dl", [0, 1],
                            format_func=lambda x: "Yes" if x == 1 else "No")
    restecg  = st.selectbox("Resting ECG Result", [0, 1, 2],
                            format_func=lambda x: [
                                "Normal",
                                "ST-T Wave Abnormality",
                                "Left Ventricular Hypertrophy"
                            ][x])

with col2:
    thalach  = st.slider("Max Heart Rate Achieved", 60, 210, 150)
    exang    = st.selectbox("Exercise-Induced Angina", [0, 1],
                            format_func=lambda x: "Yes" if x == 1 else "No")
    oldpeak  = st.slider("ST Depression (oldpeak)", 0.0, 6.0, 1.0, step=0.1)
    slope    = st.selectbox("Slope of ST Segment", [0, 1, 2],
                            format_func=lambda x: [
                                "Upsloping",
                                "Flat",
                                "Downsloping"
                            ][x])
    ca       = st.selectbox("Major Vessels (0-3)", [0, 1, 2, 3])
    thal     = st.selectbox("Thalassemia", [0, 1, 2],
                            format_func=lambda x: [
                                "Normal",
                                "Fixed Defect",
                                "Reversable Defect"
                            ][x])

st.divider()

# ── Predict button ────────────────────────────────────────────────────────────
if st.button("Predict", use_container_width=True, type="primary"):

    patient = {
        'age': age, 'sex': sex, 'cp': cp, 'trestbps': trestbps,
        'chol': chol, 'fbs': fbs, 'restecg': restecg, 'thalach': thalach,
        'exang': exang, 'oldpeak': oldpeak, 'slope': slope,
        'ca': ca, 'thal': thal
    }
    df_in = pd.DataFrame([patient])

    # Feature engineering — must match notebook exactly
    df_in['age_group'] = pd.cut(
        df_in['age'], bins=[0, 40, 50, 60, 100], labels=[0, 1, 2, 3]
    ).astype(int)
    df_in['hr_reserve']           = 220 - df_in['age'] - df_in['thalach']
    df_in['st_slope_interaction'] = df_in['oldpeak'] * (df_in['slope'] + 1)
    df_in['angina_flag']          = (
        (df_in['cp'] == 0) | (df_in['exang'] == 1)
    ).astype(int)

    df_in = df_in[FEATURE_NAMES]

    X_in  = scaler.transform(df_in)
    pred  = model.predict(X_in)[0]
    prob  = model.predict_proba(X_in)[0][1]
    risk  = "High" if prob > 0.7 else ("Medium" if prob > 0.4 else "Low")

    # Results
    st.subheader("Prediction Result")
    r1, r2, r3 = st.columns(3)
    r1.metric("Diagnosis",   "Disease" if pred == 1 else "No Disease")
    r2.metric("Probability", f"{prob:.1%}")
    r3.metric("Risk Level",  risk)

    st.progress(float(prob))
    st.caption(f"Model confidence: {prob:.2%} probability of heart disease")

    if pred == 1:
        st.warning(
            "This patient shows signs of heart disease. "
            "Please consult a cardiologist for further evaluation."
        )
    else:
        st.success(
            "No heart disease detected based on the provided indicators."
        )

    with st.expander("View entered patient data"):
        st.dataframe(pd.DataFrame([patient]).T.rename(columns={0: "Value"}))

st.divider()
st.caption(
    "Built with XGBoost · Cleveland Heart Disease Dataset · "
    "For educational purposes only — not a substitute for medical advice."
)
