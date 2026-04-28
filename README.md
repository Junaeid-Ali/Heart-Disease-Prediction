# ❤️ Heart Disease Prediction

A machine learning web application that predicts the likelihood of heart disease based on clinical patient data. Built with XGBoost and deployed using Streamlit.

[![Streamlit App]([https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://your-app-link.streamlit.app](https://heart-disease-prediction-knp3tykejmckawza5aqwyc.streamlit.app/))

---

## Live Demo

🔗 [Open the App]([https://your-app-link.streamlit.app](https://heart-disease-prediction-knp3tykejmckawza5aqwyc.streamlit.app/))

Enter patient details using the sliders and dropdowns, click **Predict**, and instantly get a diagnosis with probability score and risk level.

---

## Overview

This project follows a full end-to-end ML pipeline:

```
Data Ingestion → EDA → Preprocessing → Feature Engineering
→ Model Training → Evaluation → Hyperparameter Tuning → Deployment
```

The model was trained on the **Cleveland Heart Disease Dataset** (303 patients, 13 clinical features) and achieves strong AUC-ROC performance, optimised specifically for **recall** — minimising missed diagnoses is the priority in a medical context.

---

## Features

- Interactive web dashboard built with Streamlit
- Predicts heart disease with probability score and risk level (Low / Medium / High)
- Three models compared: Logistic Regression, Random Forest, XGBoost
- SHAP explainability for model transparency
- Feature engineering: age groups, heart rate reserve, ST interaction, angina flag
- Hyperparameter tuning via GridSearchCV with 5-fold cross-validation

---

## Dataset

| Property | Detail |
|----------|--------|
| Source | [Cleveland Heart Disease — Kaggle](https://www.kaggle.com/datasets/cherngs/heart-disease-cleveland-uci) |
| Original | [UCI ML Repository](https://archive.ics.uci.edu/ml/datasets/Heart+Disease) |
| Rows | 303 patients |
| Features | 13 clinical attributes |
| Target | `condition` — 0 = No Disease, 1 = Disease |

### Features Used

| Feature | Description |
|---------|-------------|
| `age` | Age in years |
| `sex` | 1 = Male, 0 = Female |
| `cp` | Chest pain type (0–3) |
| `trestbps` | Resting blood pressure (mmHg) |
| `chol` | Serum cholesterol (mg/dl) |
| `fbs` | Fasting blood sugar > 120 mg/dl |
| `restecg` | Resting ECG results (0–2) |
| `thalach` | Maximum heart rate achieved |
| `exang` | Exercise-induced angina |
| `oldpeak` | ST depression (exercise vs rest) |
| `slope` | Slope of peak exercise ST segment |
| `ca` | Number of major vessels (0–3) |
| `thal` | Thalassemia type |

---

## ML Pipeline

### Models Trained
- Logistic Regression (baseline)
- Random Forest
- XGBoost (best performance)

### Evaluation Metrics
- Accuracy, Precision, Recall, F1 Score
- AUC-ROC (primary metric)
- Confusion Matrix
- SHAP feature importance

### Engineered Features
- `age_group` — clinical age risk bands
- `hr_reserve` — heart rate reserve proxy
- `st_slope_interaction` — ST depression × slope
- `angina_flag` — combined angina signal

---

## Project Structure

```
heart-disease-app/
├── app.py                      # Streamlit web application
├── heart_disease_model.pkl     # Trained XGBoost model
├── heart_disease_scaler.pkl    # Fitted StandardScaler
├── requirements.txt            # Python dependencies
└── README.md                   # Project documentation
```

---

## Run Locally

**1. Clone the repository**
```bash
git clone https://github.com/Junaeid-Ali/Heart-Disease-Prediction.git
cd Heart-Disease-Prediction
```

**2. Install dependencies**
```bash
pip install -r requirements.txt
```

**3. Run the app**
```bash
streamlit run app.py
```

**4. Open in browser**
```
http://localhost:8501
```

---

## Tech Stack

| Tool | Purpose |
|------|---------|
| Python 3.10 | Core language |
| Pandas / NumPy | Data manipulation |
| Scikit-learn | Preprocessing, evaluation |
| XGBoost | Primary model |
| SHAP | Model explainability |
| Streamlit | Web dashboard |
| Joblib | Model serialization |
| Matplotlib / Seaborn | EDA visualizations |

---

## Disclaimer

This application is built for **educational purposes only**. It is not a substitute for professional medical advice, diagnosis, or treatment. Always consult a qualified healthcare provider.

---

## Author

**Junaeid Ali**
- GitHub: [@Junaeid-Ali](https://github.com/Junaeid-Ali)

---

## Acknowledgements

- Dataset: Hungarian Institute of Cardiology, University Hospital Zurich, University Hospital Basel, V.A. Medical Center Long Beach and Cleveland Clinic Foundation
- Donors: David W. Aha, UCI Machine Learning Repository
- Kaggle: [ronitf/heart-disease-uci](https://www.kaggle.com/ronitf/heart-disease-uci)
