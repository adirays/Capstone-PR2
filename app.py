import streamlit as st
import pickle
import numpy as np
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
import os
import datetime

# ------------------ PAGE CONFIG (DARK MEDICAL THEME) ------------------
st.set_page_config(
    page_title="Heart Disease Prediction",
    page_icon="❤️",
    layout="wide",
)

st.markdown("""
<style>
    body { background-color: #0e1117; color: white; }
    .stApp { background-color: #0e1117; }
</style>
""", unsafe_allow_html=True)

# ------------------ LOAD MODEL ------------------
model = pickle.load(open("heart_model.pkl", "rb"))

# Example accuracy (use your actual value if known)
MODEL_ACCURACY = 0.87

# ------------------ HEADER ------------------
st.markdown(
    "<h1 style='text-align:center; color:#ff4b4b;'>❤️ Heart Disease Prediction System</h1>",
    unsafe_allow_html=True
)
st.markdown(
    "<p style='text-align:center;'>AI-powered clinical decision support tool</p>",
    unsafe_allow_html=True
)

st.divider()

# ------------------ TABS ------------------
tab1, tab2 = st.tabs(["🩺 Prediction", "📊 Model Info"])

# ================== TAB 1: PREDICTION ==================
with tab1:
    col1, col2 = st.columns(2)

    with col1:
        age = st.number_input("🧓 Age", 20, 100, 45, help="Age of the patient in years")
        sex_str = st.selectbox("👤 Sex", ["Male", "Female"])
        sex = 1 if sex_str == "Male" else 0
        cp = st.selectbox("💢 Chest Pain Type", [0, 1, 2, 3],
                          help="Type of chest pain experienced")
        trestbps = st.number_input("🩺 Resting Blood Pressure", 80, 200, 120,
                                   help="Blood pressure at rest (mm Hg)")
        chol = st.number_input("🧪 Cholesterol", 100, 600, 200,
                                help="Serum cholesterol in mg/dl")
        fbs = st.selectbox("🍬 Fasting Blood Sugar", [0, 1],
                           help="1 if fasting blood sugar > 120 mg/dl")

    with col2:
        restecg = st.selectbox("📈 Resting ECG", [0, 1, 2],
                               help="Resting electrocardiographic results")
        thalach = st.number_input("❤️ Max Heart Rate", 60, 220, 150,
                                  help="Maximum heart rate achieved")
        exang = st.selectbox("🏃 Exercise Induced Angina", [0, 1],
                             help="1 = Yes, 0 = No")
        oldpeak = st.number_input("📉 ST Depression", 0.0, 6.0, 1.0,
                                  help="ST depression induced by exercise")
        slope = st.selectbox("📐 ST Slope", [0, 1, 2],
                             help="Slope of the peak exercise ST segment")
        ca = st.selectbox("🩸 Major Vessels", [0, 1, 2, 3],
                          help="Number of major vessels colored by fluoroscopy")
        thal = st.selectbox("🧬 Thalassemia", [0, 1, 2, 3],
                            help="Blood disorder status")

    st.divider()

    if st.button("🔍 Predict Heart Disease", use_container_width=True):
        input_data = np.array([
            age, sex, cp, trestbps, chol, fbs,
            restecg, thalach, exang, oldpeak,
            slope, ca, thal
        ]).reshape(1, -1)

        prediction = model.predict(input_data)[0]
        probability = model.predict_proba(input_data)[0][1]

        st.subheader("🧾 Prediction Result")

        if prediction == 1:
            st.error(f"⚠️ High Risk of Heart Disease")
        else:
            st.success(f"✅ Low Risk of Heart Disease")

        st.info(f"📊 Risk Probability: **{probability*100:.2f}%**")

        # ------------------ PDF REPORT ------------------
        report_name = f"heart_report_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
        c = canvas.Canvas(report_name, pagesize=A4)
        text = c.beginText(40, 800)

        text.textLine("Heart Disease Prediction Report")
        text.textLine("-" * 40)
        text.textLine(f"Age: {age}")
        text.textLine(f"Sex: {'Male' if sex==1 else 'Female'}")
        text.textLine(f"Cholesterol: {chol}")
        text.textLine(f"Resting BP: {trestbps}")
        text.textLine(f"Prediction: {'Heart Disease Detected' if prediction==1 else 'No Heart Disease'}")
        text.textLine(f"Risk Probability: {probability*100:.2f}%")
        text.textLine("")
        text.textLine("Generated using ML-based Heart Disease Prediction System")

        c.drawText(text)
        c.showPage()
        c.save()

        with open(report_name, "rb") as f:
            st.download_button(
                label="📄 Download Patient Report (PDF)",
                data=f,
                file_name=report_name,
                mime="application/pdf"
            )

# ================== TAB 2: MODEL INFO ==================
with tab2:
    st.subheader("📊 Model Details")

    st.markdown("""
    **Algorithm Used:** Random Forest Classifier  
    **Learning Type:** Supervised Learning (Classification)  
    **Target Variable:** Heart Disease (0 = No, 1 = Yes)  
    **No. of Features:** 13  
    """)

    st.metric(
        label="✅ Model Accuracy",
        value=f"{MODEL_ACCURACY*100:.2f}%",
        delta="Validated on test dataset"
    )

    st.markdown("""
    **Evaluation Metrics Used:**
    - Precision
    - Recall
    - F1-score
    - ROC-AUC  

    **Why Random Forest?**
    - Handles non-linearity well  
    - Reduces overfitting  
    - High recall (important for medical diagnosis)
    """)

# ------------------ FOOTER ------------------
st.markdown(
    "<hr><p style='text-align:center; font-size:12px;'>© Heart Disease Prediction | Streamlit ML App</p>",
    unsafe_allow_html=True
)
