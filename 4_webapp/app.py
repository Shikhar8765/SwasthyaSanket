# ============================================================
#  Streamlit App – NCD Risk  +  Multi-Disease  +  Diet & Work
# ============================================================

import streamlit as st
import pandas as pd
import joblib
import os

# ── Local helper modules ────────────────────────────────────
from chatbot import get_bot_response
from utils_disease import (
    predict_diseases_and_confidences,   # multi-label RF
    generate_diet_chart,                # PIL image
    generate_work_chart                 # list[str]
)

# ── Helper to resolve base path ─────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ── 1 ▸ Load community-level NCD proxy model ────────────────
ncd_model = joblib.load(os.path.join(BASE_DIR, "../3_model/best_ncd_model.joblib"))

# ── 2 ▸ Page config ─────────────────────────────────────────
st.set_page_config(page_title="NCD Risk Predictor", layout="wide")
st.title("🧠 NCD Risk & Disease Predictor for ASHA Workers")

# ── 3 ▸ Sidebar inputs ─────────────────────────────────────
st.sidebar.header("📋 NFHS-Based Community Factors")
obese       = st.sidebar.slider("Overweight / Obese Women (%)",            0.0, 100.0, 25.0)
underweight = st.sidebar.slider("Underweight Women (%)",                  0.0, 100.0, 25.0)
school      = st.sidebar.slider("Women ≥10 yrs Schooling (%)",            0.0, 100.0, 50.0)
cleanfuel   = st.sidebar.slider("HHs with Clean Cooking Fuel (%)",        0.0, 100.0, 60.0)
rural_flag  = st.sidebar.radio("Area Type", ["Rural", "Urban"])

st.sidebar.markdown("---")
st.sidebar.subheader("👤 Individual Risk Factors")
age         = st.sidebar.number_input("Age", 18, 100, 45)
bmi         = st.sidebar.number_input("BMI", 10.0, 45.0, 22.5)
smoker      = st.sidebar.selectbox("Smokes?",           ["Yes", "No"])
alcohol     = st.sidebar.selectbox("Consumes alcohol?", ["Yes", "No"])
fam_history = st.sidebar.selectbox("Family history of NCD?", ["Yes", "No"])

# ── 4 ▸ Predict button ─────────────────────────────────────
if st.sidebar.button("🔍 Predict"):
    # 4-A: community NCD risk --------------------------------
    comm_vec = pd.DataFrame([[obese, underweight, school, cleanfuel,
                              int(rural_flag == "Rural")]],
                            columns=["Obese_W","Underweight_W","School_W",
                                     "CleanFuel_HH","RuralFlag"])
    ncd_flag = int(ncd_model.predict(comm_vec)[0])

    st.subheader("🧪 Community-Level NCD Risk")
    if ncd_flag:
        st.error("⚠ High NCD Risk Detected")
    else:
        st.success("✅ Low NCD Risk")

    # 4-B: multi-disease probabilities -----------------------
    st.subheader("🧬 Individual Disease Probabilities")
    probs = predict_diseases_and_confidences(
                age, bmi, smoker, alcohol, fam_history)

    for disease, prob in probs.items():
        if prob >= 0.15:                         # show only if ≥15 %
            st.markdown(f"{disease}** : {prob:.2f}")
            st.progress(prob)

    # 4-C: personalised diet chart (image) -------------------
    st.subheader("🥗 Personalised Diet Chart")
    diet_img = generate_diet_chart(age, bmi, smoker, alcohol, fam_history)
    st.image(diet_img, caption="AI-generated diet plan",
             use_container_width=True)

    # 4-D: daily work-plan list for ASHA ---------------------
    st.subheader("📆 Daily Work Plan for ASHA")
    for task in generate_work_chart(age, bmi, smoker, alcohol, fam_history):
        st.markdown(f"- {task}")

# ── 5 ▸ Chatbot section ────────────────────────────────────
st.markdown("---")
st.subheader("🤖 Hindi Health Chatbot")
user_q = st.text_input("कृपया अपना प्रश्न लिखें:", "मधुमेह क्या है?")
if st.button("💬 उत्तर प्राप्त करें"):
    st.markdown(f"*बॉट:* {get_bot_response(user_q)}")

# ── 6 ▸ Optional evaluation plots ──────────────────────────
with st.expander("📊 Show Model Evaluation Plots"):
    st.image(os.path.join(BASE_DIR, "../3_model/confusion_matrix.png"),
             caption="Confusion Matrix", use_container_width=True)
    st.image(os.path.join(BASE_DIR, "../3_model/roc_curve.png"),
             caption="ROC Curve", use_container_width=True)
    st.image(os.path.join(BASE_DIR, "../3_model/risk_score_violin.png"),
             caption="Risk-Score Violin", use_container_width=True)