# ============================================================
#  NCD Care Companion – Streamlit front-end  (Firestore edition)
#  • community card shows model probability
#  • integrates updated chatbot
#  • 2025-07-14: name & phone captured, full report pushed
#  • 2025-07-15: diet / work-plan / doctor saved as sub-collections
#  • 2025-07-16: “same name + same phone ⇒ update, not new doc”
# ============================================================

from __future__ import annotations

import os, joblib, geocoder, html2text
from datetime import datetime
from typing import List, Dict, Tuple

import pandas as pd
import streamlit as st
from streamlit.components.v1 import html

# ─────────────────────────── Firestore ──────────────────────
try:
    import firebase_admin
    from firebase_admin import credentials, firestore  # type: ignore

    ENV_KEY   = os.getenv("GOOGLE_APPLICATION_CREDENTIALS", "")
    LOCAL_KEY = next(
        (
            f
            for f in os.listdir(os.path.dirname(__file__))
            if f.endswith(".json") and "firebase-adminsdk" in f
        ),
        "",
    )
    cred_path = ENV_KEY or os.path.join(os.path.dirname(__file__), LOCAL_KEY)
    if not cred_path or not os.path.exists(cred_path):
        raise FileNotFoundError(
            "Service-account JSON not found – set "
            "$GOOGLE_APPLICATION_CREDENTIALS or copy the file next to app.py"
        )

    if not firebase_admin._apps:  # initialise once
        firebase_admin.initialize_app(credentials.Certificate(cred_path))

    FS_DB = firestore.client()  # type: ignore
except Exception as e:  # noqa: BLE001
    FS_DB = None
    st.warning(f"⚠️  Firestore disabled: {e}")

def fs_safe_write(ref, payload: dict) -> None:
    """Create/update a Firestore document without crashing the UI."""
    if FS_DB is None:
        return
    try:
        ref.set(payload, merge=True)
    except Exception as exc:  # noqa: BLE001
        print(f"Firestore write failed → {exc}")

# NEW ─────────────────── reuse-or-create worker document ────
def get_worker_ref(db, name: str, phone: str):
    """
    Return a DocumentReference in workerDetails:
    • If (name, phone) already exists → existing doc
    • else → brand-new random-ID document
    """
    coll = db.collection("workerDetails")
    hits = (
        coll.where("name", "==", name.strip())
            .where("phone", "==", phone.strip())
            .limit(1)
            .stream()
    )
    hits_list = list(hits)
    if hits_list:
        return coll.document(hits_list[0].id)
    return coll.document()

# ─────────────────── local ML / UI helpers  ─────────────────
from utils_disease import (
    predict_diseases_and_confidences,
    generate_diet_chart,
    generate_work_chart,
    DIET_PLANS,
)
from chatbot import get_bot_response, get_doctor_recommendation

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
COMM_MDL = joblib.load(os.path.join(BASE_DIR, "../3_model/best_ncd_model.joblib"))

# ───────────────────────────── page look  ───────────────────
st.set_page_config("SwasthyaSanket", "🏥", layout="wide")
st.markdown(
    """
    <style>
      body {color:#dfe4ea;font-family:'Segoe UI',Tahoma,Verdana,sans-serif;
            background:#0e1117;}
      .custom-card{border-radius:10px;padding:1.5rem;margin-bottom:1.5rem;
                   background:#ffffff;box-shadow:0 2px 8px rgba(0,0,0,.1);}
      .risk-card {border-left:4px solid #dc3545;}
    </style>
    """,
    unsafe_allow_html=True,
)

# ──────────────────────────── helpers ───────────────────────
def get_user_city() -> str:
    loc = geocoder.ip("me")
    return loc.city if loc.ok and loc.city else "Bhopal"

def pct_colour(prob: float) -> Tuple[str, str]:
    if prob > 0.4:
        return "#dc3545", "High"
    if prob > 0.2:
        return "#ffc107", "Medium"
    return "#28a745", "Low"

def extract_diet_and_tasks(
    age: int, bmi: float, smoker: str, alcohol: str, fam_history: str
) -> Tuple[Dict, List[Dict]]:
    """Return dict for diet and list of dicts for tasks (plain text, no HTML)."""
    # ----- diet --------------------------------------------------------
    if bmi < 18.5:
        key = "underweight"
    elif bmi < 25:
        key = "healthy"
    else:
        key = "overweight"

    plan = DIET_PLANS[key]
    diet_doc: Dict = {
        "status": plan["status"],
        "recommendations": plan["recommendations"],
        "lifestyle_adjustments": [],
    }
    if smoker == "Yes":
        diet_doc["lifestyle_adjustments"].append("Add vitamin-C-rich foods")
    if alcohol == "Yes":
        diet_doc["lifestyle_adjustments"].append("Increase daily water to 3 L")
    if fam_history == "Yes":
        diet_doc["lifestyle_adjustments"].append("Regular BP / glucose check")
    if age > 50:
        diet_doc["lifestyle_adjustments"].append("Increase calcium intake")

    # ----- work-plan ---------------------------------------------------
    tasks: List[Dict] = [
        {
            "task": "Routine home visit – symptom check",
            "duration": "15-20 min",
            "priority": "routine",
        },
        {
            "task": "Update health records in PHC app",
            "duration": "5-10 min",
            "priority": "routine",
        },
    ]
    if age >= 50:
        tasks.append(
            {
                "task": "Elderly health monitoring (BP / sugar)",
                "duration": "20-30 min",
                "priority": "high",
            }
        )
    elif age <= 30:
        tasks.append(
            {
                "task": "Reproductive-health counselling",
                "duration": "15 min",
                "priority": "medium",
            }
        )

    if bmi >= 25:
        tasks.append(
            {
                "task": "Obesity counselling: diet & activity",
                "duration": "25 min",
                "priority": "high",
            }
        )
    elif bmi < 18.5:
        tasks.append(
            {
                "task": "Undernutrition assessment & supplements",
                "duration": "20 min",
                "priority": "high",
            }
        )

    if smoker == "Yes":
        tasks.append(
            {
                "task": "Tobacco-cessation awareness session",
                "duration": "15-20 min",
                "priority": "medium",
            }
        )
    if alcohol == "Yes":
        tasks.append(
            {
                "task": "Alcohol-risk counselling",
                "duration": "20 min",
                "priority": "medium",
            }
        )

    if fam_history == "Yes":
        tasks.append(
            {
                "task": "Follow-up high-risk family members",
                "duration": "30 min",
                "priority": "high",
            }
        )
        tasks.append(
            {
                "task": "Preventive-care education",
                "duration": "15 min",
                "priority": "medium",
            }
        )
    return diet_doc, tasks

# ─────────────────────────────── UI ─────────────────────────
st.title("🏥 NCD Care Companion")
st.caption("Decision-support tool for ASHA workers")

# -------- sidebar ------------------------------------------
with st.sidebar:
    st.header("👤 Worker Details")
    worker_name = st.text_input("Full Name")
    worker_phone = st.text_input("Phone Number")

    st.header("📋 Community Assessment")
    with st.expander("Demographic Factors", True):
        obese = st.slider("Overweight/Obese Women (%)", 0.0, 100.0, 25.0)
        underweight = st.slider("Underweight Women (%)", 0.0, 100.0, 25.0)
        school = st.slider("Women ≥10 yrs Schooling (%)", 0.0, 100.0, 50.0)
        cleanfuel = st.slider("HHs with Clean-fuel (%)", 0.0, 100.0, 60.0)
        rural_flag = st.radio("Area Type", ["Rural", "Urban"], horizontal=True)

    st.header("🩺 Patient Details")
    with st.expander("Individual Factors", True):
        age = st.number_input("Age", 18, 100, 45)
        bmi = st.number_input("BMI", 10.0, 45.0, 22.5, step=0.1)
        smoker = st.selectbox("Smoking Status", ["No", "Yes"])
        alcohol = st.selectbox("Alcohol Consumption", ["No", "Yes"])
        fam_history = st.selectbox("Family History of NCDs", ["No", "Yes"])

    if st.button("🔍 Assess Risk", use_container_width=True):
        st.session_state.assess = True

# -------- main body ----------------------------------------
if st.session_state.get("assess"):

    # ===== predictions =====================================
    X_comm = pd.DataFrame(
        [[obese, underweight, school, cleanfuel, int(rural_flag == "Rural")]],
        columns=[
            "Obese_W",
            "Underweight_W",
            "School_W",
            "CleanFuel_HH",
            "RuralFlag",
        ],
    )
    comm_flag = int(COMM_MDL.predict(X_comm)[0])
    comm_prob = float(COMM_MDL.predict_proba(X_comm)[:, 1][0])

    disease_probs = predict_diseases_and_confidences(
        age, bmi, smoker, alcohol, fam_history
    )

    diet_html = generate_diet_chart(age, bmi, smoker, alcohol, fam_history)
    work_html = generate_work_chart(age, bmi, smoker, alcohol, fam_history)
    city = get_user_city()
    doc_info_html = get_doctor_recommendation(
        max(disease_probs, key=disease_probs.get), city
    )

    # ===== plain-text versions for Firestore ================
    diet_doc, tasks_doc = extract_diet_and_tasks(
        age, bmi, smoker, alcohol, fam_history
    )
    doc_info_text = html2text.html2text(doc_info_html).strip()

    # ===== Firestore write ==================================
    if FS_DB is not None:
        worker_ref = get_worker_ref(FS_DB, worker_name, worker_phone)

        # main doc
        fs_safe_write(
            worker_ref,
            {
                "timestamp": datetime.utcnow().isoformat(),
                "name": worker_name,
                "phone": worker_phone,
                "community_inputs": X_comm.iloc[0].to_dict(),
                "community_flag": comm_flag,
                "community_prob": comm_prob,
                "patient_inputs": {
                    "age": age,
                    "bmi": bmi,
                    "smoker": smoker,
                    "alcohol": alcohol,
                    "fam_history": fam_history,
                },
                "disease_probs": disease_probs,
            },
        )
        # diet sub-collection
        fs_safe_write(worker_ref.collection("diet").document("plan"), diet_doc)

        # workPlan sub-collection
        for i, t in enumerate(tasks_doc, 1):
            fs_safe_write(
                worker_ref.collection("workPlan").document(f"task_{i:02d}"), t
            )

        # doctor sub-collection
        fs_safe_write(
            worker_ref.collection("doctor").document("recommendation"),
            {"city": city, "info": doc_info_text},
        )

    # ===== display =========================================
    st.header("📊 Assessment Results")
    st.write(
        f"**Worker:** {worker_name or '—'} &nbsp;|&nbsp; "
        f"**Phone:** {worker_phone or '—'}"
    )
    st.write(f"**Assessment Time:** {datetime.now():%d %b %Y %H:%M}")

    col1, col2 = st.columns(2, gap="large")

    with col1:
        st.subheader("🧬 Disease Risk Profile")
        for dis, prob in disease_probs.items():
            colour, level = pct_colour(prob)
            st.markdown(
                f"""
                <div style="background:#fff;border-radius:12px;padding:1.25rem;margin-bottom:1rem;
                            box-shadow:0 2px 4px rgba(0,0,0,.05);border-left:4px solid {colour};">
                  <div style="display:flex;justify-content:space-between;margin-bottom:.75rem;">
                    <span style="font-weight:600;font-size:1.1rem;color:#1a1a1a;">{dis}</span>
                    <span style="font-weight:600;color:{colour};background:{colour}20;padding:4px 12px;
                                 border-radius:12px;font-size:.9rem;">{level} Risk</span>
                  </div>
                  <div style="display:flex;align-items:center;gap:1rem;">
                    <div style="flex-grow:1;height:8px;background:#f0f2f6;border-radius:4px;">
                      <div style="width:{prob*100:.0f}%;height:100%;background:{colour};border-radius:4px;"></div>
                    </div>
                    <span style="font-weight:700;color:{colour};font-size:.95rem;">{prob:.0%}</span>
                  </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        comm_col, comm_lvl = ("#dc3545", "High") if comm_flag else ("#28a745", "Low")
        st.subheader("🌍 Community Risk Factors")
        st.markdown(
            f"""
            <div class="custom-card risk-card" style="border-left-color:{comm_col};">
              <div style="display:flex;align-items:center;gap:10px;">
                <span style="font-size:24px;">{"⚠️" if comm_flag else "✅"}</span>
                <div>
                  <h3 style="margin:0;color:{comm_col};">
                    {comm_lvl} NCD Risk Detected
                  </h3>
                  <p style='color:#6c757d;margin:0;'>Model probability: {comm_prob:.1%}</p>
                </div>
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col2:
        st.subheader("📋 Personalized Diet Chart")
        html(diet_html, height=400)

        st.subheader("📆 ASHA Work Plan")
        html(work_html, height=500)

    st.subheader("👨‍⚕️ Recommended Local Doctors")
    st.markdown(f"📍 **Location:** {city}")
    st.markdown(
        f"""
        <div class="custom-card" style="line-height:1.6;color:#1a1a1a;">
          <h4 style="margin-top:0;">Top Doctors for {max(disease_probs, key=disease_probs.get)} in {city}</h4>
          <p style="margin:0;">{doc_info_html}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

# ----------------------- Chatbot ----------------------------
st.markdown("---")
st.markdown(
    """
    <div style="background:#f8fafc;border-radius:12px;padding:1.5rem;margin-bottom:2rem;
                box-shadow:0 2px 8px rgba(0,0,0,0.1);">
      <h2 style="margin-top:0;color:#4e73df;">🤖 Health Assistant Chatbot</h2>
      <p style="color:#6c757d;">Ask questions in Hindi or English</p>
    </div>
    """,
    unsafe_allow_html=True,
)

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

with st.form("chat_form", clear_on_submit=True):
    msg = st.text_area(
        "Type your health question:", label_visibility="collapsed", height=100
    )
    c1, c2 = st.columns(2)
    send = c1.form_submit_button("💬 Send", use_container_width=True)
    clr = c2.form_submit_button("🧹 Clear", use_container_width=True)

if clr:
    st.session_state.chat_history = []
    st.rerun()

if send and msg:
    with st.spinner("Analyzing…"):
        ans = get_bot_response(msg, st.session_state.chat_history)
    st.session_state.chat_history += [
        {"role": "user", "content": msg},
        {"role": "assistant", "content": ans},
    ]
    fs_safe_write(
        FS_DB.collection("chat_logs").document(),  # type: ignore
        {
            "timestamp": datetime.utcnow().isoformat(),
            "worker_name": worker_name,
            "worker_phone": worker_phone,
            "user_msg": msg,
            "bot_reply": ans,
        },
    )

for t in st.session_state.chat_history:
    bg = "#e3f2fd" if t["role"] == "user" else "#f5f5f5"
    lbl = "You" if t["role"] == "user" else "Assistant"
    st.markdown(
        f"""
        <div style="background:{bg};color:#1a1a1a;padding:1rem;border-radius:12px;
                    margin:1rem auto;max-width:80%;box-shadow:0 1px 3px rgba(0,0,0,.1);">
          <b>{lbl}:</b><br>{t['content']}
        </div>
        """,
        unsafe_allow_html=True,
    )

st.markdown("---")
st.caption("NCD Care Companion © 2023 – For ASHA worker use only")
