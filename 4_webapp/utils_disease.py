# 4_webapp/utils_disease.py
# ------------------------------------------------------------
#  Multi-disease predictor  +  personalised Diet / Work charts
# ------------------------------------------------------------

import joblib, pandas as pd, numpy as np
from PIL import Image, ImageDraw, ImageFont
import os

# ── 1. LOAD MODELS ───────────────────────────────────────────
ML_MODEL   = joblib.load("../3_model/disease_multilabel_predictor.joblib")
LABELS     = joblib.load("../3_model/disease_labels.joblib")   # list[str]

# ── 2. PREDICT MULTIPLE DISEASES WITH CONFIDENCE ─────────────
def predict_diseases_and_confidences(age, bmi, smoker, alcohol, fam_history):
    """
    Return dict {disease_name: probability(0-1)} for an individual.
    """
    enc = {"Yes": 1, "No": 0}
    X = pd.DataFrame([[
        age, bmi, enc[smoker], enc[alcohol], enc[fam_history]
    ]], columns=["Age", "BMI", "Smokes", "Drinks", "FamilyHistory"])

    # MultiOutputClassifier → list of prob-arrays (one per label)
    probas = [arr[0][1] for arr in ML_MODEL.predict_proba(X)]
    return dict(zip(LABELS, probas))           # e.g. {"Diabetes":0.61, …}


# ── 3. DIET-CHART IMAGE  (personalised) ──────────────────────
def generate_diet_chart(age, bmi, smoker, alcohol, fam_history):
    """Return a PIL.Image with bullet-style diet suggestions."""
    # diet selection by BMI
    if bmi < 18.5:
        status = "Under-weight"
        items = [
            "Banana-Oats-Milk breakfast", "Rice-Dal-Veg lunch",
            "Paneer-Salad dinner",       "Dry fruits / eggs snacks"]
    elif bmi < 25:
        status = "Healthy-weight"
        items = [
            "Poha/Upma breakfast", "Chapati-Veg-Dal lunch",
            "Rice-Sambar-Veg dinner", "Fruits / sprouts snacks"]
    else:
        status = "Over-weight"
        items = [
            "Green-tea & fruits", "Salad + Roti (light veg)",
            "Low-carb soup dinner", "Cucumber / Buttermilk snacks"]

    # lifestyle flags
    if smoker == "Yes":
        items.append("Add citrus / antioxidant foods (quit smoking)")
    if alcohol == "Yes":
        items.append("Extra hydration, low-salt meals")
    if fam_history == "Yes":
        items.append("Monitor BP / sugar regularly")

    # Build image
    w, h = 550, 40 + 30 * len(items)
    img = Image.new("RGB", (w, h), "white")
    draw = ImageDraw.Draw(img)
    font = ImageFont.load_default()
    draw.text((10, 10), f"DIET PLAN  ({status})", fill="black", font=font)

    y = 40
    for txt in items:
        draw.text((10, y), f"• {txt}", fill="black", font=font)
        y += 30
    return img


# ── 4. WORK-CHART LIST  (personalised) ───────────────────────
def generate_work_chart(age, bmi, smoker, alcohol, fam_history):
    """Return a list[str] of daily ASHA tasks customised for the patient."""
    tasks = [
        "🏠 Routine home visits – early symptom check",
        "📝 Update village health register / PHC app"
    ]
    if age >= 50:
        tasks.insert(0, "👵 Visit elderly & monitor BP / sugar")
    if bmi >= 25:
        tasks.append("📊 Obesity counselling: diet & activity tips")
    if smoker == "Yes":
        tasks.append("🚭 Tobacco-cessation awareness session")
    if alcohol == "Yes":
        tasks.append("🍺 Alcohol-risk counselling")
    if fam_history == "Yes":
        tasks.append("🧬 Follow-up high-risk family members")

    return tasks
