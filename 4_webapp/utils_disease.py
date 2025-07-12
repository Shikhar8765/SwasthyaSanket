# Enhanced utils_disease.py with Improved UI Components

import joblib
import pandas as pd
import streamlit as st
import os

# ─── Constants ────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Load models
try:
    ML_MODEL = joblib.load(os.path.join(BASE_DIR, "3_model/disease_multilabel_predictor.joblib"))
    LABELS = joblib.load(os.path.join(BASE_DIR, "3_model/disease_labels.joblib"))
except Exception as e:
    st.error(f"Failed to load models: {str(e)}")
    raise

# Diet configuration
DIET_PLANS = {
    'underweight': {
        'status': "Underweight Diet Plan",
        'icon': "📈",
        'recommendations': [
            "🍌 High-calorie breakfast: Banana-Oats-Milk smoothie",
            "🍚 Balanced lunch: Rice + Dal + Vegetables + Ghee",
            "🧀 Protein-rich dinner: Paneer/Chicken + Salad",
            "🥜 Frequent snacks: Dry fruits, boiled eggs, nuts"
        ],
        'color': "#4682B4"
    },
    'healthy': {
        'status': "Healthy Weight Maintenance",
        'icon': "✅",
        'recommendations': [
            "🥣 Light breakfast: Poha/Upma with vegetables",
            "🍽️ Balanced lunch: Chapati + Dal + Seasonal Vegetables",
            "🍛 Moderate dinner: Rice + Sambar + Curd",
            "🍎 Healthy snacks: Fruits, sprouts, buttermilk"
        ],
        'color': "#2E8B57"
    },
    'overweight': {
        'status': "Weight Management Plan",
        'icon': "📉",
        'recommendations': [
            "🍵 Low-cal breakfast: Green tea + Fruits",
            "🥗 High-fiber lunch: Salad + Roti + Light vegetable curry",
            "🥣 Low-carb dinner: Vegetable soup + Grilled protein",
            "🥒 Healthy snacks: Cucumber, roasted chana, buttermilk"
        ],
        'color': "#CD5C5C"
    }
}

def predict_diseases_and_confidences(age, bmi, smoker, alcohol, fam_history):
    """Predict disease risks with improved validation"""
    if not (18 <= age <= 120):
        st.warning("Please enter a valid age between 18-120")
        return {}
    if not (10 <= bmi <= 50):
        st.warning("Please enter a valid BMI between 10-50")
        return {}

    enc = {"Yes": 1, "No": 0}
    try:
        X = pd.DataFrame([[
            age, bmi, enc[smoker], enc[alcohol], enc[fam_history]
        ]], columns=["Age", "BMI", "Smokes", "Drinks", "FamilyHistory"])

        probas = [arr[0][1] for arr in ML_MODEL.predict_proba(X)]
        return {label: round(prob, 3) for label, prob in zip(LABELS, probas)}
    except Exception as e:
        st.error(f"Prediction failed: {str(e)}")
        return {}

def generate_diet_chart(age, bmi, smoker, alcohol, fam_history):
    """Generate enhanced diet plan visualization"""
    # Determine diet plan
    if bmi < 18.5:
        plan = DIET_PLANS['underweight']
    elif bmi < 25:
        plan = DIET_PLANS['healthy']
    else:
        plan = DIET_PLANS['overweight']
    
    # Lifestyle modifications
    modifications = []
    if smoker == "Yes":
        modifications.append("🚭 Add vitamin C rich foods (citrus fruits, amla)")
    if alcohol == "Yes":
        modifications.append("🚰 Extra hydration: 3L water + electrolytes")
    if fam_history == "Yes":
        modifications.append("🩺 Regular monitoring: Weekly BP/sugar checks")
    if age > 50:
        modifications.append("🥛 Calcium-rich foods: Milk, sesame seeds")

    # Build recommendations HTML
    recommendations_html = "".join([
        f"""
        <div style="
            display: flex;
            align-items: flex-start;
            gap: 8px;
            padding: 8px 0;
            border-bottom: 1px solid #e9ecef;
        ">
            <div style="font-size: 1.2rem;">{item.split(' ')[0]}</div>
            <div>{' '.join(item.split(' ')[1:])}</div>
        </div>
        """ for item in plan['recommendations']
    ])
    
    # Build modifications HTML if they exist
    modifications_html = ""
    if modifications:
        modifications_items = "".join([
            f"""
            <div style="
                display: flex;
                align-items: flex-start;
                gap: 8px;
                padding: 8px 0;
                border-bottom: 1px solid #e9ecef;
            ">
                <div style="font-size: 1.2rem;">{item.split(' ')[0]}</div>
                <div>{' '.join(item.split(' ')[1:])}</div>
            </div>
            """ for item in modifications
        ])
        
        modifications_html = f"""
        <div style="margin-top: 1.5rem;">
            <h4 style="margin: 0 0 8px 0; color: #2c3e50;">Lifestyle Adjustments</h4>
            <div style="
                background: #f8f9fa;
                border-radius: 8px;
                padding: 1rem;
            ">
                {modifications_items}
            </div>
        </div>
        """

    # Return the complete HTML
    return f"""
    <div style="
        background: #ffffff;
        border-radius: 12px;
        padding: 1.5rem;
        margin-bottom: 1.5rem;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        border-left: 6px solid {plan['color']};
    ">
        <div style="display: flex; align-items: center; gap: 12px; margin-bottom: 1rem;">
            <div style="
                background: {plan['color']}20;
                width: 48px;
                height: 48px;
                border-radius: 50%;
                display: flex;
                align-items: center;
                justify-content: center;
                font-size: 24px;
            ">{plan['icon']}</div>
            <div>
                <h3 style="margin: 0; color: {plan['color']};">{plan['status']}</h3>
                <p style="margin: 4px 0 0; color: #6c757d; font-size: 0.9rem;">
                    BMI: {bmi:.1f} | Age: {age}
                </p>
            </div>
        </div>
        
        <div style="margin: 1.5rem 0;">
            <h4 style="margin: 0 0 8px 0; color: #2c3e50;">Daily Recommendations</h4>
            <div style="
                background: #f8f9fa;
                border-radius: 8px;
                padding: 1rem;
            ">
                {recommendations_html}
            </div>
        </div>
        
        {modifications_html}
    </div>
    """

def generate_work_chart(age, bmi, smoker, alcohol, fam_history):
    """Generate enhanced ASHA work plan"""
    tasks = []
    priority_colors = {
        "high": "#DC3545",
        "medium": "#FFC107",
        "routine": "#6C757D"
    }
    
    # Base tasks
    tasks.append(("🏠 Routine home visit - symptom check", "15-20 mins", "routine"))
    tasks.append(("📝 Update health records in PHC app", "5-10 mins", "routine"))
    
    # Age-based tasks
    if age >= 50:
        tasks.append(("👵 Elderly health monitoring (BP/sugar)", "20-30 mins", "high"))
    elif age <= 30:
        tasks.append(("👩‍⚕️ Reproductive health counseling", "15 mins", "medium"))
    
    # BMI-based tasks
    if bmi >= 25:
        tasks.append(("📊 Obesity counseling: diet & activity tips", "25 mins", "high"))
    elif bmi < 18.5:
        tasks.append(("🥗 Undernutrition assessment & supplements", "20 mins", "high"))
    
    # Lifestyle tasks
    if smoker == "Yes":
        tasks.append(("🚭 Tobacco cessation awareness session", "15-20 mins", "medium"))
    if alcohol == "Yes":
        tasks.append(("🍺 Alcohol-risk counseling", "20 mins", "medium"))
    
    # Family history tasks
    if fam_history == "Yes":
        tasks.append(("🧬 Follow-up high-risk family members", "30 mins", "high"))
        tasks.append(("📢 Preventive care education", "15 mins", "medium"))
    
    # Build tasks HTML
    tasks_html = "".join([
        f"""
        <div style="
            display: flex;
            gap: 12px;
            padding: 1rem;
            border-radius: 8px;
            background: #f8f9fa;
            transition: transform 0.2s;
        " onmouseover="this.style.transform='scale(1.02)'" 
        onmouseout="this.style.transform='scale(1)'">
            <div style="
                min-width: 36px;
                height: 36px;
                border-radius: 50%;
                background: {priority_colors[priority]};
                color: white;
                display: flex;
                align-items: center;
                justify-content: center;
                font-weight: bold;
                font-size: 0.9rem;
            ">{i}</div>
            <div style="flex-grow: 1;">
                <div style="font-weight: 600; color: #2c3e50;">{task}</div>
                <div style="
                    display: flex;
                    justify-content: space-between;
                    align-items: center;
                    margin-top: 4px;
                ">
                    <span style="font-size: 0.85rem; color: #6c757d;">{duration}</span>
                    <span style="
                        font-size: 0.75rem;
                        padding: 2px 8px;
                        border-radius: 12px;
                        background: {priority_colors[priority]}20;
                        color: {priority_colors[priority]};
                    ">{priority.capitalize()} priority</span>
                </div>
            </div>
        </div>
        """ for i, (task, duration, priority) in enumerate(tasks, 1)
    ])

    # Return the complete HTML
    return f"""
    <div style="
        background: #ffffff;
        border-radius: 12px;
        padding: 1.5rem;
        margin-bottom: 1.5rem;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        border-left: 6px solid #4e73df;
    ">
        <div style="display: flex; align-items: center; gap: 12px; margin-bottom: 1.5rem;">
            <div style="
                background: #4e73df20;
                width: 48px;
                height: 48px;
                border-radius: 50%;
                display: flex;
                align-items: center;
                justify-content: center;
                font-size: 24px;
            ">📆</div>
            <h3 style="margin: 0; color: #4e73df;">ASHA Work Plan</h3>
        </div>
        
        <div style="display: grid; gap: 12px;">
            {tasks_html}
        </div>
    </div>
    """