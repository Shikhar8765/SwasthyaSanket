# 🧠 SwasthyaSanket: AI-Powered NCD Risk & Care Assistant for ASHA Workers

![Streamlit](https://img.shields.io/badge/Streamlit-Enabled-brightgreen)
![License](https://img.shields.io/badge/License-MIT-blue)
![Deployment](https://img.shields.io/badge/Deployed-Railway-purple)

## 🔗 Live Demo
*👉 [Access the App on Railway](https://swasthyasanket-production.up.railway.app)*

---

## 📌 Problem Statement
In rural areas, Non-Communicable Diseases (NCDs) such as diabetes and hypertension often go undetected due to lack of screening tools and awareness. ASHA workers struggle to identify high-risk individuals early.

---

## 💡 Solution
*SwasthyaSanket* is a lightweight, offline-friendly web app built for ASHA workers to:
- 🔍 Predict community-level NCD risk using NFHS-5 features.
- 🧬 Predict individual disease probabilities (Diabetes, BP, etc.).
- 🥗 Generate personalized diet plans based on BMI & habits.
- 📆 Create daily ASHA work plans for community follow-ups.
- 🤖 Chatbot in Hindi for basic health Q&A (rule-based + Gemini API fallback).

---

## 🛠 Tech Stack
- *Frontend/UI:* Streamlit
- *ML Models:* Scikit-learn, XGBoost
- *Backend & Hosting:* Python, Railway
- *Voice Support:* pyttsx3 (for offline readiness)
- *Language Support:* Hindi + Hinglish
- *Data Source:* [NFHS-5](https://rchiips.org/nfhs/)

---

## 📁 Folder Structure
ncd-risk-predictor/
│
├── 1_data/                     # Raw and cleaned NFHS-5 data
├── 2_notebooks/               # Model development notebooks
├── 3_model/                   # Trained ML models (.joblib)
├── 4_webapp/                  # Streamlit app and logic
│   ├── app.py                 # Main Streamlit app
│   ├── chatbot.py             # Rule-based Hindi chatbot
│   ├── utils_disease.py       # Disease prediction, diet, work planner
├── requirements.txt           # Dependencies
├── README.md                  # This file
---

## ⚙ How to Run Locally
```bash
# Step 1: Clone the repository
git clone https://github.com/Shikhar8765/SwasthyaSanket.git
cd SwasthyaSanket

# Step 2: Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows

# Step 3: Install dependencies
pip install -r requirements.txt

# Step 4: Run the app
cd 4_webapp
streamlit run app.py
Deployment

The app is deployed using Railway and connected directly to GitHub. Any changes pushed to the main branch auto-trigger redeployment.
You can access project at:-https://swasthyasanket-production.up.railway.app/
⸻

🤝 Acknowledgements
	•	NFHS-5 Dataset
	•	Google Gemini API
	•	Streamlit, Scikit-learn, XGBoost, Railway