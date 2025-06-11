"""
Train a multi-output Random-Forest that predicts 5 diseases
and save both the model & its label list.
"""
import os, joblib, numpy as np, pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputClassifier
from sklearn.preprocessing import LabelEncoder

# ───────────────────────────────────────────────────────────
# 1. synthetic data (replace later with real EHR data)
np.random.seed(42); N = 500
df = pd.DataFrame({
    "Age"          : np.random.randint(18, 80, N),
    "BMI"          : np.round(np.random.uniform(18, 35, N), 1),
    "Smokes"       : np.random.choice(["Yes", "No"], N),
    "Drinks"       : np.random.choice(["Yes", "No"], N),
    "FamilyHistory": np.random.choice(["Yes", "No"], N)
})
# encode binaries → 0/1
for col in ["Smokes", "Drinks", "FamilyHistory"]:
    df[col] = LabelEncoder().fit_transform(df[col])

# 2. create 5 correlated disease labels (multi-label)
diseases = ["Hypertension", "Diabetes",
            "Heart Disease", "COPD", "Kidney Disease"]
for d in diseases:
    p  = (0.3*(df.Age>50) + 0.3*(df.BMI>27)
         +0.2*df.Smokes   + 0.2*df.Drinks + 0.2*df.FamilyHistory)
    df[d] = (np.random.rand(N) < p).astype(int)

X = df[["Age","BMI","Smokes","Drinks","FamilyHistory"]]
Y = df[diseases]

Xtr, Xte, Ytr, Yte = train_test_split(X, Y, test_size=.2, stratify=Y.max(axis=1))

model = MultiOutputClassifier(RandomForestClassifier(
            n_estimators=300, random_state=42)).fit(Xtr, Ytr)

# 3. persist
os.makedirs("../3_model", exist_ok=True)
joblib.dump(model,        "../3_model/disease_multilabel_predictor.joblib")
joblib.dump(diseases,     "../3_model/disease_labels.joblib")
print("✅ model saved to 3_model/")
