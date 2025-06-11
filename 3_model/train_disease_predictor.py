# 2_model/train_disease_predictor.py

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import joblib

# ---- Create synthetic dataset ----
np.random.seed(42)
size = 300
data = pd.DataFrame({
    'Age': np.random.randint(25, 65, size),
    'BMI': np.random.uniform(16, 35, size),
    'Smokes': np.random.randint(0, 2, size),
    'Drinks': np.random.randint(0, 2, size),
    'FamilyHistory': np.random.randint(0, 2, size)
})

# Label assignment
data['Disease'] = (
    (data['BMI'] > 30) |
    (data['Smokes'] == 1) |
    (data['Drinks'] == 1) |
    (data['FamilyHistory'] == 1)
).astype(int)

# ---- Train-test split ----
X = data[['Age', 'BMI', 'Smokes', 'Drinks', 'FamilyHistory']]
y = data['Disease']
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)

# ---- Train the model ----
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# ---- Save the model ----
joblib.dump(clf, "../3_model/disease_predictor.joblib")
print("✅ disease_predictor.joblib saved to 3_model/")
