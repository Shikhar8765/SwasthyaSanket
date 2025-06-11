"""
Utility that loads the new multi-label RF model
and returns a dict of {disease: probability}.
"""
import joblib, numpy as np, pandas as pd, os

MODEL = joblib.load("../3_model/disease_multilabel_predictor.joblib")
LABELS = joblib.load("../3_model/disease_labels.joblib")

def predict_disease_probs(age, bmi, smokes, drinks, fam_hist):
    X = pd.DataFrame([[
        age, bmi, smokes, drinks, fam_hist
    ]], columns=["Age","BMI","Smokes","Drinks","FamilyHistory"])
    # predict_proba gives list of arrays (one per label)
    probas = [p[0][1] for p in MODEL.predict_proba(X)]
    return dict(zip(LABELS, probas))
