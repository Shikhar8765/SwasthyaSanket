"""
Train Gradient-Boosting 'High NCD Risk' model
from NFHS-5 factsheet rows (robust column detection).

Outputs ➜ 3_model/best_ncd_model.joblib
"""
import os, urllib.request, warnings, joblib
import pandas as pd, numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics  import roc_auc_score, classification_report

warnings.filterwarnings("ignore")

# ── paths ────────────────────────────────────────────────────
DATA_DIR   = "1_data"
CSV_PATH   = os.path.join(DATA_DIR, "NFHS_5_Factsheets_Data.csv")
XLS_PATH   = os.path.join(DATA_DIR, "NFHS_5_Factsheets_Data.xls")
URL        = ("https://main.mohfw.gov.in/sites/default/files/"
              "NFHS5_State_FactSheet_India%28compressed%29.xls")
os.makedirs(DATA_DIR, exist_ok=True)

# ── load factsheet (offline preferred) ──────────────────────
if os.path.exists(CSV_PATH):
    df = pd.read_csv(CSV_PATH)
elif os.path.exists(XLS_PATH):
    df = pd.read_excel(XLS_PATH, sheet_name=0)
else:
    print("Downloading NFHS-5 factsheet … (once only)")
    with urllib.request.urlopen(URL) as r:
        open(XLS_PATH, "wb").write(r.read())
    df = pd.read_excel(XLS_PATH, sheet_name=0)
    df.to_csv(CSV_PATH, index=False)

# ── helper to locate a column by keywords (case-insensitive) ─
def find_col(frame, *kws):
    kws = [k.lower() for k in kws]
    for col in frame.columns:
        name = col.lower()
        if all(k in name for k in kws):
            return col
    raise KeyError(f"No column includes keywords {kws}")

# detect columns (matches list you printed)
col_obese  = find_col(df, "overweight", "obese", "women")
col_under  = find_col(df, "below normal", "bmi", "women")
col_school = find_col(df, "10 or more years of schooling", "women")
col_fuel   = find_col(df, "clean fuel for cooking")
col_area   = find_col(df, "area")           # Rural / Urban

print("\nDetected columns →")
for n,c in zip(["Obese","Under","School","Fuel","Area"],
               [col_obese,col_under,col_school,col_fuel,col_area]):
    print(f"  {n:<6}: {c}")

# rename + subset
df = df.rename(columns={
    col_obese : "Obese_W",
    col_under : "Underweight_W",
    col_school: "School_W",
    col_fuel  : "CleanFuel_HH",
    col_area  : "Area"
})[["Obese_W","Underweight_W","School_W","CleanFuel_HH","Area"]]

# numeric
for c in ["Obese_W","Underweight_W","School_W","CleanFuel_HH"]:
    df[c] = pd.to_numeric(df[c], errors="coerce")
df = df.dropna()

# proxy label
df["risk_score"] = (
      df["Obese_W"]/35
    + np.maximum(0, 30 - df["School_W"])/30
    + np.maximum(0, 50 - df["CleanFuel_HH"])/50
)
df["HighRisk"] = (df["risk_score"] > 1.2).astype(int)
print("\nClass counts:", df.HighRisk.value_counts().to_dict())

# model
X = df[["Obese_W","Underweight_W","School_W","CleanFuel_HH"]]
y = df["HighRisk"]
Xtr,Xte,ytr,yte = train_test_split(X,y,stratify=y,
                                   test_size=.25, random_state=42)

gb = GradientBoostingClassifier(
        n_estimators=300, learning_rate=0.06,
        max_depth=3, random_state=42).fit(Xtr,ytr)

print("\nHold-out AUC =", roc_auc_score(yte, gb.predict_proba(Xte)[:,1]))
print(classification_report(yte, gb.predict(Xte),
      target_names=["Low","High"]))

# save
os.makedirs("3_model", exist_ok=True)
joblib.dump(gb, "3_model/best_ncd_model.joblib")
print("✅ Saved ➜ 3_model/best_ncd_model.joblib")
