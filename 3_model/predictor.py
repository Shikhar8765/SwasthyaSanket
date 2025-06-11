# 3_model/predictor.py

import numpy as np
import os

# Try importing ONNX Runtime
try:
    import onnxruntime as ort
    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False

import joblib

# Model paths
JOBLIB_MODEL_PATH = os.path.join(os.path.dirname(__file__), "best_ncd_model.joblib")
ONNX_MODEL_PATH = os.path.join(os.path.dirname(__file__), "ncd_model.onnx")

# Define disease labels
DISEASES = ['Diabetes', 'Hypertension']

# Load model
if os.path.exists(JOBLIB_MODEL_PATH):
    model = joblib.load(JOBLIB_MODEL_PATH)
    MODEL_TYPE = "joblib"
elif ONNX_AVAILABLE and os.path.exists(ONNX_MODEL_PATH):
    onnx_session = ort.InferenceSession(ONNX_MODEL_PATH)
    input_name = onnx_session.get_inputs()[0].name
    MODEL_TYPE = "onnx"
else:
    raise Exception("No valid model found in 3_model/. Please ensure a joblib or ONNX model is present.")

# Predict function
def predict_ncd_risk(age, bmi, smoking, alcohol, family_history):
    features = np.array([[age, bmi, smoking, alcohol, family_history]]).astype(np.float32)

    if MODEL_TYPE == "joblib":
        probs = model.predict_proba(features)[0]
    elif MODEL_TYPE == "onnx":
        probs = onnx_session.run(None, {input_name: features})[0][0]
    else:
        raise ValueError("Unsupported model type")

    return {DISEASES[i]: float(probs[i]) for i in range(len(DISEASES))}
