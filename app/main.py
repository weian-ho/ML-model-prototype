import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException

# 显式引入特征工程函数，防止反序列化找不到属性
from src.preprocessing import engineer_features
from app.schemas import CustomerInput, ChurnPredictionOutput

app = FastAPI(
    title="Customer Churn Prediction API",
    description="Production-ready API for Telco Churn inference",
    version="1.0.0"
)

MODEL_PATH = "model/churn_pipeline.joblib"
try:
    model_pipeline = joblib.load(MODEL_PATH)
except Exception as e:
    raise RuntimeError(f"Failed to load model from {MODEL_PATH}: {e}")

@app.get("/health")
def health_check():
    return {"status": "healthy", "model_loaded": True}

@app.post("/predict", response_model=ChurnPredictionOutput)
def predict_churn(customer: CustomerInput):
    try:
        raw_df = pd.DataFrame([customer.model_dump()])
        features_df = engineer_features(raw_df)
        churn_prob = float(model_pipeline.predict_proba(features_df)[0, 1])
        
        thresholds = 0.49
        prediction_label = "Yes" if churn_prob >= thresholds else "No"
        
        return ChurnPredictionOutput(
            churn_probability=round(churn_prob, 4),
            churn_prediction=prediction_label,
            decision_threshold=thresholds
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")