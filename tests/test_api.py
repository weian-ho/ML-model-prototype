from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

def test_health_check_endpoint():
    """test GET /health endpoint"""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "healthy", "model_loaded": True}

def test_predict_endpoint_valid_payload():
    """Test the inference result of POST /predict with a valid payload."""
    payload = {
        "gender": "Female",
        "SeniorCitizen": 0,
        "Partner": "Yes",
        "Dependents": "No",
        "tenure": 12,
        "PhoneService": "Yes",
        "MultipleLines": "No",
        "InternetService": "Fiber optic",
        "OnlineSecurity": "No",
        "OnlineBackup": "Yes",
        "DeviceProtection": "No",
        "TechSupport": "No",
        "StreamingTV": "No",
        "StreamingMovies": "No",
        "Contract": "Month-to-month",
        "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check",
        "MonthlyCharges": 75.35,
        "TotalCharges": 904.2
    }
    
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    
    data = response.json()
    assert "churn_probability" in data
    assert "churn_prediction" in data
    assert "decision_threshold" in data
    
    # Verify probability bounds and prediction value range
    assert 0.0 <= data["churn_probability"] <= 1.0
    assert data["churn_prediction"] in ["Yes", "No"]
    assert data["decision_threshold"] == 0.49

def test_predict_endpoint_missing_field_validation():
    """Test whether POST /predict returns HTTP 422 when required fields are missing"""
    invalid_payload = {
        "gender": "Female",
        #Intentionally omitting the remaining 18 fields
    }
    
    response = client.post("/predict", json=invalid_payload)
    assert response.status_code == 422