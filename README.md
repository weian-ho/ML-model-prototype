# Customer Churn Prediction Service

An end-to-end, production-ready Machine Learning inference service designed to predict telecommunication customer churn. Built with modular feature engineering pipelines, strict Pydantic contract validation, automated Pytest suites, Docker containerization, and automated GitHub Actions CI.

---

## 🌟 Key Architecture & Engineering Features

- **Decoupled Architecture**: Modularized transformation logic (`src/`) separated from API routing (`app/`) to eliminate train-serve feature skew.
- **RESTful Inference Engine**: Built on FastAPI with asynchronous routing and strict request/response data schemas via Pydantic V2.
- **Automated Testing Suite**: High-coverage unit tests for mathematical transformations and integration tests for API boundary conditions using `pytest` and `httpx`.
- **Reproducible Containerization**: Multi-stage Linux-based `Dockerfile` with pinned dependencies and `.dockerignore` for minimal image footprints.
- **Continuous Integration (CI)**: Automated GitHub Actions pipeline executing full regression test suites on every `git push` and Pull Request.

---

## 📁 Repository Structure

```text
ML-model-prototype/
├── .github/workflows/
│   └── ci.yml             # Automated CI pipeline definition
├── app/
│   ├── __init__.py
│   ├── main.py            # FastAPI endpoints and model loading logic
│   └── schemas.py         # Pydantic input/output schemas
├── model/
│   └── churn_pipeline.joblib # Serialized Scikit-Learn pipeline artifact
├── src/
│   ├── __init__.py
│   └── preprocessing.py   # Core feature engineering logic
├── tests/
│   ├── test_api.py        # API endpoint integration tests
│   └── test_preprocessing.py # Feature engineering unit tests
├── .dockerignore
├── .gitignore
├── Dockerfile             # Container image blueprint
├── pytest.ini             # Test runner configuration
├── requirements.txt       # Pinned project dependencies
└── README.md
```

## 🚀 Quickstart Guide
### 1. Local Development Setup
```Bash
# Clone the repository
git clone [https://github.com/](https://github.com/)<your-username>/ML-model-prototype.git
cd ML-model-prototype

# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Run Automated Tests
```Bash
python -m pytest -v
```

### 3. Start Local API Server
```Bash
uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload
```
Navigate to http://127.0.0.1:8000/docs to interact with Swagger UI.

---

## 🐳 Docker Deployment
### Build the Image
```Bash
docker build -t churn-api:v1 .
```

### Run Container
```Bash
docker run -d -p 8000:8000 --name churn-service churn-api:v1
```

Access the live service at http://127.0.0.1:8000/docs.📡 

---

## API Specification
### `POST /predict`
Evaluates single-customer attributes and returns churn probability along with binary classification at decision threshold $0.49$.

**Sample Request Payload:**
```JSON{
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
```

**Sample Response:**
```JSON{
  "churn_probability": 0.582,
  "churn_prediction": "Yes",
  "decision_threshold": 0.49
}
```