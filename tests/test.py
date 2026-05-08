from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

def test_health():
    response = client.get("/health")
    assert response.status_code == 200

def test_churn_prediction():
    response = client.get(
        "/churn_prediction",
        params={
            "cltv": 5000,
            "contract": 1,
            "dependents": 0,
            "multiple_lines": 1,
            "offer": 2.0,
            "satisfaction_score": 4,
            "tenure_in_month": 12,
            "total_charges": 1200.50,
            "total_revenue": 1500.75
        }
    )

    assert response.status_code == 200

    data = response.json()

    assert "churn_prediction" in data
    assert isinstance(data["churn_prediction"], int)

def test_metrcis():
    response = client.get(
        "/metrics", 
    )

    assert response.status_code == 200
    assert "F1_score" in response.json()
