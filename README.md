# Customer Churn Predictor
Churn Prediction API is a high-performance, production-ready REST service built with FastAPI that predicts customer churn probability using machine learning. By leveraging a pre-trained model and standardized feature scaling, the API provides real-time binary predictions (Churn/No Churn). Designed for scalability and ease of integration, it enables businesses to proactively retain at-risk customers through accurate, data-driven insights.

## Install dependencies
pip install -r requirements.txt

## Run
uvicorn app.main:app --reload

### Parameters
- **cltv(int)** = Customer Lifetime Value
- **contract(int)** = Contract type [1, 12, 24]
- **dependents(int)** = Dependents (0 for False, 1 for True)
- **multiple_lines(int)** = Multiple lines (0 for False, 1 for True)
- **offer(float)** = Offer discount (0 for False, 1 for True)
- **satisfaction_score(int)** = Score (1-5)
- **tenure_in_month(int)** = Tenure in a month
- **total_charges(float)** = Total Charges
- **total_revenue(float)** = Total Revenue

## Test
- curl http://localhost:8000/ping
- curl "http://localhost:8000/churn_prediction?cltv=1500&contract=1&dependents=0&multiple_lines=1&offer=1&satisfaction_score=3&tenure_in_month=24&total_charges=2500&total_revenue=3000"

## Docs
http://localhost:8000/docs

## Dataset source
https://huggingface.co/datasets/aai510-group1/telco-customer-churn

## Response 
{"churn_prediction": 0}  // 0=No Churn, 1=Churn