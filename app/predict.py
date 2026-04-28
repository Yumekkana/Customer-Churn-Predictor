from fastapi import APIRouter
from pydantic import BaseModel
import pandas as pd
import pickle

router = APIRouter()

# Load model
with open('ml/model.pkl', 'rb') as file:
    model = pickle.load(file)

# Load the SAME scaler used during training
with open('ml/scaler.pkl', 'rb') as scaler_file:
    scaler = pickle.load(scaler_file)

class Customer(BaseModel):
    CLTV: int
    Contract: int
    Dependents: int
    Multiple_lines: int
    Offer: float
    Satisfaction_score: int
    Tenure_in_month: int
    Total_charges: float
    Total_revenue: float

FEATURE_COLUMNS = [
    "CLTV", "Contract", "Dependents", "Multiple Lines", 
    "Offer", "Satisfaction Score", "Tenure in Months", 
    "Total Charges", "Total Revenue"
]

@router.get("/churn_prediction")
def churn_prediction(cltv: int, contract: int, dependents: int, multiple_lines: int,
                     offer: float, satisfaction_score: int, tenure_in_month: int,
                     total_charges: float, total_revenue: float):
    
    customer = Customer(CLTV=cltv, Contract=contract, Dependents=dependents, Multiple_lines=multiple_lines,
                        Offer=offer, Satisfaction_score=satisfaction_score, Tenure_in_month=tenure_in_month,
                        Total_charges=total_charges, Total_revenue=total_revenue)
    
    df = pd.DataFrame([customer.model_dump()], columns=FEATURE_COLUMNS)

    features_scaled = scaler.transform(df)
    
    prediction = model.predict(features_scaled)
    
    return {"churn_prediction": int(prediction[0])}