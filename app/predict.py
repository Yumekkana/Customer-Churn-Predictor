from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
import pandas as pd
import pickle
from pathlib import Path


router = APIRouter()

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "ml" / "model.pkl"
SCALER_PATH = BASE_DIR / "ml" / "scaler.pkl"


try:
    with open(MODEL_PATH, "rb") as file:
        model = pickle.load(file)

    with open(SCALER_PATH, "rb") as scaler_file:
        scaler = pickle.load(scaler_file)

except FileNotFoundError as error:
    raise RuntimeError(f"Required ML file not found: {error}")

except pickle.UnpicklingError:
    raise RuntimeError("Failed to load model or scaler. File may be corrupted.")

except Exception as error:
    raise RuntimeError(f"Unexpected error while loading ML files: {error}")


class Customer(BaseModel):
    CLTV: int = Field(..., ge=0)
    Contract: int = Field(..., ge=0)
    Dependents: int = Field(..., ge=0, le=1)
    Multiple_lines: int = Field(..., ge=0, le=1)
    Offer: float = Field(..., ge=0)
    Satisfaction_score: int = Field(..., ge=1, le=5)
    Tenure_in_month: int = Field(..., ge=0)
    Total_charges: float = Field(..., ge=0)
    Total_revenue: float = Field(..., ge=0)


FEATURE_COLUMNS = [
    "CLTV",
    "Contract",
    "Dependents",
    "Multiple Lines",
    "Offer",
    "Satisfaction Score",
    "Tenure in Months",
    "Total Charges",
    "Total Revenue",
]


@router.get("/churn_prediction")
def churn_prediction(
    cltv: int,
    contract: int,
    dependents: int,
    multiple_lines: int,
    offer: float,
    satisfaction_score: int,
    tenure_in_month: int,
    total_charges: float,
    total_revenue: float,
):
    try:
        customer = Customer(
            CLTV=cltv,
            Contract=contract,
            Dependents=dependents,
            Multiple_lines=multiple_lines,
            Offer=offer,
            Satisfaction_score=satisfaction_score,
            Tenure_in_month=tenure_in_month,
            Total_charges=total_charges,
            Total_revenue=total_revenue,
        )

        customer_data = customer.model_dump()

        df = pd.DataFrame(
            [{
                "CLTV": customer_data["CLTV"],
                "Contract": customer_data["Contract"],
                "Dependents": customer_data["Dependents"],
                "Multiple Lines": customer_data["Multiple_lines"],
                "Offer": customer_data["Offer"],
                "Satisfaction Score": customer_data["Satisfaction_score"],
                "Tenure in Months": customer_data["Tenure_in_month"],
                "Total Charges": customer_data["Total_charges"],
                "Total Revenue": customer_data["Total_revenue"],
            }],
            columns=FEATURE_COLUMNS,
        )

        features_scaled = scaler.transform(df)
        prediction = model.predict(features_scaled)

        return {
            "churn_prediction": int(prediction[0])
        }

    except ValueError as error:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid input data: {error}",
        )

    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {error}",
        )