from fastapi import FastAPI

from app import predict, metrics
from app.middleware import api_key_middleware


app = FastAPI(
    title="Customer Churn Predictor API",
    description="A FastAPI service for predicting customer churn using machine learning.",
    version="1.0.0",
)


app.middleware("http")(api_key_middleware)


app.include_router(predict.router)
app.include_router(metrics.router)


@app.get("/")
def home():
    return {
        "message": "Customer Churn Predictor API",
        "docs": "/docs",
        "health": "/health",
    }


@app.get("/health")
def health_check():
    return {
        "status": "success",
        "message": "API is running",
    }