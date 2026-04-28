from fastapi import FastAPI

app = FastAPI()

from app import predict, metrics

app.include_router(predict.router)
app.include_router(metrics.router)

@app.get("/health")
def root():
    return {"message": "success"}
