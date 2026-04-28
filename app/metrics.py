from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score, roc_auc_score
import pickle
from fastapi import APIRouter
from ml.train import model_selection
from pydantic import BaseModel

router = APIRouter()

X_train, X_test, y_train, y_test = model_selection()

# Load model
with open('ml/model.pkl', 'rb') as file:
    model = pickle.load(file)

y_pred = model.predict(X_test)

class Metrics(BaseModel):
    F1_score: float
    Accuracy_score: float
    Precision_score: float
    Recall_score: float
    Roc_auc_score: float

@router.get("/metrics")
def metrics():
    f1 = f1_score(y_test, y_pred)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test,y_pred)
    recall = recall_score(y_test,y_pred)
    roc_auc = roc_auc_score(y_test,y_pred)

    return Metrics(F1_score=f1, Accuracy_score=accuracy, Precision_score=precision,
                   Recall_score=recall, Roc_auc_score=roc_auc)