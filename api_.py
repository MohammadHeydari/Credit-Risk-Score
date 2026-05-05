from fastapi import FastAPI, HTTPException
import joblib
import pandas as pd
from schema import CreditRequest

app = FastAPI()

model = joblib.load("model.pkl")

@app.post("/predict")
def predict(data: CreditRequest):
    try:
        df = pd.DataFrame([data.dict()])

        prob = model.predict_proba(df)[0][1]

        decision = "reject" if prob > 0.5 else "approve"

        return {
            "risk_score": round(float(prob), 4),
            "decision": decision
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))