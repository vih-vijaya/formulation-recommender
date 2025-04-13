import pickle
import os
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel

# Load model and feature order
BASE_DIR = os.path.dirname(__file__)
with open(os.path.join(BASE_DIR, "..", "model", "model.pkl"), "rb") as f:
    model = pickle.load(f)
with open(os.path.join(BASE_DIR, "..", "model", "feature_order.pkl"), "rb") as f:
    feature_order = pickle.load(f)

app = FastAPI()

class FormulationInput(BaseModel):
    Viscosity: float
    Density: float
    Toxicity_Level: int
    Client_Industry_Automotive: int
    Client_Industry_Energy: int
    Client_Industry_Electronics: int

@app.post("/recommend")
def recommend(data: FormulationInput):
    input_df = pd.DataFrame([data.dict()])
    
    # Reorder columns to match training order
    input_df = input_df[feature_order]

    pred = model.predict(input_df)[0]
    return {"Recommended": bool(pred)}
