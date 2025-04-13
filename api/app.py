from fastapi import FastAPI
from pydantic import BaseModel
import pickle
import pandas as pd

# Load model
import os

MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "model", "model.pkl")
with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)

app = FastAPI()

class FormulationInput(BaseModel):
    Viscosity: float
    Density: float
    Toxicity_Level: int  # 1 = Low, 2 = Medium, 3 = High
    Client_Industry_Automotive: int
    Client_Industry_Energy: int
    Client_Industry_Electronics: int

@app.post("/recommend")
def recommend(data: FormulationInput):
    input_df = pd.DataFrame([data.dict()])
    pred = model.predict(input_df)[0]
    return {"Recommended": bool(pred)}
