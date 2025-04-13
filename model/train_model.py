import pandas as pd
import xgboost as xgb
import pickle
import sys
import os
# Add the project root directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from etl.clean_data import clean_data

df = clean_data("data/raw_formulations.csv")
X = df.drop(columns=["Formulation_ID", "Success"])
y = df["Success"]

model = xgb.XGBClassifier()
model.fit(X, y)

# Save model
with open("model/model.pkl", "wb") as f:
    pickle.dump(model, f)
