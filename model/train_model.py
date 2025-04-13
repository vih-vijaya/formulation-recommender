import pandas as pd
from xgboost import XGBClassifier
import pickle
import sys
import os

# Add the project root directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from etl.clean_data import clean_data

# Load and clean data
df = clean_data("data/raw_formulations.csv")
X = df.drop(columns=["Success"])
y = df["Success"]

# Train with class imbalance handling
model = XGBClassifier(scale_pos_weight=4/3)
model.fit(X, y)

# Save model
with open("model/model.pkl", "wb") as f:
    pickle.dump(model, f)

# Save feature column order
with open("model/feature_order.pkl", "wb") as f:
    pickle.dump(list(X.columns), f)

# Manual F001 input test
f001_input = pd.DataFrame([{
    "Viscosity": 1.2,
    "Density": 0.9,
    "Toxicity_Level": 1,
    "Client_Industry_Automotive": 1,
    "Client_Industry_Energy": 0,
    "Client_Industry_Electronics": 0
}])[X.columns]

pred_f001 = model.predict(f001_input)[0]
proba_f001 = model.predict_proba(f001_input)[0][1]

print("🧪 F001 Prediction:", "✅ Recommended" if pred_f001 else "❌ Not Recommended")
print(f"🔍 Probability of success: {proba_f001:.2f}")
