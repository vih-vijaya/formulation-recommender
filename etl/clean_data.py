import pandas as pd

def clean_data(input_path):
    # Load raw data
    df = pd.read_csv(input_path)

    # Map string toxicity levels to numeric if needed (optional)
    toxicity_map = {"Low": 1, "Medium": 2, "High": 3}
    if df["Toxicity_Level"].dtype == "object":
        df["Toxicity_Level"] = df["Toxicity_Level"].map(toxicity_map)

    # One-hot encode the Client_Industry column
    df = pd.get_dummies(df, columns=["Client_Industry"])

    # Handle missing one-hot columns (if one of the industries is not present in this dataset)
    expected_columns = [
        "Viscosity",
        "Density",
        "Toxicity_Level",
        "Client_Industry_Automotive",
        "Client_Industry_Energy",
        "Client_Industry_Electronics"
    ]
    
    for col in expected_columns:
        if col not in df.columns:
            df[col] = 0  # Add missing industry columns with default 0

    # Reorder the DataFrame to match expected feature order + target
    df = df[expected_columns + ["Success"]]

    return df

if __name__ == "__main__":
    df_clean = clean_data("data/raw_formulations.csv")
    df_clean.to_csv("data/clean_formulations.csv", index=False)
    print("✅ Data cleaned and saved to data/clean_formulations.csv")
    print("🔎 Final columns used:", df_clean.columns.tolist())
