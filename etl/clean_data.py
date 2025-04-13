import pandas as pd

def clean_data(input_path):
    df = pd.read_csv(input_path)

    # Map toxicity level
    toxicity_map = {'Low': 1, 'Medium': 2, 'High': 3}
    df['Toxicity_Level'] = df['Toxicity_Level'].map(toxicity_map)

    # One-hot encode industry
    df = pd.get_dummies(df, columns=['Client_Industry'])

    return df

if __name__ == "__main__":
    df_clean = clean_data("data/raw_formulations.csv")
    df_clean.to_csv("data/clean_formulations.csv", index=False)
