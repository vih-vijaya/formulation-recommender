import streamlit as st
import requests

st.set_page_config(page_title="Formulation Recommender")

st.title("🧪 Intelligent Formulation Recommender")
st.markdown("Provide formulation properties below and get a success prediction based on trained ML model.")

# Inputs
viscosity = st.slider("Viscosity", 0.5, 2.0, 1.1, step=0.1)
density = st.slider("Density", 0.5, 2.0, 0.9, step=0.1)
toxicity = st.selectbox("Toxicity Level", ["Low", "Medium", "High"])
industry = st.selectbox("Client Industry", ["Automotive", "Energy", "Electronics"])

# Map to model input
toxicity_map = {"Low": 1, "Medium": 2, "High": 3}
industry_flags = {
    "Client_Industry_Automotive": 1 if industry == "Automotive" else 0,
    "Client_Industry_Energy": 1 if industry == "Energy" else 0,
    "Client_Industry_Electronics": 1 if industry == "Electronics" else 0
}

# Button
if st.button("🔍 Recommend Formulation"):
    payload = {
        "Viscosity": viscosity,
        "Density": density,
        "Toxicity_Level": toxicity_map[toxicity],
        **industry_flags
    }

    try:
        response = requests.post("http://127.0.0.1:8000/recommend", json=payload)
        if response.status_code == 200:
            result = response.json()["Recommended"]
            if result:
                st.success("✅ This formulation is RECOMMENDED!")
            else:
                st.error("❌ This formulation is NOT recommended.")
        else:
            st.error(f"API returned an error: {response.status_code}")
    except requests.exceptions.RequestException as e:
        st.error(f"Could not connect to FastAPI backend: {e}")
