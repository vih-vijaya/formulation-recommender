## 🧪 Intelligent Product Formulation Recommendation System

An end-to-end machine learning project that recommends optimal chemical formulations based on historical lab performance, customer feedback, and domain-specific characteristics. Built using **Python, XGBoost, FastAPI**, and **Streamlit**, this project simulates a real-world scenario in the specialty chemicals industry where predictive modeling improves R&D efficiency.

---

## 🚀 Features

- 🧠 Predicts success/failure of a chemical formulation
- 📊 Trained using real-like tabular data with features like viscosity, density, toxicity, and industry
- 🔁 Handles ETL (Extract-Transform-Load) and automates feature engineering
- 🧪 Uses **XGBoost** for robust binary classification
- ⚡ Deployed via **FastAPI** for backend inference
- 💬 Interactive **Streamlit frontend** for easy user testing
- 📈 Tracks training performance with accuracy, confusion matrix, and classification reports

---

---

## ⚙️ How to Run the Project

### 1️⃣ Clone the Repo
git clone https://github.com/YOUR_USERNAME/formulation-recommender.git
cd formulation-recommender

2️⃣ Set Up Virtual Environment
python -m venv venv
# Activate it
source venv/bin/activate      # On Mac/Linux
venv\Scripts\activate         # On Windows

3️⃣ Install Requirements
pip install -r requirements.txt

4️⃣ Train the Model
python model/train_model.py

5️⃣ Start the Backend
uvicorn api.app:app --reload
Visit: http://127.0.0.1:8000/docs to access the Swagger UI

6️⃣ Start the Frontend
streamlit run streamlit_app.py
Use the sliders and dropdowns to simulate a chemical formulation and get a recommendation!

🧪 Sample Inputs
Viscosity	Density	Toxicity	Industry	Prediction
1.2	0.9	Low	Automotive	✅ Recommended
0.75	1.2	High	Energy	❌ Not Recommended
📊 Sample Model Output
lua
Copy
Edit
✅ Training Accuracy: 0.86

📊 Classification Report:
              precision    recall  f1-score   support
         0       0.80      1.00      0.89         4
         1       1.00      0.75      0.86         4

🔁 Confusion Matrix:
[[4 0]
 [1 3]]
🧠 Tech Stack
Tool	Purpose
Python	Main programming language
XGBoost	Predictive modeling
Pandas	Data cleaning & wrangling
FastAPI	Backend API
Streamlit	Frontend for user testing
Scikit-learn	Metrics + Evaluation
📁 Sample Data Format
csv
Copy
Edit
Formulation_ID,Viscosity,Density,Toxicity_Level,Client_Industry,Success
F001,1.2,0.9,1,Automotive,1
F002,0.8,1.1,2,Energy,0
F003,1.1,0.95,3,Electronics,1

👤 Author
Vijayalakshmi Jakkati

📄 License
This project is for educational and demonstration purposes only. No proprietary data or business logic is used.




