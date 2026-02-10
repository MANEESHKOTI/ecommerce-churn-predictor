# 🛍️ E-Commerce Customer Churn Prediction AI

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://ecommerce-churn-predictor-5u5pvpwdafpfybn9yypfbo.streamlit.app/)
![Python](https://img.shields.io/badge/Python-3.9-blue)
![Docker](https://img.shields.io/badge/Docker-Enabled-2496ED)
![Model](https://img.shields.io/badge/Champion_Model-XGBoost-orange)

An end-to-end machine learning solution designed to predict customer churn for an e-commerce platform. This application transforms raw transaction logs into actionable insights, identifying at-risk customers to enable targeted retention campaigns.

🔴 **Live Demo:** [Click here to view the App](https://ecommerce-churn-predictiongit-qr4xwis3jph4rmfas48jdu.streamlit.app/)

---

## 📖 Project Overview

### Business Problem
**Goal:** Predict customers who will not make a purchase in the next 90 days.
* **Context:** Customer acquisition costs are **5-25x higher** than retention.
* **Impact:** Reducing churn by **15-20%** can significantly increase customer lifetime value (LTV).
* **Success Metric:** ROC-AUC > 0.75 on the test set.

### Methodology & Results
1. **Data Cleaning:** Handled missing values, removed cancellations, and filtered outliers.
2. **Feature Engineering:** Created 30+ features including RFM (Recency, Frequency, Monetary) scores and purchase velocity.
3. **Model Selection:** Evaluated Logistic Regression, Decision Trees, Random Forest, and XGBoost.
4. **Final Model:** **XGBoost Classifier** achieved the highest performance with an **ROC-AUC of 0.7812**.

---

## 🚀 Key Features

* **🔮 Single Prediction:** Enter customer metrics manually to get an instant risk assessment.
* **📊 Performance Dashboard:** Interactive views of Confusion Matrices and Feature Importance.
* **📉 EDA Insights:** Visualize monthly revenue trends and top-selling products.

---

# --- OPTION A: Run with Docker (Recommended) ---
# 1. Build the image (includes openpyxl & other dependencies)
docker-compose build

# 2. Run the container (Access at http://localhost:8501)
docker-compose up -d 

# --- OPTION B: Run Locally (Python) ---
# 1. Clone the repository
git clone https://github.com/MANEESHKOTI/ecommerce-churn-prediction.git
cd ecommerce-churn-prediction

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the application
streamlit run app/streamlit_app.py# ecommerce_churn_prediction_repo
