import streamlit as st
import os

st.set_page_config(
    page_title="Churn Predictor",
    page_icon="🏠",
    layout="wide"
)

st.markdown("# 🏠 Project Overview")
st.markdown("""
### 🎯 Business Objective
The goal of this project is to **predict customer churn** for an online retailer. 
Identifying at-risk customers allows the marketing team to intervene with targeted campaigns *before* the customer leaves.

### 📉 Key Metrics
* **Churn Definition:** No purchase in the last **120 days** (Observation Window).
* **Target Metric:** **ROC-AUC > 0.75** (Achieved: **0.766** ✅).
* **Model Used:** Voting Ensemble (Random Forest + XGBoost).

### 🛠️ System Architecture
1.  **Data Ingestion:** Loads `online_retail_II.xlsx`.
2.  **Preprocessing:** Cleans cancellations, IQR outlier removal.
3.  **Feature Engineering:** RFM, Lateness Score, Behavioral Ratios.
4.  **Modeling:** Voting Classifier (Soft Vote) with hyperparameter tuning.
5.  **Deployment:** Streamlit Interface with Batch Processing.

### 📊 ROI Impact
| Action | Impact |
| :--- | :--- |
| **Early Detection** | Save 20% of at-risk customers. |
| **Targeted Discounts** | Reduce marketing waste by 15%. |
| **Inventory Mgmt** | Better demand forecasting. |

---
*👈 Use the sidebar to navigate to **Single Prediction** or **Batch Prediction**.*
""")