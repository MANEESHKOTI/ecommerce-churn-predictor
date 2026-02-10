import streamlit as st

st.markdown("# 📘 Project Documentation")

st.markdown("""
### 1. Project Overview
This project implements an end-to-end **Customer Churn Prediction System** for the E-Commerce sector. It uses historical transaction data to predict the likelihood of a customer stopping their purchasing behavior.

### 2. Model Methodology
* **Algorithm:** **Voting Ensemble** (Soft Voting).
* **Components:** * **Random Forest:** Handles non-linear interactions and is robust to outliers.
    * **XGBoost:** Gradient boosting to minimize residual errors.
* **Training Approach:** Temporal Split (Training on past 21 months, validating on recent 3 months).
* **Target Definition:** "Churn" = No purchase in the observation window (last 120 days).

### 3. Key Feature Definitions
The model relies on 25+ features. The most critical ones are:

| Feature | Definition | Business Logic |
| :--- | :--- | :--- |
| **Recency** | Days since the last purchase. | High Recency = **High Risk**. |
| **Lateness Score** | `Recency / AvgDaysBetweenPurchases` | Measures if a customer is "late" compared to their usual habit. |
| **Frequency** | Total number of distinct orders. | High Frequency = **Low Risk**. |
| **Monetary** | Total lifetime spending (£). | High Monetary = **Low Risk**. |
| **IsSoloShopper** | Binary flag (1 if only 1 order). | Hardest segment to predict; handled separately by tree splits. |

### 4. Technical Stack
* **Python 3.9+**
* **Streamlit** (Frontend Interface)
* **Scikit-Learn & XGBoost** (Machine Learning)
* **Pandas / NumPy** (Data Processing)
* **Docker** (Containerization support)

### 5. Contact
* **Developer:** Bodepudi Maneesh Koti
* **Project:** Partnr Network Assessment
""")