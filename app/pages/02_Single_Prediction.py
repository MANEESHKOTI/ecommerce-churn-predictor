import streamlit as st
import sys
import os
import time

# --- PATH FIX ---
# Calculate path to project root
current_file_path = os.path.abspath(__file__)
project_root = os.path.dirname(os.path.dirname(os.path.dirname(current_file_path)))
if project_root not in sys.path:
    sys.path.append(project_root)

# --- IMPORT NEW PREDICTOR ---
try:
    from app.predict import ChurnPredictor
except ImportError:
    st.error("Could not load 'app/predict.py'. Check your folder structure.")
    st.stop()

# --- LOAD MODEL ---
@st.cache_resource
def get_predictor():
    return ChurnPredictor()

try:
    predictor = get_predictor()
except Exception as e:
    st.error(f"Failed to initialize model: {e}")
    st.stop()

st.markdown("# 👤 Single Customer Prediction")
st.markdown("Enter customer metrics below to predict churn probability.")

# --- INPUT FORM ---
with st.form("prediction_form"):
    col1, col2 = st.columns(2)
    
    with col1:
        recency = st.number_input("Recency (Days since last purchase)", 0, 365, 10)
        frequency = st.number_input("Frequency (Total orders)", 1, 1000, 5)
        avg_days = st.number_input("Avg Days Between Purchases", 1.0, 365.0, 30.0)
        
    with col2:
        monetary = st.number_input("Total Spend (£)", 0.0, 100000.0, 500.0)
        avg_basket = st.number_input("Avg Basket Size", 1.0, 100.0, 10.0)

    submitted = st.form_submit_button("Predict Churn Risk")

# --- LOGIC ---
if submitted:
    # Prepare data matching the features expected by predict.py
    input_data = {
        'Recency': recency,
        'Frequency': frequency,
        'TotalSpent': monetary,
        'AvgBasketSize': avg_basket,
        'AvgDaysBetweenPurchases': avg_days,
        'LatenessScore': recency / (avg_days + 1),  # Dynamic Feature Calculation
        'IsSoloShopper': 1 if frequency == 1 else 0
    }
    
    with st.spinner("Analyzing..."):
        result = predictor.predict_churn(input_data)
    
    if result['status'] == 'success':
        pred = result['prediction']
        prob = result['churn_probability']
        
        st.divider()
        st.markdown("### 🔍 Prediction Result")
        
        if prob > 0.5:
            st.error(f"🚨 **High Churn Risk** (Probability: {prob:.1%})")
            st.info("💡 **Recommendation:** Send immediate retention offer.")
        else:
            st.success(f"✅ **Loyal Customer** (Probability: {prob:.1%})")
    else:
        st.error(f"Prediction Error: {result.get('message')}")