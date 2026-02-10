import streamlit as st
import sys
import os

# --- PATH SETUP ---
# Fix path so we can import from src
current_file_path = os.path.abspath(__file__)
project_root = os.path.dirname(os.path.dirname(os.path.dirname(current_file_path)))
if project_root not in sys.path:
    sys.path.append(project_root)

# --- IMPORT UNIFIED BACKEND ---
try:
    from src._inference_api import model_service
except ImportError:
    st.error("⚠️ Backend Error: Could not find 'src/_inference_api.py'.")
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
    # Prepare data matching features expected by model
    input_data = {
        'Recency': recency,
        'Frequency': frequency,
        'TotalSpent': monetary,
        'AvgBasketSize': avg_basket,
        'AvgDaysBetweenPurchases': avg_days,
        # Recalculate derived features if logic requires them
        'LatenessScore': recency / (avg_days + 1),
        'IsSoloShopper': 1 if frequency == 1 else 0
    }
    
    with st.spinner("Analyzing..."):
        # Call the unified API
        pred, prob = model_service.predict(input_data)
    
    if pred is not None:
        st.divider()
        st.markdown("### 🔍 Prediction Result")
        
        # Display logic
        if prob > 0.5:
            st.error(f"🚨 **High Churn Risk** (Probability: {prob:.1%})")
            st.info("💡 **Recommendation:** Send immediate retention offer.")
        else:
            st.success(f"✅ **Loyal Customer** (Probability: {prob:.1%})")
    else:
        st.error("Prediction failed. Check inputs or model status.")