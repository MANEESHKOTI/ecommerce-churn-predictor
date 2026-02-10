import streamlit as st
import pandas as pd
import sys
import os

# --- PATH SETUP ---
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

st.markdown("# 📂 Batch Prediction")
st.markdown("Upload a CSV file containing customer data.")

uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

# --- LOGIC ---
if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)
        st.write(f"✅ Uploaded {len(df)} rows.")
        st.write("Preview:", df.head())
        
        if st.button("Run Batch Prediction"):
            with st.spinner("Processing batch..."):
                # Use the efficient batch method from API
                preds, probs = model_service.predict_batch(df)
                
                if preds is not None:
                    # Attach results
                    df['Churn_Prediction'] = preds
                    df['Churn_Probability'] = probs
                    
                    st.success("Batch Analysis Complete!")
                    st.dataframe(df.head())
                    
                    # Download Button
                    result_csv = df.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        "📥 Download Results CSV",
                        result_csv,
                        "churn_predictions_results.csv",
                        "text/csv"
                    )
                else:
                    st.error("Batch prediction failed. Ensure columns match training data.")
                    
    except Exception as e:
        st.error(f"Error processing file: {e}")