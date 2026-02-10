import streamlit as st
import pandas as pd
import sys
import os

# --- PATH FIX ---
current_file_path = os.path.abspath(__file__)
project_root = os.path.dirname(os.path.dirname(os.path.dirname(current_file_path)))
if project_root not in sys.path:
    sys.path.append(project_root)

# --- IMPORT ---
try:
    from app.predict import ChurnPredictor
except ImportError:
    st.error("Could not load 'app/predict.py'.")
    st.stop()

predictor = ChurnPredictor()

st.markdown("# 📂 Batch Prediction")
st.markdown("Upload a CSV file containing customer data.")

uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

# --- LOGIC ---
if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)
        st.write(f"✅ Uploaded {len(df)} rows.")
        
        if st.button("Run Batch Prediction"):
            with st.spinner("Analyzing customers..."):
                probs = []
                preds = []
                progress_bar = st.progress(0)
                
                for i, row in df.iterrows():
                    # Call the single prediction logic for each row
                    res = predictor.predict_churn(row.to_dict())
                    
                    if res['status'] == 'success':
                        probs.append(res['churn_probability'])
                        preds.append(res['prediction'])
                    else:
                        probs.append(0.0) # Fallback
                        preds.append(0)
                    
                    progress_bar.progress((i + 1) / len(df))
                
                # Append results
                df['Churn_Prediction'] = preds
                df['Churn_Probability'] = probs
                
                st.dataframe(df.head())
                
                # Download
                result_csv = df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    "📥 Download Results CSV",
                    result_csv,
                    "churn_predictions_results.csv",
                    "text/csv"
                )
                    
    except Exception as e:
        st.error(f"Error processing file: {e}")