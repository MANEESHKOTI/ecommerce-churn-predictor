import pandas as pd
import numpy as np
import os
import json
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INPUT_FILE = os.path.join(PROJECT_ROOT, "data", "processed", "customer_features.csv")
PROCESSED_DIR = os.path.join(PROJECT_ROOT, "data", "processed")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")

os.makedirs(PROCESSED_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

if __name__ == "__main__":
    print("--- Preparing Model Data ---")
    df = pd.read_csv(INPUT_FILE)
    if 'CustomerID' in df.columns: df.drop('CustomerID', axis=1, inplace=True)
    
    # Categorical Encoding
    X = df.drop('Churn', axis=1)
    y = df['Churn']
    X = pd.get_dummies(X, columns=['CustomerSegment'], drop_first=True)
    
    # Save feature names
    with open(os.path.join(PROCESSED_DIR, 'feature_names.json'), 'w') as f:
        json.dump(X.columns.tolist(), f)
        
    # Split
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42)
    
    # Scale
    scaler = StandardScaler()
    X_train = pd.DataFrame(scaler.fit_transform(X_train), columns=X.columns)
    X_val = pd.DataFrame(scaler.transform(X_val), columns=X.columns)
    X_test = pd.DataFrame(scaler.transform(X_test), columns=X.columns)
    
    # Save
    X_train.to_csv(os.path.join(PROCESSED_DIR, "X_train.csv"), index=False)
    X_val.to_csv(os.path.join(PROCESSED_DIR, "X_val.csv"), index=False)
    X_test.to_csv(os.path.join(PROCESSED_DIR, "X_test.csv"), index=False)
    y_train.to_csv(os.path.join(PROCESSED_DIR, "y_train.csv"), index=False)
    y_val.to_csv(os.path.join(PROCESSED_DIR, "y_val.csv"), index=False)
    y_test.to_csv(os.path.join(PROCESSED_DIR, "y_test.csv"), index=False)
    
    joblib.dump(scaler, os.path.join(MODELS_DIR, "scaler.pkl"))
    print("Data Preparation Complete.")