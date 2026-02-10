import pandas as pd
import numpy as np
import os
import json
import logging
from datetime import timedelta

# --- CONFIGURATION ---
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
INPUT_FILE = os.path.join(PROJECT_ROOT, "data", "processed", "cleaned_transactions.csv")
OUTPUT_FEATURES = os.path.join(PROJECT_ROOT, "data", "processed", "customer_features.csv")
OUTPUT_INFO = os.path.join(PROJECT_ROOT, "data", "processed", "feature_info.json")
LOG_DIR = os.path.join(PROJECT_ROOT, "logs")

os.makedirs(LOG_DIR, exist_ok=True)
logging.basicConfig(level=logging.INFO, filename=os.path.join(LOG_DIR, 'feature_engineering.log'))

class FeatureEngineer:
    def __init__(self):
        self.transactions = pd.read_csv(INPUT_FILE, parse_dates=['InvoiceDate'])
        self.max_date = self.transactions['InvoiceDate'].max()
        # 120-day window to stabilize churn
        self.training_cutoff = self.max_date - timedelta(days=120)
        self.training_data = self.transactions[self.transactions['InvoiceDate'] <= self.training_cutoff].copy()
        self.observation_data = self.transactions[self.transactions['InvoiceDate'] > self.training_cutoff].copy()
        self.customer_features = pd.DataFrame({'CustomerID': list(self.training_data['CustomerID'].unique())})

    def create_target(self):
        obs_customers = set(self.observation_data['CustomerID'].unique())
        self.customer_features['Churn'] = self.customer_features['CustomerID'].apply(lambda x: 1 if x not in obs_customers else 0)
        print(f"Churn Rate: {self.customer_features['Churn'].mean()*100:.2f}%")

    def engineer_features(self):
        # RFM
        ref_date = self.training_cutoff
        rfm = self.training_data.groupby('CustomerID').agg({
            'InvoiceDate': lambda x: (ref_date - x.max()).days,
            'InvoiceNo': 'nunique',
            'TotalPrice': 'sum',
            'Quantity': 'mean',
            'StockCode': 'nunique'
        }).reset_index()
        rfm.columns = ['CustomerID', 'Recency', 'Frequency', 'TotalSpent', 'AvgBasketSize', 'DistinctProducts']
        self.customer_features = self.customer_features.merge(rfm, on='CustomerID', how='left')

        # Interaction & Golden Features
        self.customer_features['AvgTransactionValue'] = self.customer_features['TotalSpent'] / self.customer_features['Frequency']
        self.customer_features['ProductDiversity'] = self.customer_features['DistinctProducts'] / self.customer_features['Frequency']
        
        # Behavioral
        df = self.training_data.sort_values(['CustomerID', 'InvoiceDate'])
        df['prev_date'] = df.groupby('CustomerID')['InvoiceDate'].shift(1)
        df['days_diff'] = (df['InvoiceDate'] - df['prev_date']).dt.days
        avg_days = df.groupby('CustomerID')['days_diff'].mean().reset_index().rename(columns={'days_diff': 'AvgDaysBetweenPurchases'})
        self.customer_features = self.customer_features.merge(avg_days, on='CustomerID', how='left')
        self.customer_features['AvgDaysBetweenPurchases'].fillna(999, inplace=True)

        # THE GOLDEN FEATURE: Lateness Score
        # Ratio of "How long since they bought" vs "How often they usually buy"
        self.customer_features['LatenessScore'] = self.customer_features['Recency'] / (self.customer_features['AvgDaysBetweenPurchases'] + 1)
        
        # Solo Shopper Flag
        self.customer_features['IsSoloShopper'] = (self.customer_features['Frequency'] == 1).astype(int)

        # Segments
        self.customer_features['R_Score'] = pd.qcut(self.customer_features['Recency'], 4, labels=[4,3,2,1]).astype(int)
        self.customer_features['F_Score'] = pd.qcut(self.customer_features['Frequency'].rank(method='first'), 4, labels=[1,2,3,4]).astype(int)
        self.customer_features['M_Score'] = pd.qcut(self.customer_features['TotalSpent'].rank(method='first'), 4, labels=[1,2,3,4]).astype(int)
        self.customer_features['RFM_Score'] = self.customer_features['R_Score'] + self.customer_features['F_Score'] + self.customer_features['M_Score']
        
        def segment(s):
            if s >= 10: return 'Champions'
            elif s >= 8: return 'Loyal'
            elif s >= 6: return 'Potential'
            elif s >= 4: return 'At Risk'
            else: return 'Lost'
        self.customer_features['CustomerSegment'] = self.customer_features['RFM_Score'].apply(segment)

    def save(self):
        self.customer_features.fillna(0, inplace=True)
        self.customer_features.to_csv(OUTPUT_FEATURES, index=False)
        
        # Save JSON
        cols = [c for c in self.customer_features.columns if c not in ['CustomerID', 'Churn']]
        info = {
            "total_features": len(cols),
            "churn_rate": self.customer_features['Churn'].mean(),
            "features": [{"name": c, "type": str(self.customer_features[c].dtype)} for c in cols],
            "feature_categories": {
                "golden": ["LatenessScore", "IsSoloShopper"],
                "rfm": ["Recency", "Frequency", "TotalSpent"]
            }
        }
        with open(OUTPUT_INFO, 'w') as f:
            json.dump(info, f, indent=4)
        print(f"Features Saved. Count: {len(cols)}")

if __name__ == "__main__":
    fe = FeatureEngineer()
    fe.create_target()
    fe.engineer_features()
    fe.save()