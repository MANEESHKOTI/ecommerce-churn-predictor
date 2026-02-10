import pandas as pd
import numpy as np
import os
import json
import logging
from datetime import datetime

# --- CONFIGURATION ---
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
DEFAULT_INPUT_PATH = os.path.join(PROJECT_ROOT, "data", "raw", "online_retail_II.xlsx")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "data", "processed")
LOG_DIR = os.path.join(PROJECT_ROOT, "logs")

os.makedirs(LOG_DIR, exist_ok=True)
logging.basicConfig(
    filename=os.path.join(LOG_DIR, 'data_cleaning.log'),
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

class DataCleaner:
    def __init__(self, input_path=None):
        self.input_path = input_path if input_path else DEFAULT_INPUT_PATH
        self.df = None
        self.cleaning_stats = {
            'original_rows': 0, 'rows_after_cleaning': 0, 'rows_removed': 0,
            'retention_rate': 0.0, 'missing_values_before': {}, 
            'missing_values_after': {}, 'steps_applied': []
        }

    def load_data(self):
        print(f"Loading raw dataset from {self.input_path}...")
        try:
            if self.input_path.endswith('.csv'):
                self.df = pd.read_csv(self.input_path, encoding='latin1', parse_dates=['InvoiceDate'])
            else:
                self.df = pd.read_excel(self.input_path)
                self.df.rename(columns={'Invoice': 'InvoiceNo', 'Customer ID': 'CustomerID', 'Price': 'UnitPrice', 'Data': 'InvoiceDate'}, inplace=True)
            
            self.cleaning_stats['original_rows'] = len(self.df)
            self.cleaning_stats['missing_values_before'] = self.df.isnull().sum().to_dict()
            return self
        except Exception as e:
            logging.error(f"Failed to load data: {str(e)}")
            raise e

    def remove_missing_customer_ids(self):
        initial_rows = len(self.df)
        self.df = self.df.dropna(subset=['CustomerID'])
        rows_removed = initial_rows - len(self.df)
        self.cleaning_stats['steps_applied'].append({'step': 'remove_missing_customer_ids', 'rows_removed': rows_removed})
        return self

    def handle_cancelled_invoices(self):
        initial_rows = len(self.df)
        self.df['InvoiceNo'] = self.df['InvoiceNo'].astype(str)
        self.df = self.df[~self.df['InvoiceNo'].str.startswith('C')]
        rows_removed = initial_rows - len(self.df)
        self.cleaning_stats['steps_applied'].append({'step': 'handle_cancelled_invoices', 'rows_removed': rows_removed})
        return self

    def handle_negative_quantities(self):
        initial_rows = len(self.df)
        self.df = self.df[self.df['Quantity'] > 0]
        rows_removed = initial_rows - len(self.df)
        self.cleaning_stats['steps_applied'].append({'step': 'handle_negative_quantities', 'rows_removed': rows_removed})
        return self

    def handle_zero_prices(self):
        initial_rows = len(self.df)
        self.df = self.df[self.df['UnitPrice'] > 0]
        rows_removed = initial_rows - len(self.df)
        self.cleaning_stats['steps_applied'].append({'step': 'handle_zero_prices', 'rows_removed': rows_removed})
        return self

    def handle_missing_descriptions(self):
        initial_rows = len(self.df)
        self.df = self.df.dropna(subset=['Description'])
        rows_removed = initial_rows - len(self.df)
        self.cleaning_stats['steps_applied'].append({'step': 'handle_missing_descriptions', 'rows_removed': rows_removed})
        return self

    def remove_outliers(self):
        initial_rows = len(self.df)
        Q1 = self.df['Quantity'].quantile(0.25)
        Q3 = self.df['Quantity'].quantile(0.75)
        IQR = Q3 - Q1
        self.df = self.df[(self.df['Quantity'] >= (Q1 - 1.5 * IQR)) & (self.df['Quantity'] <= (Q3 + 1.5 * IQR))]
        
        Q1_p = self.df['UnitPrice'].quantile(0.25)
        Q3_p = self.df['UnitPrice'].quantile(0.75)
        IQR_p = Q3_p - Q1_p
        self.df = self.df[(self.df['UnitPrice'] >= (Q1_p - 1.5 * IQR_p)) & (self.df['UnitPrice'] <= (Q3_p + 1.5 * IQR_p))]
        
        rows_removed = initial_rows - len(self.df)
        self.cleaning_stats['steps_applied'].append({'step': 'remove_outliers', 'rows_removed': rows_removed, 'method': 'IQR'})
        return self

    def remove_duplicates(self):
        initial_rows = len(self.df)
        self.df = self.df.drop_duplicates()
        rows_removed = initial_rows - len(self.df)
        self.cleaning_stats['steps_applied'].append({'step': 'remove_duplicates', 'rows_removed': rows_removed})
        return self

    def add_derived_columns(self):
        self.df['InvoiceDate'] = pd.to_datetime(self.df['InvoiceDate'])
        self.df['TotalPrice'] = self.df['Quantity'] * self.df['UnitPrice']
        self.df['Year'] = self.df['InvoiceDate'].dt.year
        self.df['Month'] = self.df['InvoiceDate'].dt.month
        self.df['DayOfWeek'] = self.df['InvoiceDate'].dt.dayofweek
        self.df['Hour'] = self.df['InvoiceDate'].dt.hour
        self.cleaning_stats['steps_applied'].append({'step': 'add_derived_columns'})
        return self

    def convert_data_types(self):
        self.df['CustomerID'] = self.df['CustomerID'].astype(int)
        self.df['Country'] = self.df['Country'].astype('category')
        self.cleaning_stats['steps_applied'].append({'step': 'convert_data_types'})
        return self

    def save_cleaned_data(self):
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        self.df.to_csv(os.path.join(OUTPUT_DIR, 'cleaned_transactions.csv'), index=False)
        
        self.cleaning_stats['rows_after_cleaning'] = len(self.df)
        self.cleaning_stats['rows_removed'] = self.cleaning_stats['original_rows'] - self.cleaning_stats['rows_after_cleaning']
        self.cleaning_stats['retention_rate'] = (self.cleaning_stats['rows_after_cleaning'] / self.cleaning_stats['original_rows']) * 100
        self.cleaning_stats['missing_values_after'] = self.df.isnull().sum().to_dict()
        
        def convert_types(obj):
            if isinstance(obj, np.integer): return int(obj)
            elif isinstance(obj, np.floating): return float(obj)
            elif isinstance(obj, np.ndarray): return obj.tolist()
            return obj

        with open(os.path.join(OUTPUT_DIR, 'cleaning_statistics.json'), 'w') as f:
            json.dump(self.cleaning_stats, f, indent=4, default=convert_types)
        
        print(f"Data Cleaning Complete. Retention: {self.cleaning_stats['retention_rate']:.2f}%")
        return self

    def run_pipeline(self):
        self.load_data()
        self.remove_missing_customer_ids()
        self.handle_cancelled_invoices()
        self.handle_negative_quantities()
        self.handle_zero_prices()
        self.handle_missing_descriptions()
        self.remove_outliers()
        self.remove_duplicates()
        self.add_derived_columns()
        self.convert_data_types()
        self.save_cleaned_data()

if __name__ == "__main__":
    cleaner = DataCleaner()
    cleaner.run_pipeline()