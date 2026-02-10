import pandas as pd
import numpy as np
import os
import joblib
import xgboost as xgb
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.metrics import roc_auc_score

# --- CONFIGURATION ---
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
DATA_DIR = os.path.join(PROJECT_ROOT, 'data', 'processed')
MODELS_DIR = os.path.join(PROJECT_ROOT, 'models')

# 1. Load Data
print("Loading data for optimization...")
X_train = pd.read_csv(os.path.join(DATA_DIR, 'X_train.csv'))
y_train = pd.read_csv(os.path.join(DATA_DIR, 'y_train.csv')).values.ravel()
X_val = pd.read_csv(os.path.join(DATA_DIR, 'X_val.csv'))
y_val = pd.read_csv(os.path.join(DATA_DIR, 'y_val.csv')).values.ravel()

# Calculate scale_pos_weight
negatives = np.sum(y_train == 0)
positives = np.sum(y_train == 1)
scale_pos_weight = negatives / positives

# 2. Define Parameter Grid (The search space)
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [3, 4, 5, 6],
    'learning_rate': [0.01, 0.05, 0.1, 0.2],
    'subsample': [0.7, 0.8, 0.9],
    'colsample_bytree': [0.7, 0.8, 0.9],
    'gamma': [0, 0.1, 0.2, 0.5],
    'scale_pos_weight': [scale_pos_weight, scale_pos_weight * 1.2, scale_pos_weight * 0.8]
}

print(f"Starting Randomized Search (Target AUC > 0.75)...")

# 3. Setup Search
xgb_model = xgb.XGBClassifier(
    objective='binary:logistic',
    eval_metric='logloss',
    use_label_encoder=False,
    random_state=42
)

search = RandomizedSearchCV(
    estimator=xgb_model,
    param_distributions=param_grid,
    n_iter=50,             # Try 50 random combinations
    scoring='roc_auc',     # Optimize for AUC
    cv=3,                  # 3-Fold Cross Validation
    verbose=1,
    random_state=42,
    n_jobs=-1              # Use all CPU cores
)

# 4. Run Search
search.fit(X_train, y_train)

# 5. Evaluate Best Model
best_model = search.best_estimator_
y_prob = best_model.predict_proba(X_val)[:, 1]
best_auc = roc_auc_score(y_val, y_prob)

print("\n" + "="*40)
print(f"OPTIMIZATION RESULTS")
print("="*40)
print(f"Best Params: {search.best_params_}")
print(f"🏆 Best AUC: {best_auc:.4f}")

if best_auc >= 0.75:
    print("✅ SUCCESS: Target Hit!")
else:
    print("⚠️ WARNING: Still under 0.75, but this is the best possible.")

# 6. Save the optimized model
joblib.dump(best_model, os.path.join(MODELS_DIR, 'xgboost.pkl'))
joblib.dump(best_model, os.path.join(MODELS_DIR, 'best_model.pkl')) # Overwrite best model
print("Saved optimized model to models/xgboost.pkl and models/best_model.pkl")