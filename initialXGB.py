import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from xgboost import XGBRegressor
import joblib

# Paths
csv_path   = r'C:\Users\lando\Desktop\AI\8090challenge\public_cases.csv'
model_path_full  = r'C:\Users\lando\Desktop\AI\8090challenge\xgb_reimbursement_model.joblib'

# Load data
df = pd.read_csv(csv_path)

# Prepare features and target
X = df[['trip_duration_days', 'miles_traveled', 'total_receipts_amount']]
y = df['expected_output'] if 'expected_output' in df.columns else df['reimbursement']

# CV settings
n_splits = 5
test_size = 0.2
base_seed = 42
mae_list = []

# Cross-validation loop
for i in range(n_splits):
    seed = base_seed + i
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed
    )
    model = XGBRegressor(objective='reg:squarederror', random_state=seed)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    mae = mean_absolute_error(y_test, preds)
    mae_list.append(mae)
    print(f"[Split {i+1}] MAE: {mae:.4f}")

# Average MAE across splits
avg_mae = np.mean(mae_list)
print(f"\nAverage MAE over {n_splits} splits: {avg_mae:.4f}")

# Final model on full data
final_model = XGBRegressor(objective='reg:squarederror', random_state=base_seed)
final_model.fit(X, y)
joblib.dump(final_model, model_path_full)
print(f"\nFinal model trained on 100% of data and saved to {model_path_full}")
