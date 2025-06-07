import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
import joblib

# Paths
csv_path = '/Users/landoncummings/Desktop/ai proj/8090/top-coder-challenge/public_cases.csv'
model_path = 'xgb_reimbursement_model.joblib'

# Ensure output directory exists
import os
os.makedirs(os.path.dirname(model_path), exist_ok=True)

# Load data
df = pd.read_csv(csv_path)

# Prepare features and target
X = df[['trip_duration_days', 'miles_traveled', 'total_receipts_amount']]
y = df['expected_output'] if 'expected_output' in df.columns else df['reimbursement']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train XGB model
model = XGBRegressor(objective='reg:squarederror', random_state=42)
model.fit(X_train, y_train)

# Save the trained model
joblib.dump(model, model_path)

print(f"Model trained and saved to {model_path}")
