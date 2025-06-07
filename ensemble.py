import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.ensemble import RandomForestRegressor
import joblib

# Paths
csv_path = r'C:\Users\lando\Desktop\AI\8090challenge\public_caseswithfeatures.csv'

# Define paths for saving final models
xgb_model_path_full = r'C:\Users\lando\Desktop\AI\8090challenge\xgb_reimbursement_model_final.joblib'
lgbm_model_path_full = r'C:\Users\lando\Desktop\AI\8090challenge\lgbm_reimbursement_model_final.joblib'
rf_model_path_full = r'C:\Users\lando\Desktop\AI\8090challenge\rf_reimbursement_model_final.joblib'
ensemble_model_path_full = r'C:\Users\lando\Desktop\AI\8090challenge\ensemble_reimbursement_model_final.joblib' # Placeholder for ensemble info

# Load data
df = pd.read_csv(csv_path)

# Prepare features and target
X = df.drop("expected_output", axis=1)
y = df['expected_output'] if 'expected_output' in df.columns else df['reimbursement']

# CV settings
n_splits = 10  # Changed to 10 splits
test_size = 0.2
base_seed = 42

# Lists to store MAEs for each model and the ensemble
xgb_mae_list = []
lgbm_mae_list = []
rf_mae_list = []
ensemble_mae_list = []

print(f"Starting cross-validation with {n_splits} splits...")

# Cross-validation loop
for i in range(n_splits):
    seed = base_seed + i
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed
    )

    print(f"\n--- Split {i+1}/{n_splits} (Seed: {seed}) ---")

    # XGBoost Model
    xgb_model = XGBRegressor(objective='reg:squarederror', random_state=seed, n_jobs=-1)
    xgb_model.fit(X_train, y_train)
    xgb_preds = xgb_model.predict(X_test)
    xgb_mae = mean_absolute_error(y_test, xgb_preds)
    xgb_mae_list.append(xgb_mae)
    print(f"XGBoost MAE: {xgb_mae:.4f}")

    # LightGBM Model
    lgbm_model = LGBMRegressor(random_state=seed, n_jobs=-1)
    lgbm_model.fit(X_train, y_train)
    lgbm_preds = lgbm_model.predict(X_test)
    lgbm_mae = mean_absolute_error(y_test, lgbm_preds)
    lgbm_mae_list.append(lgbm_mae)
    print(f"LightGBM MAE: {lgbm_mae:.4f}")

    # Random Forest Model
    rf_model = RandomForestRegressor(random_state=seed, n_jobs=-1)
    rf_model.fit(X_train, y_train)
    rf_preds = rf_model.predict(X_test)
    rf_mae = mean_absolute_error(y_test, rf_preds)
    rf_mae_list.append(rf_mae)
    print(f"Random Forest MAE: {rf_mae:.4f}")

    # Ensemble Predictions (simple averaging)
    ensemble_preds = (xgb_preds + lgbm_preds + rf_preds) / 3
    ensemble_mae = mean_absolute_error(y_test, ensemble_preds)
    ensemble_mae_list.append(ensemble_mae)
    print(f"Ensemble (XGB+LGB+RF) MAE: {ensemble_mae:.4f}")

# Average MAE across splits
avg_xgb_mae = np.mean(xgb_mae_list)
avg_lgbm_mae = np.mean(lgbm_mae_list)
avg_rf_mae = np.mean(rf_mae_list)
avg_ensemble_mae = np.mean(ensemble_mae_list)

print(f"\n--- Average MAE over {n_splits} splits ---")
print(f"Average XGBoost MAE: {avg_xgb_mae:.4f}")
print(f"Average LightGBM MAE: {avg_lgbm_mae:.4f}")
print(f"Average Random Forest MAE: {avg_rf_mae:.4f}")
print(f"Average Ensemble MAE: {avg_ensemble_mae:.4f}")

# Train final models on the full dataset
print("\nTraining final models on 100% of data...")

# XGBoost final model
final_xgb_model = XGBRegressor(objective='reg:squarederror', random_state=base_seed, n_jobs=-1)
final_xgb_model.fit(X, y)
joblib.dump(final_xgb_model, xgb_model_path_full)
print(f"Final XGBoost model saved to {xgb_model_path_full}")

# LightGBM final model
final_lgbm_model = LGBMRegressor(random_state=base_seed, n_jobs=-1)
final_lgbm_model.fit(X, y)
joblib.dump(final_lgbm_model, lgbm_model_path_full)
print(f"Final LightGBM model saved to {lgbm_model_path_full}")

# Random Forest final model
final_rf_model = RandomForestRegressor(random_state=base_seed, n_jobs=-1)
final_rf_model.fit(X, y)
joblib.dump(final_rf_model, rf_model_path_full)
print(f"Final Random Forest model saved to {rf_model_path_full}")

print("\nNote: The 'ensemble_model_path_full' is a placeholder. An ensemble model is typically a combination of predictions, not a single savable model in this direct averaging approach.")
print("To save an ensemble, you would typically save the individual models and then combine their predictions at inference time.")