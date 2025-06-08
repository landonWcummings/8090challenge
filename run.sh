#!/bin/bash

# Black Box Challenge - Your Implementation
# This script should take three parameters and output the reimbursement amount
# Usage: ./run.sh <trip_duration_days> <miles_traveled> <total_receipts_amount>

# --- Input Validation ---
if [ "$#" -ne 3 ]; then
  echo "Usage: $0 <trip_duration_days> <miles_traveled> <total_receipts_amount>"
  exit 1
fi

trip_days=$1
miles=$2
receipts=$3

# --- Embedded Python Logic ---
# This section directly executes the Python code using 'python3 -c'
# Note: This still requires Python, numpy, joblib, and lightgbm to be installed.
result=$(python3 -c "
import sys
import os
import numpy as np
import joblib
import lightgbm

# === MODEL FILE NAMES (must live in the same folder as this script) ===
MODEL_FILES = [
    \"xgb_reimbursement_model_final.joblib\",
    \"lgbm_reimbursement_model_final.joblib\",
    \"rf_reimbursement_model_final.joblib\",
]

def main():
    # Parse raw inputs from command line arguments passed to the python -c block
    # These are passed as strings, so they need to be converted to floats.
    days       = float(\"$trip_days\")
    miles      = float(\"$miles\")
    receipts   = float(\"$receipts\")

    # --- Feature engineering (same as before) ---
    miles_per_day    = miles / days    if days else 0.0
    receipts_per_day = receipts / days if days else 0.0

    base_per_diem = days * 100.0
    is_five_day   = 1 if days == 5 else 0

    tier1_miles = min(miles, 100.0)
    tier2_miles = max(miles - 100.0, 0.0)

    log_miles    = np.log1p(miles)
    log_receipts = np.log1p(receipts)

    if days <= 3:
        length_short, length_medium, length_long = 1, 0, 0
    elif days <= 6:
        length_short, length_medium, length_long = 0, 1, 0
    else:
        length_short, length_medium, length_long = 0, 0, 1

    X = np.array([[
        days,
        miles,
        receipts,
        miles_per_day,
        receipts_per_day,
        base_per_diem,
        is_five_day,
        tier1_miles,
        tier2_miles,
        log_miles,
        log_receipts,
        length_short,
        length_medium,
        length_long
    ]])

    # --- Load models from same directory as this script ---
    # When using 'python3 -c', __file__ refers to '<stdin>', so we use PWD
    script_dir = os.environ.get('PWD', os.path.dirname(os.path.realpath(sys.argv[0])))
    
    preds = []
    for fname in MODEL_FILES:
        path = os.path.join(script_dir, fname)
        if not os.path.exists(path):
            print(f\"Error: Model file not found: {path}\", file=sys.stderr)
            sys.exit(1)
        try:
            model = joblib.load(path)
        except Exception as e:
            print(f\"Error loading model at '{path}': {e}\", file=sys.stderr)
            sys.exit(1)
        preds.append(model.predict(X)[0])

    avg_pred = np.mean(preds)
    print(f\"{avg_pred:.2f}\")

if __name__ == \"__main__\":
    main()
"
)

# --- Output the result ---
echo "$result"