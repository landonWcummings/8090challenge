import sys
import os
import json
import numpy as np
import joblib
import lightgbm # Ensure this is imported if your models use it
import xgboost # Ensure this is imported if your models use it
import sklearn # Required for RandomForestRegressor if you don't have it explicitly in your imports

# --- Configuration ---
PRIVATE_CASES_FILE = "private_cases.json"
PRIVATE_ANSWERS_FILE = "private_answers.txt"

# === MODEL FILE NAMES (must live in the same folder as this script) ===
MODEL_FILES = [
    "xgb_reimbursement_model_final.joblib",
    "lgbm_reimbursement_model_final.joblib",
    "rf_reimbursement_model_final.joblib",
]

def load_models(script_dir):
    """Loads all specified models into memory."""
    models = []
    print("Loading models...")
    for fname in MODEL_FILES:
        path = os.path.join(script_dir, fname)
        if not os.path.exists(path):
            print(f"Error: Model file not found: {path}", file=sys.stderr)
            sys.exit(1)
        try:
            model = joblib.load(path)
            models.append(model)
            print(f"  Loaded {fname}")
        except Exception as e:
            print(f"Error loading model at '{path}': {e}", file=sys.stderr)
            sys.exit(1)
    print("All models loaded successfully.")
    return models

def featurize_and_predict(case_data, loaded_models):
    """
    Performs feature engineering and predicts reimbursement using loaded models.
    Returns the averaged prediction or 'ERROR' if an issue occurs.
    """
    try:
        days = float(case_data.get("trip_duration_days", 0))
        miles = float(case_data.get("miles_traveled", 0))
        receipts = float(case_data.get("total_receipts_amount", 0))

        # --- Feature engineering ---
        miles_per_day = miles / days if days else 0.0
        receipts_per_day = receipts / days if days else 0.0

        base_per_diem = days * 100.0
        is_five_day = 1 if days == 5 else 0

        tier1_miles = min(miles, 100.0)
        tier2_miles = max(miles - 100.0, 0.0)

        log_miles = np.log1p(miles)
        log_receipts = np.log1p(receipts)

        length_short, length_medium, length_long = 0, 0, 0
        if days <= 3:
            length_short = 1
        elif days <= 6:
            length_medium = 1
        else:
            length_long = 1

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

        preds = [model.predict(X)[0] for model in loaded_models]
        avg_pred = np.mean(preds)
        return f"{avg_pred:.2f}"
    except Exception as e:
        print(f"Error processing case: {e}", file=sys.stderr)
        return "ERROR"

def main():
    script_dir = os.path.dirname(os.path.realpath(__file__))

    # Verify input file existence
    private_cases_path = os.path.join(script_dir, PRIVATE_CASES_FILE)
    if not os.path.exists(private_cases_path):
        print(f"Error: {PRIVATE_CASES_FILE} not found at {private_cases_path}", file=sys.stderr)
        sys.exit(1)

    # Load all models once
    models = load_models(script_dir)

    # Read private cases
    try:
        with open(private_cases_path, 'r') as f:
            private_cases = json.load(f)
        print(f"Successfully loaded {len(private_cases)} test cases from {PRIVATE_CASES_FILE}")
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON from {PRIVATE_CASES_FILE}: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Error reading {PRIVATE_CASES_FILE}: {e}", file=sys.stderr)
        sys.exit(1)

    # Process each case and write to output file
    print(f"Processing test cases and saving results to {PRIVATE_ANSWERS_FILE}...")
    with open(PRIVATE_ANSWERS_FILE, 'w') as outfile:
        for i, case in enumerate(private_cases):
            result = featurize_and_predict(case, models)
            outfile.write(f"{result}\n")
            if (i + 1) % 100 == 0 or (i + 1) == len(private_cases):
                print(f"  Processed {i + 1}/{len(private_cases)} cases...")

    print("\n✅ Processing complete!")
    print(f"Results saved to {PRIVATE_ANSWERS_FILE}")
    print("Each line in the output file corresponds to the reimbursement for each case in private_cases.json.")

if __name__ == "__main__":
    main()