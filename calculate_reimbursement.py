#!/usr/bin/env python3
import sys
import numpy as np
import joblib

def main():
    if len(sys.argv) != 4:
        print("Usage: calculate_reimbursement.py <trip_duration_days> <miles_traveled> <total_receipts_amount>")
        sys.exit(1)

    # Parse inputs
    trip_days = float(sys.argv[1])
    miles     = float(sys.argv[2])
    receipts  = float(sys.argv[3])

    # Load model (update path if needed)
    model = joblib.load('models/xgb_reimbursement_model.joblib')

    # Construct feature array (1×3)
    X = np.array([[trip_days, miles, receipts]])

    # Predict and round to 2 decimals
    pred = model.predict(X)[0]
    print(f"{pred:.2f}")

if __name__ == "__main__":
    main()
