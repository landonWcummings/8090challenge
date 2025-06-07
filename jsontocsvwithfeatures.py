import json
import pandas as pd
import numpy as np  # add numpy import

# Path to your JSON file
json_path = r'C:\Users\lando\Desktop\AI\8090challenge\public_cases.json'

# Load JSON
with open(json_path, 'r') as f:
    records = json.load(f)

# Normalize into flat records and compute features
flat = []
for rec in records:
    inp = rec['input']
    days = inp['trip_duration_days']
    miles = inp['miles_traveled']
    receipts = inp['total_receipts_amount']
    
    # Basic features
    miles_per_day = miles / days if days else 0
    receipts_per_day = receipts / days if days else 0
    base_per_diem = days * 100  # assumed base rate
    is_five_day = int(days == 5)
    # Tiered mileage: full rate up to 100 mi, reduced rate above
    tier1_miles = min(miles, 100)
    tier2_miles = max(miles - 100, 0)
    
    # Log transforms for non-linear patterns
    log_miles = np.log1p(miles)
    log_receipts = np.log1p(receipts)
    
    # Trip length category
    if days <= 3:
        length_cat = 'short'
    elif days <= 6:
        length_cat = 'medium'
    else:
        length_cat = 'long'
    
    flat.append({
        'trip_duration_days': days,
        'miles_traveled': miles,
        'total_receipts_amount': receipts,
        'expected_output': rec['expected_output'],
        
        # Engineered features
        'miles_per_day': miles_per_day,
        'receipts_per_day': receipts_per_day,
        'base_per_diem': base_per_diem,
        'is_five_day': is_five_day,
        'tier1_miles': tier1_miles,
        'tier2_miles': tier2_miles,
        'log_miles': log_miles,
        'log_receipts': log_receipts,
        'length_cat': length_cat,
    })

# Create DataFrame and save CSV
df = pd.DataFrame(flat)

# One-hot encode length_cat
df = pd.get_dummies(df, columns=['length_cat'], prefix='length')
df = df.round(4)

# Save to CSV
df.to_csv('public_caseswithfeatures.csv', index=False)

print("Wrote", len(df), "rows with engineered features to public_cases_features.csv")
