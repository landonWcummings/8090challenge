import json
import pandas as pd

# Path to your JSON file
json_path = r'/Users/landoncummings/Desktop/ai proj/8090/top-coder-challenge/public_cases.json'

# Load JSON
with open(json_path, 'r') as f:
    records = json.load(f)

# Normalize into flat records
flat = []
for rec in records:
    inp = rec['input']
    flat.append({
        'trip_duration_days': inp['trip_duration_days'],
        'miles_traveled':      inp['miles_traveled'],
        'total_receipts_amount': inp['total_receipts_amount'],
        'expected_output':     rec['expected_output']
    })

# Create DataFrame and save CSV
df = pd.DataFrame(flat)
df.to_csv('public_cases.csv', index=False)

print("Wrote", len(df), "rows to public_cases.csv")
