#!/bin/bash

# Black Box Challenge - Your Implementation
# This script should take three parameters and output the reimbursement amount
# Usage: ./run.sh <trip_duration_days> <miles_traveled> <total_receipts_amount>

if [ "$#" -ne 3 ]; then
  echo "Usage: $0 <trip_duration_days> <miles_traveled> <total_receipts_amount>"
  exit 1
fi

trip_days=$1
miles=$2
receipts=$3

# Call your Python predictor
result=$(python3 calculate_reimbursement.py "$trip_days" "$miles" "$receipts")

# Output just the number
echo "$result"
