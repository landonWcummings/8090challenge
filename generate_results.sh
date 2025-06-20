#!/bin/bash

# Black Box Challenge - Results Generation Script
# This script runs your implementation against test cases and outputs results to private_results.txt

set -e

echo "🧾 Black Box Challenge - Generating Private Results"
echo "===================================================="
echo

# --- Configuration for Parallel Processing ---
# Set the number of parallel jobs. Adjust this based on your CPU cores and model's resource usage.
# A good starting point is the number of CPU cores you have, or slightly less if models are memory-intensive.
NUM_PARALLEL_JOBS=8 
# ---------------------------------------------

# Check if jq is available
if ! command -v jq &> /dev/null; then
    echo "❌ Error: jq is required but not installed!"
    echo "Please install jq to parse JSON files:"
    echo "  macOS: brew install jq"
    echo "  Ubuntu/Debian: sudo apt-get install jq"
    echo "  CentOS/RHEL: sudo yum install jq"
    exit 1
fi

# Check if run.sh exists
if [ ! -f "run.sh" ]; then
    echo "❌ Error: run.sh not found!"
    echo "Please create a run.sh script that takes three parameters:"
    echo "  ./run.sh <trip_duration_days> <miles_traveled> <total_receipts_amount>"
    echo "  and outputs the reimbursement amount"
    exit 1
fi

# Make run.sh executable
chmod +x run.sh

# Check if private cases exist
if [ ! -f "private_cases.json" ]; then
    echo "❌ Error: private_cases.json not found!"
    echo "Please ensure the private cases file is in the current directory."
    exit 1
fi

echo "📊 Processing test cases and generating results..."
echo "📝 Output will be saved to private_results.txt"
echo

# Extract all test data upfront in a single jq call for better performance
echo "Extracting test data..."
# For parallel processing, we'll output each case with its index for later re-ordering
test_data=$(jq -r 'keys[] as $i | "\($i):\(.[$i].trip_duration_days):\(.[$i].miles_traveled):\(.[$i].total_receipts_amount)"' private_cases.json)

# Remove existing results file if it exists
rm -f private_results.txt

echo "Processing test cases with $NUM_PARALLEL_JOBS parallel jobs..." >&2

# Create a temporary directory for results from parallel processes
mkdir -p ./.tmp_results

# Export the run.sh path for xargs to find it
export PATH="$PATH:$(pwd)"

# Define a wrapper function for processing each case. This allows us to handle errors and output formatting for each individual run.
# This function will be called by xargs.
process_case() {
    local index="$1"
    local trip_duration="$2"
    local miles_traveled="$3"
    local receipts_amount="$4"
    local output_file=".tmp_results/result_$index.txt"

    if script_output=$(./run.sh "$trip_duration" "$miles_traveled" "$receipts_amount" 2>/dev/null); then
        output=$(echo "$script_output" | tr -d '[:space:]')
        if [[ $output =~ ^-?[0-9]+\.?[0-9]*$ ]]; then
            echo "$output" > "$output_file"
        else
            echo "Error on case $((index+1)): Invalid output format: $output" >&2
            echo "ERROR" > "$output_file"
        fi
    else
        error_msg=$(./run.sh "$trip_duration" "$miles_traveled" "$receipts_amount" 2>&1 >/dev/null | tr -d '\n')
        echo "Error on case $((index+1)): Script failed: $error_msg" >&2
        echo "ERROR" > "$output_file"
    fi
}

# Export the function so xargs can use it
export -f process_case

# Use xargs to process cases in parallel
echo "$test_data" | xargs -P "$NUM_PARALLEL_JOBS" -I {} bash -c 'process_case $(echo {} | cut -d: -f1) $(echo {} | cut -d: -f2) $(echo {} | cut -d: -f3) $(echo {} | cut -d: -f4)'

echo "Combining results..."

# Combine results from temporary files in the correct order
total_cases=$(jq '. | length' private_cases.json)
for ((i=0; i<total_cases; i++)); do
    cat ".tmp_results/result_$i.txt" >> private_results.txt
done

# Clean up temporary files
rm -rf ./.tmp_results

echo
echo "✅ Results generated successfully!" >&2
echo "📄 Output saved to private_results.txt" >&2
echo "📊 Each line contains the result for the corresponding test case in private_cases.json" >&2

echo
echo "🎯 Next steps:"
echo "  1. Check private_results.txt - it should contain one result per line"
echo "  2. Each line corresponds to the same-numbered test case in private_cases.json"
echo "  3. Lines with 'ERROR' indicate cases where your script failed"
echo "  4. Submit your private_results.txt file when ready!"
echo
echo "📈 File format:"
echo "  Line 1: Result for private_cases.json[0]"
echo "  Line 2: Result for private_cases.json[1]"
echo "  Line 3: Result for private_cases.json[2]"
echo "  ..."
echo "  Line N: Result for private_cases.json[N-1]"