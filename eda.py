import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Ensure output directories exist
os.makedirs('plots', exist_ok=True)
os.makedirs('data', exist_ok=True)

# Path to your CSV file
csv_path = r'/Users/landoncummings/Desktop/ai proj/8090/top-coder-challenge/public_cases.csv'

# Load historical data from CSV
df = pd.read_csv(csv_path)

# Rename for consistency with previous EDA
df = df.rename(columns={'expected_output': 'reimbursement'})

# Basic overview
print("Dataset shape:", df.shape)
print(df.head())
print(df.describe())

# Distribution plots for each input and the target
features = ['trip_duration_days', 'miles_traveled', 'total_receipts_amount', 'reimbursement']
for col in features:
    plt.figure(figsize=(8, 4))
    sns.histplot(df[col], kde=True)
    plt.title(f'Distribution of {col}')
    plt.tight_layout()
    plt.savefig(f'plots/{col}_distribution.png')
    plt.close()

# Scatter plots: inputs vs reimbursement
inputs = ['trip_duration_days', 'miles_traveled', 'total_receipts_amount']
for col in inputs:
    plt.figure(figsize=(6, 6))
    sns.scatterplot(x=df[col], y=df['reimbursement'])
    plt.title(f'Reimbursement vs {col}')
    plt.xlabel(col)
    plt.ylabel('Reimbursement')
    plt.tight_layout()
    plt.savefig(f'plots/reimb_vs_{col}.png')
    plt.close()

# Pairplot to observe pairwise relationships
sns.pairplot(df[features], kind='reg')
plt.savefig('plots/pairplot_all.png')
plt.close()

# Correlation matrix heatmap
corr = df[features].corr()
plt.figure(figsize=(6, 5))
sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm', linewidths=0.5)
plt.title('Correlation Matrix')
plt.tight_layout()
plt.savefig('plots/correlation_matrix.png')
plt.close()

# Time-based analysis: none available, but if interviews mention thresholds...
def load_interview_hints(path='INTERVIEWS.md'):
    hints = []
    with open(path) as f:
        for line in f:
            if 'threshold' in line.lower():
                hints.append(line.strip())
    return hints

hints = load_interview_hints()
print("Interview hints about thresholds:")
for hint in hints:
    print("-", hint)

# Save cleaned DataFrame for further modeling
df.to_csv('data/clean_public_cases.csv', index=False)

print("EDA complete. Plots saved in 'plots/' directory and cleaned data in 'data/'.")
