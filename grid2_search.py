import pandas as pd
import numpy as np
import random
import time
from itertools import product

def calculate_reimbursement(df: pd.DataFrame, params: dict) -> pd.Series:
    D = df['trip_duration_days']
    M = df['miles_traveled']
    R = df['total_receipts_amount']
    E = M / D.replace(0, np.nan)
    S_daily = R / D.replace(0, np.nan)

    # Base
    per_diem = params['p_diem'] * D
    tier1 = np.minimum(M, params['mileage_tier1_cutoff']) * params['mileage_rate_tier1']
    tier2 = np.maximum(0, M - params['mileage_tier1_cutoff']) * params['mileage_rate_tier2']
    mileage = tier1 + tier2

    # Receipts
    s_cap = np.select(
        [D <= 3, (D >= 4) & (D <= 6), D > 6],
        [params['s_cap_short'], params['s_cap_medium'], params['s_cap_long']],
        default=params['s_cap_medium']
    )
    r_opt = s_cap * D
    receipt = np.minimum(R, r_opt) * params['receipt_rate_optimal'] + np.maximum(0, R - r_opt) * params['receipt_rate_overage']

    base = per_diem + mileage + receipt

    # Bonuses
    b5 = np.where(D == 5, params['bonus_5_day'], 0)
    eff_bonus = np.select(
        [(E >= params['eff_low']) & (E <= params['eff_high']),
         (E >= params['eff_low']*0.8) & (E < params['eff_low']),
         (E > params['eff_high']) & (E <= params['eff_high']*1.2)],
        [params['bonus_eff_high'], params['bonus_eff_med'], params['bonus_eff_med']],
        default=0
    )
    cents = R % 1
    b_round = np.where(np.isclose(cents, 0.49) | np.isclose(cents, 0.99), params['bonus_round'], 0)
    bonus = b5 + eff_bonus + b_round

    # Penalties
    p_low = np.where((R > 0) & (R < params['penalty_low_cutoff']), params['penalty_low_amt'], 0)
    p_vac = np.where((D >= 8) & (S_daily > s_cap), params['penalty_vac'], 0)
    p_ineff = np.where(E < params['penalty_ineff_cutoff'], params['penalty_ineff_amt'], 0)
    penalty = p_low + p_vac + p_ineff

    return (base + bonus - penalty).round(2)

# Load data
df = pd.read_csv('public_cases.csv')

# Parameter space
param_space = {
    'p_diem': [90, 95, 100, 105],
    'mileage_tier1_cutoff': [50, 100, 150],
    'mileage_rate_tier1': [0.50, 0.58, 0.65],
    'mileage_rate_tier2': [0.40, 0.45, 0.50],
    'receipt_rate_optimal': [0.80, 0.85, 0.90, 0.95],
    'receipt_rate_overage': [0.20, 0.30, 0.40],
    's_cap_short': [50, 75, 100],
    's_cap_medium': [110, 120, 130],
    's_cap_long': [70, 90, 110],
    'bonus_5_day': [100, 125, 150, 175],
    'bonus_eff_high': [150, 175, 200, 225],
    'bonus_eff_med': [50, 75, 100],
    'eff_low': [150, 180, 200],
    'eff_high': [200, 220, 250],
    'bonus_round': [0, 1.5, 3],
    'penalty_low_cutoff': [10, 25, 50],
    'penalty_low_amt': [25, 40, 50, 60],
    'penalty_vac': [250, 300, 350],
    'penalty_ineff_cutoff': [25, 50, 75],
    'penalty_ineff_amt': [50, 100, 150],
}

# Random search for 5 minutes
best_mae = float('inf')
best_params = None
start_time = time.time()
time_limit = 300  # seconds

# Pre-generate all keys for sampling
keys = list(param_space.keys())

while time.time() - start_time < time_limit:
    # Randomly sample a parameter set
    params = {k: random.choice(param_space[k]) for k in keys}
    # Evaluate
    preds = calculate_reimbursement(df, params)
    mae = (abs(df['expected_output'] - preds)).mean()
    if mae < best_mae:
        best_mae = mae
        best_params = params.copy()

# Report
print(f"Search complete. Best MAE: {best_mae:.4f}")
print("Best parameters:")
for k, v in best_params.items():
    print(f"  {k}: {v}")
