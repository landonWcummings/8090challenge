import pandas as pd
import numpy as np
import optuna
import time

# Load data
# Ensure 'public_cases.csv' is in the same directory or provide the full path
try:
    df = pd.read_csv('public_cases.csv')
except FileNotFoundError:
    print("Error: 'public_cases.csv' not found. Please make sure the file is in the correct directory.")
    # Create a dummy DataFrame for demonstration if the file is not found
    print("Creating a dummy DataFrame for demonstration purposes.")
    data = {
        'trip_duration_days': np.random.randint(1, 10, 100),
        'miles_traveled': np.random.randint(10, 500, 100),
        'total_receipts_amount': np.random.uniform(10, 500, 100),
        'expected_output': np.random.uniform(100, 1000, 100)
    }
    df = pd.DataFrame(data)


def calculate_reimbursement(df: pd.DataFrame, params: dict) -> pd.Series:
    """
    Calculates the reimbursement amount based on various parameters.

    Args:
        df (pd.DataFrame): DataFrame containing trip details.
        params (dict): Dictionary of parameters for reimbursement calculation.

    Returns:
        pd.Series: Calculated reimbursement amounts for each trip.
    """
    D = df['trip_duration_days']
    M = df['miles_traveled']
    R = df['total_receipts_amount']
    # Handle division by zero for trip_duration_days
    E = M / D.replace(0, np.nan)
    S_daily = R / D.replace(0, np.nan)

    # Base Reimbursement
    per_diem = params['p_diem'] * D
    tier1 = np.minimum(M, params['mileage_tier1_cutoff']) * params['mileage_rate_tier1']
    tier2 = np.maximum(0, M - params['mileage_tier1_cutoff']) * params['mileage_rate_tier2']
    mileage = tier1 + tier2

    # Receipts Reimbursement
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

def objective(trial):
    """
    Objective function for Optuna to minimize. It calculates the Mean Absolute Error (MAE)
    for a given set of suggested parameters.

    Args:
        trial (optuna.trial.Trial): A trial object from Optuna.

    Returns:
        float: The Mean Absolute Error.
    """
    # Suggest parameters with ranges, using 'step' to encourage multiples of 5/10.
    # Optuna's TPE sampler will learn to focus on promising regions within these bounds.
    params = {
        'p_diem': trial.suggest_int('p_diem', 80, 120, step=5),
        'mileage_tier1_cutoff': trial.suggest_int('mileage_tier1_cutoff', 40, 200, step=10),
        'mileage_rate_tier1': trial.suggest_float('mileage_rate_tier1', 0.45, 0.70, step=0.01),
        'mileage_rate_tier2': trial.suggest_float('mileage_rate_tier2', 0.35, 0.55, step=0.01),
        'receipt_rate_optimal': trial.suggest_float('receipt_rate_optimal', 0.75, 0.99, step=0.01),
        'receipt_rate_overage': trial.suggest_float('receipt_rate_overage', 0.15, 0.45, step=0.01),
        's_cap_short': trial.suggest_int('s_cap_short', 40, 120, step=10),
        's_cap_medium': trial.suggest_int('s_cap_medium', 100, 150, step=10),
        's_cap_long': trial.suggest_int('s_cap_long', 60, 130, step=10),
        'bonus_5_day': trial.suggest_int('bonus_5_day', 90, 250, step=25),
        'bonus_eff_high': trial.suggest_int('bonus_eff_high', 140, 300, step=20),
        'bonus_eff_med': trial.suggest_int('bonus_eff_med', 40, 150, step=10),
        'eff_low': trial.suggest_int('eff_low', 140, 220, step=10),
        'eff_high': trial.suggest_int('eff_high', 190, 280, step=10),
        'bonus_round': trial.suggest_float('bonus_round', 0, 5, step=0.5),
        'penalty_low_cutoff': trial.suggest_int('penalty_low_cutoff', 5, 75, step=5),
        'penalty_low_amt': trial.suggest_int('penalty_low_amt', 20, 80, step=10),
        'penalty_vac': trial.suggest_int('penalty_vac', 200, 400, step=50),
        'penalty_ineff_cutoff': trial.suggest_int('penalty_ineff_cutoff', 20, 100, step=10),
        'penalty_ineff_amt': trial.suggest_int('penalty_ineff_amt', 40, 200, step=20),
    }
    preds = calculate_reimbursement(df, params)
    mae = np.mean(np.abs(df['expected_output'] - preds))
    return mae

def print_progress_callback(study, trial):
    """
    Callback function to print optimization progress every 1000 trials.
    """
    if trial.number % 1000 == 0:
        print(f"Trial {trial.number}: Current best MAE = {study.best_value:.4f}")

if __name__ == "__main__":
    # Create an Optuna study to minimize the MAE
    study = optuna.create_study(direction="minimize")

    # Optimize for 15,000 trials, using the callback for progress updates
    print("Starting Optuna optimization...")
    study.optimize(objective, n_trials=15000, callbacks=[print_progress_callback])

    print("\nOptimization finished!")
    print("---")
    print("Best MAE: {:.4f}".format(study.best_value))
    print("Best trial parameters:")
    for key, val in study.best_trial.params.items():
        print(f"  {key}: {val}")