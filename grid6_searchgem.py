import pandas as pd
import numpy as np
import optuna
import os
import time
import logging # For better logging control

# Suppress Optuna logs unless there's a warning or error
optuna.logging.set_verbosity(optuna.logging.WARNING)

# Your calculate_reimbursement function (unchanged)
def calculate_reimbursement(df: pd.DataFrame, params: dict) -> pd.Series:
    D = df['trip_duration_days'].copy()
    M = df['miles_traveled'].copy()
    R = df['total_receipts_amount'].copy()

    E = M.divide(D).fillna(0)
    S_daily = R.divide(D).fillna(0)

    # a) Per Diem Calculation
    per_diem_total = params['p_diem'] * D

    # b) Tiered Mileage Calculation
    miles_tier1 = np.minimum(M, params['mileage_tier1_cutoff'])
    miles_tier2 = np.maximum(0, M - params['mileage_tier1_cutoff'])
    mileage_calc = (miles_tier1 * params['mileage_rate_tier1']) + (miles_tier2 * params['mileage_rate_tier2'])

    # c) Receipt Calculation with Diminishing Returns
    s_cap_conditions = [
        D <= 3,
        (D >= 4) & (D <= 6),
        D > 6
    ]
    s_cap_values = [params['s_cap_short'], params['s_cap_medium'], params['s_cap_long']]
    s_cap = np.select(s_cap_conditions, s_cap_values, default=params['s_cap_medium']) # Default if no condition matches
    
    r_optimal = s_cap * D
    receipts_under_optimal = np.minimum(R, r_optimal)
    receipts_over_optimal = np.maximum(0, R - r_optimal)
    receipt_calc = (receipts_under_optimal * params['receipt_rate_optimal']) + \
                   (receipts_over_optimal * params['receipt_rate_overage'])

    reimbursement_base = per_diem_total + mileage_calc + receipt_calc

    # --- 3. "EFFECTS" (Previously Bonuses) ---
    effect_5_day = np.where(D == 5, params['effect_5_day'], 0)

    eff_bonus_conditions = [
        (E >= params['efficiency_sweet_spot_low']) & (E <= params['efficiency_sweet_spot_high']),
        (E >= 150) & (E < params['efficiency_sweet_spot_low']), # Assuming 150 is a lower bound for 'medium'
        (E > params['efficiency_sweet_spot_high']) & (E <= 250) # Assuming 250 is an upper bound for 'medium'
    ]
    eff_bonus_values = [params['effect_efficiency_high'], params['effect_efficiency_medium'], params['effect_efficiency_medium']]
    effect_efficiency = np.select(eff_bonus_conditions, eff_bonus_values, default=0)

    r_cents = R % 1
    effect_rounding = np.where(np.isclose(r_cents, 0.49) | np.isclose(r_cents, 0.99), params['effect_rounding'], 0)
    effects_part1 = effect_5_day + effect_efficiency + effect_rounding

    # --- 4. "EFFECTS" (Previously Penalties) ---
    effect_low_receipt = np.where((R > 0) & (R < params['penalty_low_receipt_cutoff']), params['effect_low_receipt'], 0)
    effect_vacation = np.where((D >= 8) & (S_daily > s_cap), params['effect_vacation'], 0)
    effect_inefficiency = np.where(E < params['penalty_inefficiency_cutoff'], params['effect_inefficiency'], 0)
    
    effects_part2 = effect_low_receipt + effect_vacation + effect_inefficiency

    # --- 5. FINAL CALCULATION ---
    total_reimbursement = reimbursement_base + effects_part1 - effects_part2
    return total_reimbursement.round(2)

# --- SCRIPT EXECUTION ---
if __name__ == "__main__":
    # Create a dummy file if one isn't available
    if not os.path.exists('public_cases.csv'):
        print("Creating dummy 'public_cases.csv' for demonstration...")
        dummy_data = {
            'trip_duration_days': [5, 3, 8, 5, 1, 6, 2, 9, 4, 7],
            'miles_traveled': [1050, 200, 400, 250, 30, 800, 150, 1200, 500, 950],
            'total_receipts_amount': [550.49, 150.75, 900.00, 12.00, 80.00, 400.00, 75.00, 1100.00, 250.00, 600.00],
            'expected_output': [1437.24, 432.14, 1280.00, 500.00, 117.4, 980.00, 280.00, 1500.00, 620.00, 1100.00]
        }
        pd.DataFrame(dummy_data).to_csv('public_cases.csv', index=False)

    print("Loading historical data from 'public_cases.csv'...")
    df_cases = pd.read_csv('public_cases.csv')

    def objective(trial: optuna.Trial) -> float:
        """
        Defines the objective function for Optuna optimization.
        It samples hyperparameters and returns the Mean Absolute Error (MAE).
        """
        params = {
            # Base Reimbursement Parameters
            'p_diem': trial.suggest_float('p_diem', 50.0, 125.0),
            'mileage_tier1_cutoff': trial.suggest_float('mileage_tier1_cutoff', 50.0, 150.0),
            'mileage_rate_tier1': trial.suggest_float('mileage_rate_tier1', 0.4, 0.7),
            'mileage_rate_tier2': trial.suggest_float('mileage_rate_tier2', 0.35, 0.75),
            'receipt_rate_optimal': trial.suggest_float('receipt_rate_optimal', 0.70, 1.0),
            'receipt_rate_overage': trial.suggest_float('receipt_rate_overage', 0.1, 0.5),
            's_cap_short': trial.suggest_float('s_cap_short', 50.0, 100.0),
            's_cap_medium': trial.suggest_float('s_cap_medium', 100.0, 180.0),
            's_cap_long': trial.suggest_float('s_cap_long', 70.0, 120.0),
            
            # Effect Parameters (can be positive or negative)
            'effect_5_day': trial.suggest_float('effect_5_day', -100.0, 200.0),
            'effect_efficiency_medium': trial.suggest_float('effect_efficiency_medium', -50.0, 100.0),
            'effect_efficiency_high': trial.suggest_float('effect_efficiency_high', -150.0, 250.0),
            'effect_rounding': trial.suggest_float('effect_rounding', -5.0, 5.0),
            'effect_low_receipt': trial.suggest_float('effect_low_receipt', -50.0, 50.0),
            'effect_vacation': trial.suggest_float('effect_vacation', -300.0, 300.0),
            'effect_inefficiency': trial.suggest_float('effect_inefficiency', -50.0, 150.0),

            # Cutoff Parameters
            'efficiency_sweet_spot_low': trial.suggest_float('efficiency_sweet_spot_low', 150.0, 200.0),
            'efficiency_sweet_spot_high': trial.suggest_float('efficiency_sweet_spot_high', 200.0, 250.0),
            'penalty_low_receipt_cutoff': trial.suggest_float('penalty_low_receipt_cutoff', 10.0, 40.0),
            'penalty_inefficiency_cutoff': trial.suggest_float('penalty_inefficiency_cutoff', 30.0, 70.0),
        }
        
        # --- Parameter Relationship Check (Optional but Recommended) ---
        # Ensure logical constraints: low <= high
        # This reduces "impossible" combinations that Optuna might try
        if params['efficiency_sweet_spot_low'] >= params['efficiency_sweet_spot_high']:
            # Assign a very high MAE to penalize illogical parameter combinations
            return float('inf') 
        
        # You might also want to ensure receipt_rate_optimal > receipt_rate_overage etc.
        if params['receipt_rate_optimal'] <= params['receipt_rate_overage']:
            return float('inf')

        calculated_output = calculate_reimbursement(df_cases, params)
        mae = np.mean(np.abs(df_cases['expected_output'] - calculated_output))
        return mae

    # --- Optuna Study Configuration ---
    # Use a SQLite database for storage. This allows:
    # 1. Resuming the study later if it's stopped.
    # 2. Running multiple optimization processes in parallel (e.g., on different CPU cores).
    storage_name = "sqlite:///reimbursement_optimization.db"
    study_name = "reimbursement_hyperparams"

    # If the study already exists, load it; otherwise, create a new one.
    try:
        study = optuna.load_study(study_name=study_name, storage=storage_name)
        print(f"Resuming existing study: {study_name}")
    except KeyError:
        study = optuna.create_study(study_name=study_name, storage=storage_name, direction="minimize")
        print(f"Created new study: {study_name}")

    print("\nStarting Optuna optimization...")
    # You can set either n_trials (number of evaluations) or timeout (duration in seconds)
    # A good starting point might be a few thousand trials.
    # Let's target a relatively high number of trials given the large space.
    num_trials_to_run = 2000 # Example: Run 10,000 trials

    start_time = time.time()
    try:
        study.optimize(objective, n_trials=num_trials_to_run, show_progress_bar=True)
    except KeyboardInterrupt:
        print("\nOptimization interrupted by user.")
    end_time = time.time()

    print(f"\n--- Optuna Optimization Complete (or interrupted) ---")
    print(f"Total time elapsed: {end_time - start_time:.2f} seconds")
    print(f"Number of finished trials: {len(study.trials)}")
    
    # Filter out trials that returned 'inf' due to illogical parameter combinations
    valid_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE and t.value != float('inf')]

    if valid_trials:
        # Find the best trial among the valid ones
        best_trial_valid = min(valid_trials, key=lambda t: t.value)
        print(f"Best valid trial found:")
        print(f"  Value (MAE): ${best_trial_valid.value:.4f}")
        print("  Optimal Hyperparameters:")
        for key, value in best_trial_valid.params.items():
            print(f"    '{key}': {value},")
    else:
        print("No valid trials completed successfully.")

    # You can also analyze the study results further
    # print("\nPlotting optimization history...")
    # fig = optuna.visualization.plot_optimization_history(study)
    # fig.show()

    # print("\nPlotting parameter importances...")
    # fig = optuna.visualization.plot_param_importances(study)
    # fig.show()