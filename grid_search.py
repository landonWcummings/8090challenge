#!/usr/bin/env python3
"""
optuna_search.py  –  Bayesian hyperparameter tuning for the legacy-expense model
using Optuna’s TPE sampler.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import optuna


def calc_reimbursement(
    days: int,
    miles: int,
    receipts: float,
    *,
    base_per_diem: float,
    five_day_bonus: float,
    per_mile_rate_first: float,
    per_mile_rate_drop: float,
    receipt_scale: float,
    receipt_diminishing: float,
) -> float:
    """
    Heuristic replica of ACME's black-box formula with six tunable parameters.
    Returns reimbursement rounded to 2 decimals.
    """
    # Per-diem component
    per_diem = base_per_diem * days
    if days == 5:
        per_diem += five_day_bonus

    # Mileage component (two-tier)
    if miles <= 100:
        mileage_pay = miles * per_mile_rate_first
    else:
        mileage_pay = (
            100 * per_mile_rate_first
            + (miles - 100) * per_mile_rate_first * per_mile_rate_drop
        )

    # Receipts component – diminishing returns past $600
    if receipts <= 600:
        reimb_receipts = receipts * receipt_scale
    else:
        reimb_receipts = (
            600 * receipt_scale
            + (receipts - 600)
            * receipt_scale
            / (1 + receipt_diminishing * (receipts - 600))
        )

    return round(per_diem + mileage_pay + reimb_receipts, 2)


def objective(trial: optuna.trial.Trial) -> float:
    # Suggest hyperparameters within the last promising bounds
    base_per_diem = trial.suggest_int("base_per_diem", 40, 60)
    five_day_bonus = trial.suggest_int("five_day_bonus", 100, 300, step=50)
    per_mile_rate_first = trial.suggest_float("per_mile_rate_first", 0.65, 0.85)
    per_mile_rate_drop = trial.suggest_float("per_mile_rate_drop", 0.50, 0.80)
    receipt_scale = trial.suggest_float("receipt_scale", 0.60, 1.00)
    receipt_diminishing = trial.suggest_float("receipt_diminishing", 0.0005, 0.0015)

    # Compute predictions and MAE
    preds = df.apply(
        lambda r: calc_reimbursement(
            int(r.trip_duration_days),
            int(r.miles_traveled),
            float(r.total_receipts_amount),
            base_per_diem=base_per_diem,
            five_day_bonus=five_day_bonus,
            per_mile_rate_first=per_mile_rate_first,
            per_mile_rate_drop=per_mile_rate_drop,
            receipt_scale=receipt_scale,
            receipt_diminishing=receipt_diminishing,
        ),
        axis=1,
    )
    mae = float(np.mean(np.abs(preds - df.expected_output)))
    return mae


def main() -> None:
    if len(sys.argv) != 2:
        print("Usage: python optuna_search.py <csv_file>")
        sys.exit(1)

    csv_path = Path(sys.argv[1])
    if not csv_path.exists():
        print(f"File not found: {csv_path}")
        sys.exit(1)

    global df
    df = pd.read_csv(csv_path)

    # Create and run the Optuna study
    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(),
    )
    study.optimize(objective, n_trials=200)

    # Output results
    print("\nBest MAE: {:.4f}".format(study.best_value))
    print("Best parameters:")
    for key, val in study.best_params.items():
        print(f"  {key:22} = {val}")


if __name__ == "__main__":
    main()
