#!/usr/bin/env python3
"""
hybrid_model_search.py – A more advanced parameter search for the legacy-expense model.

This script implements a "Hybrid Engine" theory, which combines discrete rules with
smooth mathematical functions to better model the system's nuances.
- Mileage reimbursement uses a Gaussian bell curve to model the "efficiency sweet spot".
- Receipt reimbursement uses a dynamic threshold based on trip duration.
- A smooth penalty is applied for very low receipt totals.
"""

import itertools
import sys
import math
from pathlib import Path
import numpy as np
import pandas as pd

def calc_reimbursement_hybrid(
    days: int,
    miles: int,
    receipts: float,
    *,
    # Per Diem Engine (2 params)
    p_base_diem: float,
    p_five_day_bonus: float,
    # Mileage Engine (4 params)
    p_base_mileage_rate: float,
    p_efficiency_sweet_spot: float,
    p_efficiency_bonus_max: float,
    p_efficiency_width: float,
    # Receipt Engine (5 params)
    p_receipt_threshold_daily: float,
    p_receipt_scale: float,
    p_receipt_diminishing: float,
    p_low_receipt_penalty: float,
    p_low_receipt_falloff: float,
) -> float:
    """Calculates reimbursement based on the Hybrid Engine model."""

    # --- 1. Per Diem Engine ---
    per_diem_pay = p_base_diem * days
    if days == 5:
        per_diem_pay += p_five_day_bonus

    # --- 2. Mileage Engine ---
    # The core idea is a base rate plus a bonus multiplier derived from a bell curve.
    efficiency = miles / days if days > 0 else 0
    
    # Gaussian function for the efficiency bonus multiplier
    exponent = -((efficiency - p_efficiency_sweet_spot) ** 2) / (2 * p_efficiency_width ** 2)
    bonus_multiplier = p_efficiency_bonus_max * math.exp(exponent)
    
    mileage_pay = (miles * p_base_mileage_rate) * (1 + bonus_multiplier)

    # --- 3. Receipt Engine ---
    # A dynamic threshold and a smooth penalty for low values.
    total_receipt_threshold = p_receipt_threshold_daily * days

    if receipts <= total_receipt_threshold:
        reimb_receipts = receipts * p_receipt_scale
    else:
        overage = receipts - total_receipt_threshold
        # Diminishing returns formula applied only to the overage amount
        reimb_receipts = (total_receipt_threshold * p_receipt_scale) + \
                         (overage * p_receipt_scale / (1 + p_receipt_diminishing * overage))

    # Add a smooth penalty for very low receipts that fades as receipts increase
    # The term is e.g., -50 * exp(-R/10), which is -50 at R=0 and approx -18 at R=10.
    low_receipt_penalty_val = p_low_receipt_penalty * math.exp(-receipts / p_low_receipt_falloff)
    
    receipt_pay = reimb_receipts - low_receipt_penalty_val
    
    # --- Final Summation ---
    return round(per_diem_pay + mileage_pay + receipt_pay, 2)


def grid_search_hybrid(csv_path: str | Path) -> tuple[dict, float]:
    """Performs a grid search to find the best parameters for the hybrid model."""
    df = pd.read_csv(csv_path)

    # A grid of 11 parameters to explore
    grid = {
        "p_base_diem":               [50, 75, 100],
        "p_five_day_bonus":          [50, 100, 150],
        "p_base_mileage_rate":       np.linspace(0.40, 0.60, 3),
        "p_efficiency_sweet_spot":   [180, 200, 220], # Centered around Kevin's observation
        "p_efficiency_bonus_max":    np.linspace(0.1, 0.4, 4), # Max bonus of 10%-40%
        "p_efficiency_width":        [25, 50], # How "wide" the sweet spot is
        "p_receipt_threshold_daily": [60, 80],
        "p_receipt_scale":           np.linspace(0.8, 1.0, 3),
        "p_receipt_diminishing":     [0.001, 0.002],
        "p_low_receipt_penalty":     [40, 60], # The max penalty at $0 receipts
        "p_low_receipt_falloff":     [10, 20], # How quickly the penalty fades
    }
    names, values = list(grid.keys()), list(grid.values())

    total = np.prod([len(v) for v in values])
    print(f"Starting hybrid grid search with {total:,} combinations...")

    best_params = None
    best_mae = float("inf")

    # Use a simplified progress tracker
    for i, combo in enumerate(itertools.product(*values)):
        if (i + 1) % 1000 == 0:
            print(f"\rProgress: {i+1}/{total}...", end="")
            sys.stdout.flush()

        params = dict(zip(names, combo, strict=True))
        preds = df.apply(
            lambda r: calc_reimbursement_hybrid(
                int(r["trip_duration_days"]),
                int(r["miles_traveled"]),
                float(r["total_receipts_amount"]),
                **params,
            ),
            axis=1,
        )
        mae = np.mean(np.abs(preds - df["expected_output"]))

        if mae < best_mae:
            best_mae = mae
            best_params = params

    print(f"\rProgress: 100% ({total}/{total})\n")
    return best_params, best_mae


def main() -> None:
    if len(sys.argv) != 2:
        print("Usage: python hybrid_model_search.py <csv_file>")
        sys.exit(1)

    csv_path = Path(sys.argv[1])
    if not csv_path.exists():
        print(f"File not found: {csv_path}")
        sys.exit(1)

    best_params, best_mae = grid_search_hybrid(csv_path)

    print("\nBest parameter set for Hybrid Engine model:")
    for k, v in best_params.items():
        print(f"  {k:28} = {v}")
    print(f"\nBest MAE over {csv_path.name}: {best_mae:,.4f}")


if __name__ == "__main__":
    main()