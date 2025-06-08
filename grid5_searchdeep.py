#!/usr/bin/env python3
"""
grid_search.py – brute-force parameter tuning for the legacy-expense model
"""

import itertools
import sys
from pathlib import Path

import numpy as np
import pandas as pd


# ----------------------------------------------------------------------
# 1. Enhanced parametric formula with interview insights
# ----------------------------------------------------------------------
def calc_reimbursement(
    days: int,
    miles: int,
    receipts: float,
    *,
    # Core parameters
    base_per_diem: float,
    five_day_bonus: float,
    per_mile_rate_first: float,
    per_mile_rate_drop: float,
    receipt_scale: float,
    receipt_diminishing: float,
    
    # Efficiency bonus parameters (can be zeroed)
    eff_threshold: float = 0,        # Miles/day threshold for bonus
    eff_bonus: float = 0,            # Fixed bonus amount
    
    # Spending cap parameters (can be zeroed)
    spend_cap: float = 0,            # Daily spending cap
    spend_cap_penalty: float = 0,    # Penalty rate over cap
    
    # Penalty parameters (can be zeroed)
    low_receipt_penalty: float = 0,  # Penalty for very low receipts
    low_receipt_cutoff: float = 0,   # Receipt cutoff for penalty
    long_trip_penalty: float = 0,    # Penalty for long trips
    long_trip_days: int = 0,         # Day threshold for long trip
    
    # Rounding bonus (can be zeroed)
    rounding_bonus: float = 0,       # Bonus for .49/.99 receipts
) -> float:
    """
    Enhanced heuristic replica of ACME's black-box formula with interview insights.
    All new parameters can be disabled by setting to zero.
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

    # Receipts component - diminishing returns
    if receipts <= 600:
        reimb_receipts = receipts * receipt_scale
    else:
        reimb_receipts = (
            600 * receipt_scale
            + (receipts - 600)
            * receipt_scale
            / (1 + receipt_diminishing * (receipts - 600))
        )

    # ---- Interview-based enhancements ----
    # Efficiency bonus (if parameters are non-zero)
    efficiency_bonus = 0
    if eff_threshold > 0 and eff_bonus > 0:
        miles_per_day = miles / max(days, 1)
        if miles_per_day > eff_threshold:
            efficiency_bonus = eff_bonus

    # Spending cap penalty (if parameters are non-zero)
    spend_penalty = 0
    if spend_cap > 0 and spend_cap_penalty > 0:
        daily_spend = receipts / max(days, 1)
        if daily_spend > spend_cap:
            overage = receipts - (spend_cap * days)
            spend_penalty = overage * spend_cap_penalty

    # Low receipt penalty (if parameters are non-zero)
    receipt_penalty = 0
    if low_receipt_cutoff > 0 and low_receipt_penalty > 0:
        if receipts < low_receipt_cutoff:
            receipt_penalty = low_receipt_penalty

    # Long trip penalty (if parameters are non-zero)
    trip_penalty = 0
    if long_trip_days > 0 and long_trip_penalty > 0:
        if days > long_trip_days:
            trip_penalty = long_trip_penalty

    # Rounding bonus (if parameter is non-zero)
    rounding = 0
    if rounding_bonus > 0:
        cents = int(receipts * 100) % 100
        if cents in [49, 99]:
            rounding = rounding_bonus

    # Final result with enhancements
    total = (
        per_diem 
        + mileage_pay 
        + reimb_receipts 
        + efficiency_bonus 
        + rounding 
        - spend_penalty 
        - receipt_penalty 
        - trip_penalty
    )
    return max(0, round(total, 2))


# ----------------------------------------------------------------------
# 2. Grid search driver with enhanced parameters
# ----------------------------------------------------------------------
def grid_search_legacy(csv_path: str | Path) -> tuple[dict, float]:
    """
    Exhaustive grid search over hand-chosen parameter ranges.
    Returns (best_params_dict, best_mae).
    """

    df = pd.read_csv(csv_path)

    # Core parameter grid (from your best result)
    core_grid = {
        "base_per_diem":       [45, 50, 55],
        "five_day_bonus":      [80, 100, 120],
        "per_mile_rate_first": np.linspace(0.7, 0.8, 5),
        "per_mile_rate_drop":  np.linspace(0.45, 0.55, 5),
        "receipt_scale":       [0.85, 0.9, 0.95],
        "receipt_diminishing": np.linspace(0.001, 0.0012, 5),
    }

    # Enhanced parameters grid (all can be zeroed)
    enhance_grid = {
        # Efficiency bonus
        "eff_threshold": [0, 150, 180],
        "eff_bonus": [0, 40, 80],
        
        # Spending cap
        "spend_cap": [0, 100, 120],
        "spend_cap_penalty": [0, 0.3, 0.5],
        
        # Penalties
        "low_receipt_penalty": [0, 30, 60],
        "low_receipt_cutoff": [0, 15, 30],
        "long_trip_penalty": [0, 100, 200],
        "long_trip_days": [0, 7, 8],
        
        # Rounding bonus
        "rounding_bonus": [0, 0.5, 1.0],
    }

    # Combine grids
    full_grid = {**core_grid, **enhance_grid}
    grid_names = list(full_grid)
    grid_values = list(full_grid.values())

    best_params = None
    best_mae = float("inf")
    total_combos = np.prod([len(v) for v in grid_values])
    combo_count = 0

    print(f"Total parameter combinations: {total_combos:,}")
    print("Starting grid search...")

    # ------- Exhaustive search -------
    for combo in itertools.product(*grid_values):
        combo_count += 1
        if combo_count % 1000 == 0:
            print(f"Progress: {combo_count}/{total_combos} "
                  f"({combo_count/total_combos:.1%}), "
                  f"Best MAE: {best_mae:.2f}")
        
        params = dict(zip(grid_names, combo, strict=True))

        # Vectorized prediction
        preds = df.apply(
            lambda r: calc_reimbursement(
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
            print(f"New best MAE: {best_mae:.4f} at combo {combo_count}")

    return best_params, best_mae


# ----------------------------------------------------------------------
# 3. CLI wrapper
# ----------------------------------------------------------------------
def main() -> None:
    if len(sys.argv) != 2:
        print("Usage: python grid_search.py <csv_file>")
        sys.exit(1)

    csv_path = Path(sys.argv[1])
    if not csv_path.exists():
        print(f"File not found: {csv_path}")
        sys.exit(1)

    best_params, best_mae = grid_search_legacy(csv_path)

    print("\nBest parameter set:")
    for k, v in best_params.items():
        print(f"  {k:22} = {v}")

    print(f"\nBest MAE over {csv_path.name}: {best_mae:,.4f}")


if __name__ == "__main__":
    main()