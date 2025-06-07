#!/usr/bin/env python3
"""
grid_search.py  –  brute-force parameter tuning for the legacy-expense model
Author: (your name)
"""

import itertools
import sys
from pathlib import Path

import numpy as np
import pandas as pd


# ----------------------------------------------------------------------
# 1.  Canonical parametric formula
# ----------------------------------------------------------------------
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
    Returns a single float reimbursement rounded to 2 decimals (like the legacy system).

    Parameters
    ----------
    days, miles, receipts : trip inputs
    base_per_diem         : flat $/day               (≈ 100)
    five_day_bonus        : extra bump if days == 5  (≈ 25–75)
    per_mile_rate_first   : $/mile for first 100 mi  (≈ 0.45–0.65)
    per_mile_rate_drop    : multiplier after 100 mi  (≈ 0.5–1.0)
    receipt_scale         : baseline % reimbursed    (≈ 0.6–1.0)
    receipt_diminishing   : curvature beyond $600    (≈ 0.0005–0.002)

    Returns
    -------
    float  – reimbursement (2-decimals)
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

    # Receipts component – diminishing returns past the “sweet spot” ($600-800)
    if receipts <= 600:
        reimb_receipts = receipts * receipt_scale
    else:
        reimb_receipts = (
            600 * receipt_scale
            + (receipts - 600)
            * receipt_scale
            / (1 + receipt_diminishing * (receipts - 600))
        )

    # Final result – legacy system always rounds to nearest cent
    return round(per_diem + mileage_pay + reimb_receipts, 2)


# ----------------------------------------------------------------------
# 2.  Grid search driver
# ----------------------------------------------------------------------
def grid_search_legacy(csv_path: str | Path) -> tuple[dict, float]:
    """
    Exhaustive grid search over hand-chosen parameter ranges.
    Returns (best_params_dict, best_mae).
    """

    df = pd.read_csv(csv_path)

    # ------- Hyper-parameter ranges (edit freely) -------
        # original: 5 * 5 * 5 * 6 * 5 * 4 = 15 000
    grid = {
        # best ≈50 → search ±10 in 5 steps
        "base_per_diem":       [40, 45, 50, 55, 60],

        # best=100 → search 100–300 in 5 steps
        "five_day_bonus":      [100, 150, 200, 250, 300],

        # best≈0.75 → finer around [0.65–0.85] in 10 steps
        "per_mile_rate_first": np.linspace(0.65, 0.85, 10),

        # best=0.50 → span 0.50–0.80 in 6 steps
        "per_mile_rate_drop":  np.linspace(0.50, 0.80, 6),

        # best=0.90 → span 0.60–1.00 in 5 steps
        "receipt_scale":       [0.60, 0.70, 0.80, 0.90, 1.00],

        # best≈0.0011 → span 0.0005–0.0015 in 5 steps
        "receipt_diminishing": np.linspace(0.0005, 0.0015, 5),
    }




    grid_names = list(grid)
    grid_values = list(grid.values())

    best_params = None
    best_mae = float("inf")

    # ------- Exhaustive search -------
    for combo in itertools.product(*grid_values):
        params = dict(zip(grid_names, combo, strict=True))

        # Vectorised prediction – fast enough for a few 10^4 combos
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

    return best_params, best_mae


# ----------------------------------------------------------------------
# 3.  CLI wrapper
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
