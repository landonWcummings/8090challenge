#!/usr/bin/env python3
"""
feature_explorer.py

Keeps your six “champion” parameters fixed, then exhaustively
evaluates the impact of every interview‐inspired enhancement—
both individually and in combination—by brute‐forcing small grids
for each new parameter. Reports any MAE improvements over the
baseline.

Usage:
    python feature_explorer.py public_cases.csv
"""
import itertools
import sys
from pathlib import Path

import numpy as np
import pandas as pd


# ─────────────────────────────────────────────────────────────────────
# 1) Load data & cement the baseline parameters
# ─────────────────────────────────────────────────────────────────────
if len(sys.argv) != 2:
    print("Usage: python feature_explorer.py public_cases.csv")
    sys.exit(1)

CSV = Path(sys.argv[1])
if not CSV.exists():
    print(f"File not found: {CSV}")
    sys.exit(1)

df = pd.read_csv(CSV)
D = df["trip_duration_days"].astype(float).values
M = df["miles_traveled"].astype(float).values
T = df["total_receipts_amount"].astype(float).values
Y = df["expected_output"].astype(float).values

# dummy weekday/EoQ arrays (all zeros—replace if you have dates)
weekday_idx = np.zeros_like(D)
eoq_idx     = np.zeros_like(D)

# fixed “champion” params
BASE = {
    "base_per_diem":       48.0,
    "five_day_bonus":     100.0,
    "per_mile_rate_first": 0.6681975816074796,
    "per_mile_rate_drop":  0.5740571156003343,
    "receipt_scale":       0.9391470633911543,
    "receipt_diminishing": 0.001239189954927641,
}


# ─────────────────────────────────────────────────────────────────────
# 2) Enhanced reimbursement formula
# ─────────────────────────────────────────────────────────────────────
def calc_reimbursement(
    days, miles, receipts,
    *,
    # core (from BASE)
    base_per_diem, five_day_bonus,
    per_mile_rate_first, per_mile_rate_drop,
    receipt_scale, receipt_diminishing,
    # enhancements
    eff_threshold=0, eff_bonus=0,
    spend_cap=0, spend_cap_penalty=0,
    low_receipt_cutoff=0, low_receipt_penalty=0,
    long_trip_days=0, long_trip_penalty=0,
    rounding_bonus=0,
    weekday_bump=0,    # Tue/Thu bump
    eoq_bump=0,        # end-of-quarter bump
    weekday_flag=0,    # scalar 1/0
    eoq_flag=0,        # scalar 1/0
):
    total = base_per_diem * days + (five_day_bonus if days == 5 else 0)

    # two-tier mileage
    tier1 = min(miles, 100)
    tier2 = max(0.0, miles - 100)
    total += tier1 * per_mile_rate_first
    total += tier2 * per_mile_rate_first * per_mile_rate_drop

    # receipts diminishing
    if receipts <= 600:
        total += receipts * receipt_scale
    else:
        total += (600 * receipt_scale
                  + (receipts - 600) * receipt_scale
                    / (1 + receipt_diminishing * (receipts - 600)))

    # efficiency bonus
    if eff_threshold > 0 and eff_bonus > 0:
        if (miles / max(days, 1)) > eff_threshold:
            total += eff_bonus

    # spending cap penalty
    if spend_cap > 0 and spend_cap_penalty > 0:
        if (receipts / max(days, 1)) > spend_cap:
            over = receipts - spend_cap * days
            total -= over * spend_cap_penalty

    # low receipt penalty
    if low_receipt_cutoff > 0 and low_receipt_penalty > 0:
        if receipts < low_receipt_cutoff:
            total -= low_receipt_penalty

    # long trip penalty
    if long_trip_days > 0 and long_trip_penalty > 0:
        if days > long_trip_days:
            total -= long_trip_penalty

    # rounding bonus
    if rounding_bonus > 0:
        cents = int(receipts * 100) % 100
        if cents in (49, 99):
            total += rounding_bonus

    # weekday & EoQ bumps
    total += weekday_bump * weekday_flag
    total += eoq_bump     * eoq_flag

    return round(max(0.0, total), 2)


# ─────────────────────────────────────────────────────────────────────
# 3) Baseline MAE (no enhancements)
# ─────────────────────────────────────────────────────────────────────
baseline_preds = [
    calc_reimbursement(D[i], M[i], T[i], **BASE)
    for i in range(len(D))
]
baseline_mae = float(np.mean(np.abs(np.array(baseline_preds) - Y)))
print(f"\nBaseline MAE (core only): {baseline_mae:.4f}\n")


# ─────────────────────────────────────────────────────────────────────
# 4) Define enhancement grids
# ─────────────────────────────────────────────────────────────────────
enhancements = {
    "efficiency": {
        # threshold: 150→180 in 4‐step increments of 10
        "eff_threshold": [0, 150, 160, 170, 180],
        # bonus: 40→80 in 3‐step increments of 20
        "eff_bonus":     [0, 40, 60, 80],
    },
    "spend_cap": {
        # cap: 100→120 in 3‐step increments of 10
        "spend_cap":         [0, 100, 110, 120],
        # penalty: 0.3→0.5 in 3‐step increments of 0.1
        "spend_cap_penalty": [0, 0.30, 0.40, 0.50],
    },
    "low_receipts": {
        # cutoff: 15→30 in 3‐step increments of 5
        "low_receipt_cutoff": [0, 15, 20, 25, 30],
        # penalty: 30→60 in 3‐step increments of 15
        "low_receipt_penalty":[0, 30, 45, 60],
    },
    "long_trip": {
        # days: test 7,8,9
        "long_trip_days":    [0, 7, 8, 9],
        # penalty: 100→200 in 3‐step of 50
        "long_trip_penalty": [0, 100, 150, 200],
    },
    "rounding": {
        # rounding bonus: 0.5→1.0 in 3‐step of 0.25
        "rounding_bonus": [0, 0.50, 0.75, 1.00],
    },
    "calendar": {
        # weekday bump: 0→5 in 6‐step of 1
        "weekday_bump": list(range(0, 6)),       # [0,1,2,3,4,5]
        # EOQ bump: 0→5 in 6‐step of 1
        "eoq_bump":     list(range(0, 6)),       # [0,1,2,3,4,5]
    },
}



# ─────────────────────────────────────────────────────────────────────
# 5) Tune each enhancement in isolation
# ─────────────────────────────────────────────────────────────────────
def tune(feature_name, grid):
    best = (None, baseline_mae)
    print(f"→ Tuning {feature_name}…")
    keys, lists = zip(*grid.items())
    for combo in itertools.product(*lists):
        params = dict(zip(keys, combo))
        preds = []
        for i in range(len(D)):
            preds.append(calc_reimbursement(
                D[i], M[i], T[i],
                **BASE,
                **params,
                weekday_flag=weekday_idx[i],
                eoq_flag=eoq_idx[i],
            ))
        mae = float(np.mean(np.abs(np.array(preds) - Y)))
        if mae < best[1]:
            best = (params, mae)
    status = "(improved)" if best[1] < baseline_mae else "(no change)"
    print(f"  Best {feature_name}: {best[0]}, MAE={best[1]:.4f} {status}\n")
    return best

results = {}
for name, grid in enhancements.items():
    results[name] = tune(name, grid)


# ─────────────────────────────────────────────────────────────────────
# 6) Combined search over all enhancements together
# ─────────────────────────────────────────────────────────────────────
print("→ Tuning all enhancements jointly…")
all_keys, all_lists = zip(*[
    item for sub in enhancements.values() for item in sub.items()
])
best_all = (None, baseline_mae)
for combo in itertools.product(*all_lists):
    params = dict(zip(all_keys, combo))
    preds = [
        calc_reimbursement(
            D[i], M[i], T[i],
            **BASE,
            **params,
            weekday_flag=weekday_idx[i],
            eoq_flag=eoq_idx[i],
        )
        for i in range(len(D))
    ]
    mae = float(np.mean(np.abs(np.array(preds) - Y)))
    if mae < best_all[1]:
        best_all = (params, mae)
status = "(improved)" if best_all[1] < baseline_mae else "(no change)"
print(f"  Best joint enhancements: {best_all[0]}, MAE={best_all[1]:.4f} {status}\n")


# ─────────────────────────────────────────────────────────────────────
# 7) Summary
# ─────────────────────────────────────────────────────────────────────
print("=== Summary of tuning results ===")
print(f"Baseline MAE: {baseline_mae:.4f}")
for name, (params, mae) in results.items():
    print(f"{name:12s} → MAE {mae:.4f}, params={params}")
print(f"{'joint':12s} → MAE {best_all[1]:.4f}, params={best_all[0]}")
