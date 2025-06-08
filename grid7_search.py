#!/usr/bin/env python3
"""
optuna_acme_reimbursement_random.py
----------------------------------
Minimise MAE by randomly sampling the full union search space
for each hyperparameter, rather than using TPE/Bayesian search.
"""

import pandas as pd
import numpy as np
import optuna

# ─────────────────────────────────────────────────────────────────────
# Load data
# ─────────────────────────────────────────────────────────────────────
CSV_PATH = "public_cases.csv"
df = pd.read_csv(CSV_PATH)

D = df["trip_duration_days"].values
M = df["miles_traveled"].values
T = df["total_receipts_amount"].values
Y = df["expected_output"].values


# ─────────────────────────────────────────────────────────────────────
# Helper: sample from a union distribution
# ─────────────────────────────────────────────────────────────────────
def suggest_union_float(trial, name, neg_range, pos_range):
    segment = trial.suggest_categorical(f"{name}_segment", ["neg", "pos"])
    low, high = neg_range if segment == "neg" else pos_range
    return trial.suggest_float(name, low, high)


# ─────────────────────────────────────────────────────────────────────
# Core reimbursement logic (vectorised)
# ─────────────────────────────────────────────────────────────────────
def reimbursement(p):
    R = np.full_like(D, p["base_stipend"], dtype=float)

    # piecewise per-diem
    first7 = np.minimum(D, 7)
    rest   = np.maximum(0, D - 7)
    R += p["per_diem_1"] * first7 + p["per_diem_2"] * rest

    # 5-day bonus & long-trip penalty
    R += np.where(D == 5,  p["bonus_5day"],   0)
    R -= np.where(D >= 8,  p["penalty_long"], 0)

    # tiered mileage
    tier1 = np.minimum(M, p["tier1_cutoff"])
    tier2 = np.maximum(0, M - p["tier1_cutoff"])
    R += p["mile_rate_1"] * tier1 + p["mile_rate_2"] * tier2

    # efficiency bonus
    surplus = np.maximum(0, M - p["eff_thresh"] * D)
    R += p["eff_bonus"] * surplus

    # receipts piecewise
    per_day = T / D
    excess  = np.maximum(0, T - p["rec_thresh"] * D)
    R += np.where(
        per_day <= p["rec_thresh"],
        T,
        p["rec_thresh"] * D + p["rec_over"] * excess
    )

    # rounding glitch
    cents = np.round(T - np.floor(T), 2)
    R += np.where((cents == 0.49) | (cents == 0.99), 0.01, 0.0)

    return np.round(R, 2)


# ─────────────────────────────────────────────────────────────────────
# Optuna objective with random sampling
# ─────────────────────────────────────────────────────────────────────
def objective(trial):
    params = {
        "base_stipend" : suggest_union_float(trial, "base_stipend",
                                             neg_range=(-20, 0),
                                             pos_range=(0, 90)),
        "per_diem_1"   : suggest_union_float(trial, "per_diem_1",
                                             neg_range=(-20, 0),
                                             pos_range=(0, 80)),
        "per_diem_2"   : suggest_union_float(trial, "per_diem_2",
                                             neg_range=(-20, 0),
                                             pos_range=(0, 50)),
        "bonus_5day"   : suggest_union_float(trial, "bonus_5day",
                                             neg_range=(-150, 0),
                                             pos_range=(0, 150)),
        "penalty_long" : suggest_union_float(trial, "penalty_long",
                                             neg_range=(-150, 0),
                                             pos_range=(0, 150)),
        "tier1_cutoff" : suggest_union_float(trial, "tier1_cutoff",
                                             neg_range=(-100, 0),
                                             pos_range=(150, 450)),
        "mile_rate_1"  : suggest_union_float(trial, "mile_rate_1",
                                             neg_range=(-0.10, 0),
                                             pos_range=(0, 0.70)),
        "mile_rate_2"  : suggest_union_float(trial, "mile_rate_2",
                                             neg_range=(-0.10, 0),
                                             pos_range=(0, 0.55)),
        "eff_thresh"   : suggest_union_float(trial, "eff_thresh",
                                             neg_range=(-20, 0),
                                             pos_range=(50, 120)),
        "eff_bonus"    : suggest_union_float(trial, "eff_bonus",
                                             neg_range=(-0.05, 0),
                                             pos_range=(0, 0.15)),
        "rec_thresh"   : suggest_union_float(trial, "rec_thresh",
                                             neg_range=(-20, 0),
                                             pos_range=(50, 100)),
        "rec_over"     : suggest_union_float(trial, "rec_over",
                                             neg_range=(-0.30, 0),
                                             pos_range=(0.20, 0.80)),
    }

    preds = reimbursement(params)
    mae = float(np.mean(np.abs(preds - Y)))
    return mae


# ─────────────────────────────────────────────────────────────────────
# Main entry: run random search
# ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    N_TRIALS = 30000
    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.RandomSampler(seed=42)
    )
    study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=True)

    print("\n=== OPTUNA RESULTS (random search) ===")
    print(f"Best MAE  : {study.best_value:.4f}")
    for k, v in study.best_params.items():
        if not k.endswith("_segment"):
            print(f"{k:<15} = {v}")
