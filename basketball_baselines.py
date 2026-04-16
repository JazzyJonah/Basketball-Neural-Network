from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd


# Created baseline model for comparison (e.g., constant prediction, random, simple heuristic) (3 pts)
def random_score_baseline(train_df: pd.DataFrame, test_df: pd.DataFrame, seed: int = 42) -> np.ndarray:
    rng = np.random.default_rng(seed)
    train_team1_scores = train_df["team1pts"].to_numpy(dtype=np.float32)
    train_team2_scores = train_df["team2pts"].to_numpy(dtype=np.float32)

    pred_team1 = rng.choice(train_team1_scores, size=len(test_df), replace=True)
    pred_team2 = rng.choice(train_team2_scores, size=len(test_df), replace=True)
    return np.column_stack([pred_team1, pred_team2]).astype(np.float32)


# Created baseline model for comparison (e.g., constant prediction, random, simple heuristic) (3 pts)
def record_based_baseline(train_df: pd.DataFrame, test_df: pd.DataFrame, seed: int = 42) -> np.ndarray:
    rng = np.random.default_rng(seed)
    mean_total = float((train_df["team1pts"] + train_df["team2pts"]).mean())

    wp1 = test_df["team1_win_pct"].to_numpy(dtype=np.float32)
    wp2 = test_df["team2_win_pct"].to_numpy(dtype=np.float32)
    win_pct_diff = wp1 - wp2

    tie_mask = np.isclose(win_pct_diff, 0.0)
    tie_breaks = rng.choice(np.array([-1.0, 1.0], dtype=np.float32), size=len(test_df))
    adjusted_diff = win_pct_diff.copy()
    adjusted_diff[tie_mask] = 0.05 * tie_breaks[tie_mask]

    margin = np.clip(20.0 * adjusted_diff, -20.0, 20.0)
    pred_team1 = (mean_total + margin) / 2.0
    pred_team2 = (mean_total - margin) / 2.0

    return np.column_stack([pred_team1, pred_team2]).astype(np.float32)


# Modular code design with reusable functions and classes rather than monolithic scripts
def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    y_true = np.asarray(y_true, dtype=np.float32)
    y_pred = np.asarray(y_pred, dtype=np.float32)

    mse = float(np.mean((y_true - y_pred) ** 2))
    mae = float(np.mean(np.abs(y_true - y_pred)))
    rmse = float(np.sqrt(mse))

    true_margin = y_true[:, 0] - y_true[:, 1]
    pred_margin = y_pred[:, 0] - y_pred[:, 1]
    winner_acc = float(np.mean((true_margin > 0) == (pred_margin > 0)))

    true_total = y_true[:, 0] + y_true[:, 1]
    pred_total = y_pred[:, 0] + y_pred[:, 1]
    total_mae = float(np.mean(np.abs(true_total - pred_total)))
    margin_mae = float(np.mean(np.abs(true_margin - pred_margin)))

    return {
        "mse": mse,
        "rmse": rmse,
        "mae": mae,
        "winner_accuracy": winner_acc,
        "total_mae": total_mae,
        "margin_mae": margin_mae,
    }
