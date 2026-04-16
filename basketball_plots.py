from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# Tracked and visualized training curves showing loss and/or metrics over time (3 pts)
def plot_training_curves(history: Dict[str, List[float]], output_path: str, title: str) -> None:
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(10, 6))
    plt.plot(history["train_loss"], label="Train Loss")
    plt.plot(history["val_loss"], label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


# Tracked and visualized training curves showing loss and/or metrics over time (3 pts)
def plot_metric_bars(results_df: pd.DataFrame, metric: str, output_path: str, title: str) -> None:
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    labels = results_df["experiment_name"].tolist()
    values = results_df[metric].to_numpy(dtype=float)

    plt.figure(figsize=(12, 6))
    plt.bar(range(len(labels)), values)
    plt.xticks(range(len(labels)), labels, rotation=45, ha="right")
    plt.ylabel(metric)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def plot_actual_vs_predicted(y_true: np.ndarray, y_pred: np.ndarray, output_path: str, title: str) -> None:
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    true_total = y_true[:, 0] + y_true[:, 1]
    pred_total = y_pred[:, 0] + y_pred[:, 1]

    plt.figure(figsize=(7, 7))
    plt.scatter(true_total, pred_total, alpha=0.35)
    min_val = min(true_total.min(), pred_total.min())
    max_val = max(true_total.max(), pred_total.max())
    plt.plot([min_val, max_val], [min_val, max_val])
    plt.xlabel("Actual Total Points")
    plt.ylabel("Predicted Total Points")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def plot_margin_residuals(y_true: np.ndarray, y_pred: np.ndarray, output_path: str, title: str) -> None:
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    true_margin = y_true[:, 0] - y_true[:, 1]
    pred_margin = y_pred[:, 0] - y_pred[:, 1]
    residuals = pred_margin - true_margin

    plt.figure(figsize=(10, 6))
    plt.hist(residuals, bins=40)
    plt.xlabel("Predicted Margin - Actual Margin")
    plt.ylabel("Frequency")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()
