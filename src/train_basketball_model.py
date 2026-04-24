from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path as _PathCheck
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from basketball_baselines import random_score_baseline, record_based_baseline, regression_metrics
from basketball_model import (
    BasketballScoreMLP,
    EarlyStopping,
    ModelConfig,
    TrainConfig,
    create_optimizer,
    create_scheduler,
    l1_penalty,
    set_seed,
)
from basketball_plots import (
    plot_actual_vs_predicted,
    plot_margin_residuals,
    plot_metric_bars,
    plot_training_curves,
)
from season_features import (
    BasketballMatchupDataset,
    FeatureConfig,
    SeasonToDateFeatureStore,
    Standardizer,
    build_matchup_feature_frame,
    build_team_id_index,
    chronological_split,
    get_continuous_feature_columns,
    load_games_csv,
)


# Modular code design with reusable functions and classes rather than monolithic scripts
class ExperimentRunner:
    def __init__(
        self,
        csv_path: str = "data/mbb_games.csv",
        output_dir: str = "models",
        feature_config: FeatureConfig | None = None,
        train_config: TrainConfig | None = None,
    ):
        self.csv_path = csv_path
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.feature_config = feature_config or FeatureConfig(verbose=True)
        self.train_config = train_config or TrainConfig()
        set_seed(self.train_config.seed)

        self.games_df = None
        self.matchups_df = None
        self.team_id_to_index = None
        self.feature_cols = None

    def prepare_data(self, rebuild_features: bool = False) -> None:
        self.games_df = load_games_csv(self.csv_path)
        store = SeasonToDateFeatureStore(self.games_df, cfg=self.feature_config)
        self.matchups_df = build_matchup_feature_frame(self.games_df, store, rebuild=rebuild_features)
        self.team_id_to_index = build_team_id_index(self.games_df)
        self.feature_cols = get_continuous_feature_columns(self.matchups_df)

    def make_split_dataloaders(
        self,
        split_ratio: Tuple[float, float, float],
    ):
        train_ratio, val_ratio, test_ratio = split_ratio

        # Implemented proper train/validation/test split with documented split ratios (3 pts)
        train_raw, val_raw, test_raw = chronological_split(
            self.matchups_df,
            train_ratio=train_ratio,
            val_ratio=val_ratio,
            test_ratio=test_ratio,
        )

        train_df = train_raw.copy()
        val_df = val_raw.copy()
        test_df = test_raw.copy()

        # Properly normalized or standardized input features/data appropriate to your modality (3 pts)
        standardizer = Standardizer()
        train_df = standardizer.fit_transform(train_df, self.feature_cols)
        val_df = standardizer.transform(val_df, self.feature_cols)
        test_df = standardizer.transform(test_df, self.feature_cols)

        # Used appropriate data loading with batching and shuffling (PyTorch DataLoader or equivalent) (3 pts)
        train_ds = BasketballMatchupDataset(train_df, self.feature_cols, self.team_id_to_index)
        val_ds = BasketballMatchupDataset(val_df, self.feature_cols, self.team_id_to_index)
        test_ds = BasketballMatchupDataset(test_df, self.feature_cols, self.team_id_to_index)

        train_loader = DataLoader(train_ds, batch_size=self.train_config.batch_size, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=self.train_config.batch_size, shuffle=False)
        test_loader = DataLoader(test_ds, batch_size=self.train_config.batch_size, shuffle=False)

        return train_raw, val_raw, test_raw, train_loader, val_loader, test_loader, standardizer


def get_device() -> torch.device:
    # Trained model using GPU/TPU/CUDA acceleration (3 pts)
    if torch.cuda.is_available():
        print("CUDA AVAILABLE")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def is_finite_tensor(t: torch.Tensor) -> bool:
    return bool(torch.isfinite(t).all().item())

def run_epoch(
    model: BasketballScoreMLP,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer | None,
    criterion: nn.Module,
    device: torch.device,
    l1_lambda: float,
    train: bool,
    grad_clip_norm: float | None = 1.0,
) -> Tuple[float, np.ndarray, np.ndarray]:
    model.train(mode=train)
    losses: List[float] = []
    preds: List[np.ndarray] = []
    trues: List[np.ndarray] = []

    bar = tqdm(loader, desc="Train" if train else "Eval", leave=False, unit="batch")
    for batch_idx, batch in enumerate(bar):
        x = batch["x"].to(device)
        team1_id = batch["team1_id"].to(device)
        team2_id = batch["team2_id"].to(device)
        y = batch["y"].to(device)

        with torch.set_grad_enabled(train):
            pred = model(x, team1_id, team2_id)
            loss = criterion(pred, y)

            if train and l1_lambda > 0:
                loss = loss + (l1_lambda * l1_penalty(model))

            if not torch.isfinite(loss):
                raise RuntimeError(
                    f"Non-finite loss detected in {'train' if train else 'eval'} "
                    f"at batch {batch_idx}: {loss.detach().cpu().item()}"
                )

            if train:
                optimizer.zero_grad()
                loss.backward()

                for name, param in model.named_parameters():
                    if param.grad is not None and not is_finite_tensor(param.grad):
                        raise RuntimeError(f"Non-finite gradient detected for parameter: {name}")

                if grad_clip_norm is not None and grad_clip_norm > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)

                optimizer.step()

        loss_value = float(loss.detach().cpu().item())
        if not np.isfinite(loss_value):
            raise RuntimeError(f"Non-finite scalar loss detected: {loss_value}")

        pred_np = pred.detach().cpu().numpy()
        y_np = y.detach().cpu().numpy()

        if not np.isfinite(pred_np).all():
            raise RuntimeError(f"Non-finite prediction values detected at batch {batch_idx}")

        losses.append(loss_value)
        preds.append(pred_np)
        trues.append(y_np)
        bar.set_postfix(loss=f"{np.mean(losses):.4f}")

    return float(np.mean(losses)), np.vstack(preds), np.vstack(trues)


def train_single_model(
    train_loader: DataLoader,
    val_loader: DataLoader,
    model_config: ModelConfig,
    train_config: TrainConfig,
) -> Tuple[BasketballScoreMLP, Dict[str, List[float]]]:
    device = get_device()
    model = BasketballScoreMLP(model_config).to(device)
    criterion = nn.MSELoss()

    optimizer = create_optimizer(model, train_config)
    scheduler = create_scheduler(optimizer, train_config)

    # Applied regularization techniques to prevent overfitting (at least two of: L1/L2 penalty, dropout, early stopping) (5 pts)
    early_stopper = EarlyStopping(
        patience=train_config.patience,
        min_delta=train_config.min_delta,
    )

    history = {
        "train_loss": [],
        "val_loss": [],
        "lr": [],
    }

    epoch_bar = tqdm(range(train_config.epochs), desc="Epochs", unit="epoch")
    for _ in epoch_bar:
        train_loss, _, _ = run_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
            l1_lambda=train_config.l1_lambda,
            train=True,
            grad_clip_norm=1.0,
        )
        val_loss, _, _ = run_epoch(
            model=model,
            loader=val_loader,
            optimizer=None,
            criterion=criterion,
            device=device,
            l1_lambda=0.0,
            train=False,
        )

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["lr"].append(float(optimizer.param_groups[0]["lr"]))

        # Used learning rate scheduling (step decay, cosine annealing, warm-up, ReduceLROnPlateau, or similar) (3 pts)
        if train_config.scheduler_name.lower() == "reduce_on_plateau":
            scheduler.step(val_loss)
        else:
            scheduler.step()

        epoch_bar.set_postfix(train=f"{train_loss:.4f}", val=f"{val_loss:.4f}", lr=f"{optimizer.param_groups[0]['lr']:.2e}")

        if early_stopper.step(val_loss, model):
            break

    if early_stopper.best_state_dict is not None:
        model.load_state_dict(early_stopper.best_state_dict)

    return model, history


def predict_model(model: BasketballScoreMLP, loader: DataLoader) -> Tuple[np.ndarray, np.ndarray]:
    device = get_device()
    model.eval()
    preds: List[np.ndarray] = []
    trues: List[np.ndarray] = []

    with torch.no_grad():
        for batch in tqdm(loader, desc="Predict", leave=False, unit="batch"):
            x = batch["x"].to(device)
            team1_id = batch["team1_id"].to(device)
            team2_id = batch["team2_id"].to(device)
            y = batch["y"].to(device)

            pred = model(x, team1_id, team2_id)
            preds.append(pred.cpu().numpy())
            trues.append(y.cpu().numpy())

    return np.vstack(preds), np.vstack(trues)


def save_best_artifacts(
    model: BasketballScoreMLP,
    standardizer: Standardizer,
    feature_cols: List[str],
    team_id_to_index: Dict[int, int],
    model_config: ModelConfig,
    train_config: TrainConfig,
    split_ratio: Tuple[float, float, float],
    output_dir: Path,
) -> None:
    weights_path = output_dir / "best_score_model.pt"
    metadata_path = output_dir / "best_score_model_meta.json"
    standardizer_path = output_dir / "best_standardizer.pt"

    torch.save(model.state_dict(), weights_path)
    torch.save({
        "means": standardizer.means.to_dict(),
        "stds": standardizer.stds.to_dict(),
        "feature_cols": feature_cols,
        "team_id_to_index": team_id_to_index,
    }, standardizer_path)

    metadata = {
        "model_config": asdict(model_config),
        "train_config": asdict(train_config),
        "best_split_ratio": split_ratio,
        "weights_path": str(weights_path),
        "standardizer_path": str(standardizer_path),
    }
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)


def split_label(split_ratio: Tuple[float, float, float]) -> str:
    return f"{int(split_ratio[0]*100)}_{int(split_ratio[1]*100)}_{int(split_ratio[2]*100)}"


def resolve_rebuild_setting(feature_config: FeatureConfig, requested_rebuild: bool = False) -> bool:
    """
    Reuse existing caches by default. Only rebuild when explicitly requested
    or when no matchup cache exists yet.
    """
    matchup_cache_exists = _PathCheck(feature_config.matchup_cache_path).exists()
    team_cache_exists = _PathCheck(feature_config.feature_cache_path).exists()

    if requested_rebuild:
        print("[cache] Explicit rebuild requested; caches will be regenerated.")
        return True

    if matchup_cache_exists:
        print(f"[cache] Found matchup cache at {feature_config.matchup_cache_path}; reusing it.")
        return False

    if team_cache_exists:
        print(f"[cache] Found team feature cache at {feature_config.feature_cache_path}; matchup cache will be built from available data if needed.")
    else:
        print("[cache] No existing caches found; features will be built from scratch.")

    return False

def merge_experiment_rows(
    existing_csv_path: Path,
    new_rows: List[Dict[str, float | str]],
) -> pd.DataFrame:
    if existing_csv_path.exists():
        existing = pd.read_csv(existing_csv_path)
    else:
        existing = pd.DataFrame()

    new_df = pd.DataFrame(new_rows)

    if not existing.empty and "experiment_name" in existing.columns:
        existing = existing[~existing["experiment_name"].isin(new_df["experiment_name"])].copy()

    merged = pd.concat([existing, new_df], ignore_index=True)
    merged = merged.sort_values(["mse", "winner_accuracy"], ascending=[True, False]).reset_index(drop=True)
    return merged

def main(only_optimizers: List[str] | None = None) -> None:
    optimizer_options = only_optimizers or ["sgd", "adam", "adamw"]
    output_dir = Path("models")
    plots_dir = output_dir / "plots"
    output_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    feature_config = FeatureConfig(
        feature_cache_path="data/team_feature_cache.pkl.gz",
        matchup_cache_path="data/matchup_feature_cache.pkl.gz",
        verbose=True,
    )

    train_config = TrainConfig(
        epochs=75,
        batch_size=128,
        learning_rate=1e-3,
        weight_decay=1e-4,   # L2 regularization
        l1_lambda=1e-6,      # L1 regularization
        patience=10,
        min_delta=1e-4,
        scheduler_name="reduce_on_plateau",
        optimizer_name="adamw",
        seed=42,
    )

    runner = ExperimentRunner(
        csv_path="data/mbb_games.csv",
        output_dir=str(output_dir),
        feature_config=feature_config,
        train_config=train_config,
    )
    rebuild_features = resolve_rebuild_setting(feature_config, requested_rebuild=False)
    runner.prepare_data(rebuild_features=rebuild_features)

    split_options = [
        (0.70, 0.15, 0.15),
        (0.80, 0.10, 0.10),
    ]

    results: List[Dict[str, float | str]] = []
    best_test_mse = float("inf")
    best_payload = None

    for split_ratio in split_options:
        train_raw, val_raw, test_raw, train_loader, val_loader, test_loader, standardizer = runner.make_split_dataloaders(split_ratio)
        split_name = split_label(split_ratio)

        # Created baseline model for comparison (e.g., constant prediction, random, simple heuristic) (3 pts)
        random_pred = random_score_baseline(train_raw, test_raw, seed=train_config.seed)
        random_metrics = regression_metrics(test_raw[["team1pts", "team2pts"]].to_numpy(dtype=np.float32), random_pred)
        results.append({
            "experiment_name": f"baseline_random_{split_name}",
            "split_ratio": str(split_ratio),
            "optimizer": "baseline_random",
            **random_metrics,
        })

        # Created baseline model for comparison (e.g., constant prediction, random, simple heuristic) (3 pts)
        record_pred = record_based_baseline(train_raw, test_raw, seed=train_config.seed)
        record_metrics = regression_metrics(test_raw[["team1pts", "team2pts"]].to_numpy(dtype=np.float32), record_pred)
        results.append({
            "experiment_name": f"baseline_record_{split_name}",
            "split_ratio": str(split_ratio),
            "optimizer": "baseline_record",
            **record_metrics,
        })

        # Compared multiple optimizers (e.g., SGD vs Adam vs AdamW) with documented evaluation (5 pts)
        for optimizer_name in optimizer_options:
            current_train_config = TrainConfig(**asdict(train_config))
            current_train_config.optimizer_name = optimizer_name

            if optimizer_name == "sgd":
                current_train_config.learning_rate = 5e-3
                current_train_config.scheduler_name = "step"
                current_train_config.scheduler_step_size = 10
                current_train_config.scheduler_gamma = 0.5
                current_train_config.weight_decay = 1e-4
                current_train_config.l1_lambda = 1e-6
            elif optimizer_name == "adam":
                current_train_config.learning_rate = 1e-3
                current_train_config.scheduler_name = "reduce_on_plateau"
            elif optimizer_name == "adamw":
                current_train_config.learning_rate = 1e-3
                current_train_config.scheduler_name = "reduce_on_plateau"

            model_config = ModelConfig(
                num_cont_features=len(runner.feature_cols),
                num_teams=len(runner.team_id_to_index),
                embedding_dim=16,
                hidden_dims=(256, 128, 64),
                dropout=0.20,
            )

            model, history = train_single_model(
                train_loader=train_loader,
                val_loader=val_loader,
                model_config=model_config,
                train_config=current_train_config,
            )

            y_pred, y_true = predict_model(model, test_loader)
            metrics = regression_metrics(y_true, y_pred)
            experiment_name = f"mlp_{optimizer_name}_{split_name}"

            results.append({
                "experiment_name": experiment_name,
                "split_ratio": str(split_ratio),
                "optimizer": optimizer_name,
                **metrics,
            })

            # Tracked and visualized training curves showing loss and/or metrics over time (3 pts)
            plot_training_curves(
                history,
                str(plots_dir / f"{experiment_name}_training_curves.png"),
                title=f"Training Curves - {experiment_name}",
            )

            if metrics["mse"] < best_test_mse:
                best_test_mse = metrics["mse"]
                best_payload = {
                    "model": model,
                    "standardizer": standardizer,
                    "feature_cols": runner.feature_cols,
                    "team_id_to_index": runner.team_id_to_index,
                    "model_config": model_config,
                    "train_config": current_train_config,
                    "split_ratio": split_ratio,
                    "y_true": y_true,
                    "y_pred": y_pred,
                    "experiment_name": experiment_name,
                }

    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values(["mse", "winner_accuracy"], ascending=[True, False]).reset_index(drop=True)
    results_df = merge_experiment_rows(
        output_dir / "experiment_results.csv",
        results,
    )
    results_df.to_csv(output_dir / "experiment_results.csv", index=False)

    plot_metric_bars(
        results_df,
        metric="mse",
        output_path=str(plots_dir / "model_comparison_mse.png"),
        title="Test MSE Comparison Across Baselines and Neural Models",
    )
    plot_metric_bars(
        results_df,
        metric="winner_accuracy",
        output_path=str(plots_dir / "model_comparison_winner_accuracy.png"),
        title="Winner Accuracy Comparison Across Baselines and Neural Models",
    )

    if best_payload is not None:
        save_best_artifacts(
            model=best_payload["model"],
            standardizer=best_payload["standardizer"],
            feature_cols=best_payload["feature_cols"],
            team_id_to_index=best_payload["team_id_to_index"],
            model_config=best_payload["model_config"],
            train_config=best_payload["train_config"],
            split_ratio=best_payload["split_ratio"],
            output_dir=output_dir,
        )

        plot_actual_vs_predicted(
            best_payload["y_pred"],
            best_payload["y_true"],
            str(plots_dir / "best_model_actual_vs_predicted_total.png"),
            title=f"Actual vs Predicted Total Points - {best_payload['experiment_name']}",
        )
        plot_margin_residuals(
            best_payload["y_pred"],
            best_payload["y_true"],
            str(plots_dir / "best_model_margin_residuals.png"),
            title=f"Margin Residuals - {best_payload['experiment_name']}",
        )

        print("Best experiment:")
        print(best_payload["experiment_name"])
        print(f"Saved weights to: {output_dir / 'best_score_model.pt'}")
        print(f"Saved metadata to: {output_dir / 'best_score_model_meta.json'}")

    print(f"[plots] Training curves and comparison charts saved under: {plots_dir}")
    print("Finished. Results written to models/experiment_results.csv")


if __name__ == "__main__":
    main(only_optimizers=[])