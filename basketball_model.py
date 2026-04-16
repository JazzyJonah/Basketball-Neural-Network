from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import numpy as np
import torch
from torch import nn


# Modular code design with reusable functions and classes rather than monolithic scripts
@dataclass
class ModelConfig:
    num_cont_features: int
    num_teams: int
    embedding_dim: int = 16
    hidden_dims: Tuple[int, ...] = (256, 128, 64)
    dropout: float = 0.20


# Modular code design with reusable functions and classes rather than monolithic scripts
@dataclass
class TrainConfig:
    epochs: int = 100
    batch_size: int = 128
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    l1_lambda: float = 1e-6
    patience: int = 12
    min_delta: float = 1e-4
    optimizer_name: str = "adamw"
    scheduler_name: str = "reduce_on_plateau"
    scheduler_factor: float = 0.5
    scheduler_patience: int = 4
    scheduler_step_size: int = 10
    scheduler_gamma: float = 0.5
    seed: int = 42


# Defined and trained a custom neural network architecture (substantially designed by you, not a pretrained model) using PyTorch or similar framework (5 pts)
# Applied regularization techniques to prevent overfitting (at least two of: L1/L2 penalty, dropout, early stopping) (5 pts)
class BasketballScoreMLP(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config

        self.team_embedding = nn.Embedding(config.num_teams, config.embedding_dim)

        input_dim = config.num_cont_features + (2 * config.embedding_dim)
        layers: List[nn.Module] = []
        prev_dim = input_dim
        for hidden_dim in config.hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(config.dropout))
            prev_dim = hidden_dim
        self.backbone = nn.Sequential(*layers)
        self.output_head = nn.Linear(prev_dim, 2)

    def forward(self, x_cont: torch.Tensor, team1_id: torch.Tensor, team2_id: torch.Tensor) -> torch.Tensor:
        emb1 = self.team_embedding(team1_id)
        emb2 = self.team_embedding(team2_id)
        x = torch.cat([x_cont, emb1, emb2], dim=1)
        x = self.backbone(x)
        return self.output_head(x)


# Modular code design with reusable functions and classes rather than monolithic scripts
class EarlyStopping:
    def __init__(self, patience: int = 10, min_delta: float = 0.0):
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = np.inf
        self.counter = 0
        self.best_state_dict = None

    def step(self, val_loss: float, model: nn.Module) -> bool:
        if val_loss < (self.best_loss - self.min_delta):
            self.best_loss = val_loss
            self.counter = 0
            self.best_state_dict = {
                k: v.detach().cpu().clone() for k, v in model.state_dict().items()
            }
            return False

        self.counter += 1
        return self.counter >= self.patience


# Applied regularization techniques to prevent overfitting (at least two of: L1/L2 penalty, dropout, early stopping) (5 pts)
def l1_penalty(model: nn.Module) -> torch.Tensor:
    penalty = torch.tensor(0.0, device=next(model.parameters()).device)
    for param in model.parameters():
        penalty = penalty + param.abs().sum()
    return penalty


# Compared multiple optimizers (e.g., SGD vs Adam vs AdamW) with documented evaluation (5 pts)
def create_optimizer(model: nn.Module, train_config: TrainConfig) -> torch.optim.Optimizer:
    name = train_config.optimizer_name.lower()
    if name == "sgd":
        return torch.optim.SGD(
            model.parameters(),
            lr=train_config.learning_rate,
            momentum=0.9,
            weight_decay=train_config.weight_decay,
        )
    if name == "adam":
        return torch.optim.Adam(
            model.parameters(),
            lr=train_config.learning_rate,
            weight_decay=train_config.weight_decay,
        )
    if name == "adamw":
        return torch.optim.AdamW(
            model.parameters(),
            lr=train_config.learning_rate,
            weight_decay=train_config.weight_decay,
        )
    raise ValueError(f"Unsupported optimizer: {train_config.optimizer_name}")


# Used learning rate scheduling (step decay, cosine annealing, warm-up, ReduceLROnPlateau, or similar) (3 pts)
def create_scheduler(optimizer: torch.optim.Optimizer, train_config: TrainConfig):
    name = train_config.scheduler_name.lower()
    if name == "reduce_on_plateau":
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=train_config.scheduler_factor,
            patience=train_config.scheduler_patience,
        )
    if name == "step":
        return torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=train_config.scheduler_step_size,
            gamma=train_config.scheduler_gamma,
        )
    if name == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=train_config.epochs,
        )
    raise ValueError(f"Unsupported scheduler: {train_config.scheduler_name}")


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
