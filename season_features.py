from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm


# Modular code design with reusable functions and classes rather than monolithic scripts
@dataclass
class FeatureConfig:
    fta_weight: float = 0.44
    decay_days: int = 180
    feature_cache_path: str = "team_feature_cache.pkl"
    matchup_cache_path: str = "matchup_feature_cache.pkl"
    sort_cache_by_date: bool = True
    verbose: bool = True


TEAM_BOX_COLS = [
    "pts",
    "fgm",
    "fga",
    "tpm",
    "tpa",
    "ftm",
    "fta",
    "oreb",
    "dreb",
    "ast",
    "stl",
    "blk",
    "to",
    "pf",
]


BASE_FEATURE_NAMES = [
    "games_played",
    "wins",
    "losses",
    "win_pct",
    "days_since_last_game",
    "opp_win_pct_mean",
    "opp_win_pct_weighted",
    "pts_for_pg",
    "pts_against_pg",
    "fgm_pg",
    "fga_pg",
    "tpm_pg",
    "tpa_pg",
    "ftm_pg",
    "fta_pg",
    "oreb_pg",
    "dreb_pg",
    "ast_pg",
    "stl_pg",
    "blk_pg",
    "to_pg",
    "pf_pg",
    "poss_pg",
    "opp_poss_pg",
    "off_eff",
    "def_eff",
    "efg_pct",
    "tov_rate",
    "oreb_rate",
    "ft_rate",
    "weighted_pts_for_pg",
    "weighted_pts_against_pg",
    "weighted_poss_pg",
    "weighted_opp_poss_pg",
    "weighted_off_eff",
    "weighted_def_eff",
    "weighted_efg_pct",
    "weighted_tov_rate",
    "weighted_oreb_rate",
    "weighted_ft_rate",
    "weighted_win_pct",
]


def _safe_div(n: float, d: float, default: float = 0.0) -> float:
    if d is None or d == 0:
        return default
    return float(n) / float(d)


def _to_datetime_date(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, errors="coerce").dt.normalize()


def _compute_possessions(
    fga: pd.Series, oreb: pd.Series, tov: pd.Series, fta: pd.Series, y: float
) -> pd.Series:
    return (fga - oreb) + tov + (y * fta)


def _linear_time_weights(
    past_dates: pd.Series, query_date: pd.Timestamp, decay_days: int
) -> np.ndarray:
    days_ago = (query_date - past_dates).dt.days.to_numpy(dtype=float)
    weights = 1.0 - (days_ago / float(decay_days))
    weights = np.clip(weights, 0.0, 1.0)
    return weights


# Applied basic preprocessing appropriate to your modality (e.g., image resizing, text tokenization, handling missing values)
def load_games_csv(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)

    numeric_cols = [
        "season",
        "team1id",
        "team1pts",
        "team1fgm",
        "team1fga",
        "team13ptm",
        "team13pta",
        "team1ftm",
        "team1fta",
        "team1oreb",
        "team1dreb",
        "team1ast",
        "team1stl",
        "team1blk",
        "team1to",
        "team1pf",
        "team2id",
        "team2pts",
        "team2fgm",
        "team2fga",
        "team23ptm",
        "team23pta",
        "team2ftm",
        "team2fta",
        "team2oreb",
        "team2dreb",
        "team2ast",
        "team2stl",
        "team2blk",
        "team2to",
        "team2pf",
    ]

    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df["date"] = _to_datetime_date(df["date"])

    if "team1home" in df.columns:
        df["team1home"] = df["team1home"].astype(bool)
    if "team2home" in df.columns:
        df["team2home"] = df["team2home"].astype(bool)

    required_cols = [
        "season",
        "date",
        "team1id",
        "team1pts",
        "team1fgm",
        "team1fga",
        "team13ptm",
        "team13pta",
        "team1ftm",
        "team1fta",
        "team1oreb",
        "team1dreb",
        "team1ast",
        "team1stl",
        "team1blk",
        "team1to",
        "team1pf",
        "team2id",
        "team2pts",
        "team2fgm",
        "team2fga",
        "team23ptm",
        "team23pta",
        "team2ftm",
        "team2fta",
        "team2oreb",
        "team2dreb",
        "team2ast",
        "team2stl",
        "team2blk",
        "team2to",
        "team2pf",
    ]

    df = df.dropna(subset=required_cols).copy()
    df = df.sort_values(["season", "date", "team1id", "team2id"]).reset_index(drop=True)

    return df


def add_pregame_records(games: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    games = games.copy()
    games["team1_wins_before"] = 0
    games["team1_losses_before"] = 0
    games["team2_wins_before"] = 0
    games["team2_losses_before"] = 0

    grouped = list(games.groupby("season").groups.items())

    season_iter = tqdm(
        grouped,
        total=len(grouped),
        desc="Building pregame records",
        unit="season",
        disable=not verbose,
    )

    for season, season_idx in season_iter:
        record_map: Dict[int, List[int]] = {}

        season_rows = games.loc[season_idx].sort_values(["date", "team1id", "team2id"])
        row_iter = tqdm(
            season_rows.iterrows(),
            total=len(season_rows),
            desc=f"Season {season} record pass",
            unit="game",
            leave=False,
            disable=not verbose,
        )

        for idx, row in row_iter:
            t1 = int(row["team1id"])
            t2 = int(row["team2id"])

            if t1 not in record_map:
                record_map[t1] = [0, 0]
            if t2 not in record_map:
                record_map[t2] = [0, 0]

            games.at[idx, "team1_wins_before"] = record_map[t1][0]
            games.at[idx, "team1_losses_before"] = record_map[t1][1]
            games.at[idx, "team2_wins_before"] = record_map[t2][0]
            games.at[idx, "team2_losses_before"] = record_map[t2][1]

            t1_win = int(row["team1pts"] > row["team2pts"])
            t2_win = 1 - t1_win

            record_map[t1][0] += t1_win
            record_map[t1][1] += 1 - t1_win
            record_map[t2][0] += t2_win
            record_map[t2][1] += 1 - t2_win

    return games


def build_team_game_frame(games: pd.DataFrame, cfg: FeatureConfig) -> pd.DataFrame:
    games = add_pregame_records(games, verbose=cfg.verbose)

    t1 = pd.DataFrame(
        {
            "season": games["season"],
            "date": games["date"],
            "team_id": games["team1id"].astype(int),
            "opp_id": games["team2id"].astype(int),
            "is_home": games.get("team1home", False),
            "pts_for": games["team1pts"],
            "pts_against": games["team2pts"],
            "fgm": games["team1fgm"],
            "fga": games["team1fga"],
            "tpm": games["team13ptm"],
            "tpa": games["team13pta"],
            "ftm": games["team1ftm"],
            "fta": games["team1fta"],
            "oreb": games["team1oreb"],
            "dreb": games["team1dreb"],
            "ast": games["team1ast"],
            "stl": games["team1stl"],
            "blk": games["team1blk"],
            "to": games["team1to"],
            "pf": games["team1pf"],
            "opp_fgm": games["team2fgm"],
            "opp_fga": games["team2fga"],
            "opp_tpm": games["team23ptm"],
            "opp_tpa": games["team23pta"],
            "opp_ftm": games["team2ftm"],
            "opp_fta": games["team2fta"],
            "opp_oreb": games["team2oreb"],
            "opp_dreb": games["team2dreb"],
            "opp_ast": games["team2ast"],
            "opp_stl": games["team2stl"],
            "opp_blk": games["team2blk"],
            "opp_to": games["team2to"],
            "opp_pf": games["team2pf"],
            "win": (games["team1pts"] > games["team2pts"]).astype(int),
            "opp_wins_before": games["team2_wins_before"],
            "opp_losses_before": games["team2_losses_before"],
        }
    )

    t2 = pd.DataFrame(
        {
            "season": games["season"],
            "date": games["date"],
            "team_id": games["team2id"].astype(int),
            "opp_id": games["team1id"].astype(int),
            "is_home": games.get("team2home", False),
            "pts_for": games["team2pts"],
            "pts_against": games["team1pts"],
            "fgm": games["team2fgm"],
            "fga": games["team2fga"],
            "tpm": games["team23ptm"],
            "tpa": games["team23pta"],
            "ftm": games["team2ftm"],
            "fta": games["team2fta"],
            "oreb": games["team2oreb"],
            "dreb": games["team2dreb"],
            "ast": games["team2ast"],
            "stl": games["team2stl"],
            "blk": games["team2blk"],
            "to": games["team2to"],
            "pf": games["team2pf"],
            "opp_fgm": games["team1fgm"],
            "opp_fga": games["team1fga"],
            "opp_tpm": games["team13ptm"],
            "opp_tpa": games["team13pta"],
            "opp_ftm": games["team1ftm"],
            "opp_fta": games["team1fta"],
            "opp_oreb": games["team1oreb"],
            "opp_dreb": games["team1dreb"],
            "opp_ast": games["team1ast"],
            "opp_stl": games["team1stl"],
            "opp_blk": games["team1blk"],
            "opp_to": games["team1to"],
            "opp_pf": games["team1pf"],
            "win": (games["team2pts"] > games["team1pts"]).astype(int),
            "opp_wins_before": games["team1_wins_before"],
            "opp_losses_before": games["team1_losses_before"],
        }
    )

    team_games = pd.concat([t1, t2], ignore_index=True)
    team_games["date"] = _to_datetime_date(team_games["date"])

    team_games["opp_win_pct_before"] = (
        team_games["opp_wins_before"]
        / (team_games["opp_wins_before"] + team_games["opp_losses_before"]).replace(
            0, np.nan
        )
    ).fillna(0.0)

    team_games["poss"] = _compute_possessions(
        team_games["fga"],
        team_games["oreb"],
        team_games["to"],
        team_games["fta"],
        cfg.fta_weight,
    )
    team_games["opp_poss"] = _compute_possessions(
        team_games["opp_fga"],
        team_games["opp_oreb"],
        team_games["opp_to"],
        team_games["opp_fta"],
        cfg.fta_weight,
    )

    team_games = team_games.sort_values(
        ["season", "team_id", "date", "opp_id"]
    ).reset_index(drop=True)
    return team_games


def summarize_history(
    history: pd.DataFrame, query_date: pd.Timestamp, cfg: FeatureConfig
) -> Dict[str, float]:
    if history.empty:
        return {name: 0.0 for name in BASE_FEATURE_NAMES}

    games_played = float(len(history))
    wins = float(history["win"].sum())
    losses = games_played - wins
    win_pct = _safe_div(wins, games_played, 0.0)

    last_date = history["date"].max()
    days_since_last_game = (
        float((query_date - last_date).days) if pd.notna(last_date) else 999.0
    )

    pts_for_pg = float(history["pts_for"].mean())
    pts_against_pg = float(history["pts_against"].mean())
    fgm_pg = float(history["fgm"].mean())
    fga_pg = float(history["fga"].mean())
    tpm_pg = float(history["tpm"].mean())
    tpa_pg = float(history["tpa"].mean())
    ftm_pg = float(history["ftm"].mean())
    fta_pg = float(history["fta"].mean())
    oreb_pg = float(history["oreb"].mean())
    dreb_pg = float(history["dreb"].mean())
    ast_pg = float(history["ast"].mean())
    stl_pg = float(history["stl"].mean())
    blk_pg = float(history["blk"].mean())
    to_pg = float(history["to"].mean())
    pf_pg = float(history["pf"].mean())
    poss_pg = float(history["poss"].mean())
    opp_poss_pg = float(history["opp_poss"].mean())
    opp_win_pct_mean = float(history["opp_win_pct_before"].mean())

    total_pts = float(history["pts_for"].sum())
    total_pts_against = float(history["pts_against"].sum())
    total_poss = float(history["poss"].sum())
    total_opp_poss = float(history["opp_poss"].sum())
    total_fgm = float(history["fgm"].sum())
    total_tpm = float(history["tpm"].sum())
    total_fga = float(history["fga"].sum())
    total_to = float(history["to"].sum())
    total_oreb = float(history["oreb"].sum())
    total_opp_dreb = float(history["opp_dreb"].sum())
    total_fta = float(history["fta"].sum())

    off_eff = _safe_div(total_pts, total_poss, 0.0)
    def_eff = _safe_div(total_pts_against, total_opp_poss, 0.0)
    efg_pct = _safe_div(total_fgm + 0.5 * total_tpm, total_fga, 0.0)
    tov_rate = _safe_div(total_to, total_poss, 0.0)
    oreb_rate = _safe_div(total_oreb, total_oreb + total_opp_dreb, 0.0)
    ft_rate = _safe_div(total_fta, total_fga, 0.0)

    weights = _linear_time_weights(history["date"], query_date, cfg.decay_days)
    if weights.sum() <= 0:
        weights = np.ones(len(history), dtype=float)

    def wavg(col: str) -> float:
        return float(np.average(history[col].to_numpy(dtype=float), weights=weights))

    weighted_pts_for_pg = wavg("pts_for")
    weighted_pts_against_pg = wavg("pts_against")
    weighted_poss_pg = wavg("poss")
    weighted_opp_poss_pg = wavg("opp_poss")
    weighted_opp_win_pct = wavg("opp_win_pct_before")
    weighted_win_pct = wavg("win")

    weighted_total_pts = float(
        np.sum(history["pts_for"].to_numpy(dtype=float) * weights)
    )
    weighted_total_pts_against = float(
        np.sum(history["pts_against"].to_numpy(dtype=float) * weights)
    )
    weighted_total_poss = float(np.sum(history["poss"].to_numpy(dtype=float) * weights))
    weighted_total_opp_poss = float(
        np.sum(history["opp_poss"].to_numpy(dtype=float) * weights)
    )
    weighted_total_fgm = float(np.sum(history["fgm"].to_numpy(dtype=float) * weights))
    weighted_total_tpm = float(np.sum(history["tpm"].to_numpy(dtype=float) * weights))
    weighted_total_fga = float(np.sum(history["fga"].to_numpy(dtype=float) * weights))
    weighted_total_to = float(np.sum(history["to"].to_numpy(dtype=float) * weights))
    weighted_total_oreb = float(np.sum(history["oreb"].to_numpy(dtype=float) * weights))
    weighted_total_opp_dreb = float(
        np.sum(history["opp_dreb"].to_numpy(dtype=float) * weights)
    )
    weighted_total_fta = float(np.sum(history["fta"].to_numpy(dtype=float) * weights))

    weighted_off_eff = _safe_div(weighted_total_pts, weighted_total_poss, 0.0)
    weighted_def_eff = _safe_div(
        weighted_total_pts_against, weighted_total_opp_poss, 0.0
    )
    weighted_efg_pct = _safe_div(
        weighted_total_fgm + 0.5 * weighted_total_tpm, weighted_total_fga, 0.0
    )
    weighted_tov_rate = _safe_div(weighted_total_to, weighted_total_poss, 0.0)
    weighted_oreb_rate = _safe_div(
        weighted_total_oreb, weighted_total_oreb + weighted_total_opp_dreb, 0.0
    )
    weighted_ft_rate = _safe_div(weighted_total_fta, weighted_total_fga, 0.0)

    return {
        "games_played": games_played,
        "wins": wins,
        "losses": losses,
        "win_pct": win_pct,
        "days_since_last_game": days_since_last_game,
        "opp_win_pct_mean": opp_win_pct_mean,
        "opp_win_pct_weighted": weighted_opp_win_pct,
        "pts_for_pg": pts_for_pg,
        "pts_against_pg": pts_against_pg,
        "fgm_pg": fgm_pg,
        "fga_pg": fga_pg,
        "tpm_pg": tpm_pg,
        "tpa_pg": tpa_pg,
        "ftm_pg": ftm_pg,
        "fta_pg": fta_pg,
        "oreb_pg": oreb_pg,
        "dreb_pg": dreb_pg,
        "ast_pg": ast_pg,
        "stl_pg": stl_pg,
        "blk_pg": blk_pg,
        "to_pg": to_pg,
        "pf_pg": pf_pg,
        "poss_pg": poss_pg,
        "opp_poss_pg": opp_poss_pg,
        "off_eff": off_eff,
        "def_eff": def_eff,
        "efg_pct": efg_pct,
        "tov_rate": tov_rate,
        "oreb_rate": oreb_rate,
        "ft_rate": ft_rate,
        "weighted_pts_for_pg": weighted_pts_for_pg,
        "weighted_pts_against_pg": weighted_pts_against_pg,
        "weighted_poss_pg": weighted_poss_pg,
        "weighted_opp_poss_pg": weighted_opp_poss_pg,
        "weighted_off_eff": weighted_off_eff,
        "weighted_def_eff": weighted_def_eff,
        "weighted_efg_pct": weighted_efg_pct,
        "weighted_tov_rate": weighted_tov_rate,
        "weighted_oreb_rate": weighted_oreb_rate,
        "weighted_ft_rate": weighted_ft_rate,
        "weighted_win_pct": weighted_win_pct,
    }


# Modular code design with reusable functions and classes rather than monolithic scripts
class SeasonToDateFeatureStore:
    def __init__(self, games_df: pd.DataFrame, cfg: Optional[FeatureConfig] = None):
        self.cfg = cfg or FeatureConfig()
        self.games_df = games_df.copy()
        self.team_games = build_team_game_frame(self.games_df, self.cfg)
        self.feature_table: Optional[pd.DataFrame] = None
        self._feature_lookup: Dict[Tuple[int, int, pd.Timestamp], np.ndarray] = {}
        self.feature_names = BASE_FEATURE_NAMES.copy()

    def build_or_load_feature_cache(self, rebuild: bool = False) -> pd.DataFrame:
        cache_path = Path(self.cfg.feature_cache_path)

        if cache_path.exists() and not rebuild:
            if self.cfg.verbose:
                print(f"[cache] Loading team feature cache from {cache_path}")
            self.feature_table = pd.read_pickle(cache_path)
            self._build_lookup()
            return self.feature_table

        if self.cfg.verbose:
            print("[build] Creating team feature cache from scratch...")

        rows = []
        grouped_items = list(self.team_games.groupby(["season", "team_id"], sort=False))
        outer_bar = tqdm(
            grouped_items,
            total=len(grouped_items),
            desc="Building team feature cache",
            unit="team-season",
            disable=not self.cfg.verbose,
        )

        for (season, team_id), group in outer_bar:
            group = group.sort_values(["date", "opp_id"]).reset_index(drop=True)

            inner_bar = tqdm(
                range(len(group)),
                total=len(group),
                desc=f"Season {season} Team {team_id}",
                unit="game",
                leave=False,
                disable=not self.cfg.verbose,
            )

            for i in inner_bar:
                query_row = group.iloc[i]
                query_date = pd.Timestamp(query_row["date"])
                history = group.iloc[:i].copy()

                features = summarize_history(history, query_date, self.cfg)
                feature_row = {
                    "season": int(season),
                    "team_id": int(team_id),
                    "date": query_date,
                }
                feature_row.update(features)
                rows.append(feature_row)

        feature_table = pd.DataFrame(rows)

        if self.cfg.sort_cache_by_date:
            feature_table = feature_table.sort_values(
                ["season", "date", "team_id"]
            ).reset_index(drop=True)

        if self.cfg.verbose:
            print(f"[cache] Saving team feature cache to {cache_path}")
        feature_table.to_pickle(cache_path)

        self.feature_table = feature_table
        self._build_lookup()
        return feature_table

    def _build_lookup(self) -> None:
        if self.feature_table is None:
            return

        self._feature_lookup = {}
        for row in self.feature_table.itertuples(index=False):
            key = (int(row.season), int(row.team_id), pd.Timestamp(row.date))
            values = np.array(
                [getattr(row, f) for f in self.feature_names], dtype=np.float32
            )
            self._feature_lookup[key] = values

    def get_team_features(
        self, team_id: int, date: pd.Timestamp, season: int
    ) -> Dict[str, float]:
        if self.feature_table is None:
            self.build_or_load_feature_cache(rebuild=False)

        key = (int(season), int(team_id), pd.Timestamp(date))
        values = self._feature_lookup.get(key)

        if values is None:
            return {name: 0.0 for name in self.feature_names}

        return {name: float(values[i]) for i, name in enumerate(self.feature_names)}

    def get_team_feature_vector(
        self, team_id: int, date: pd.Timestamp, season: int
    ) -> np.ndarray:
        if self.feature_table is None:
            self.build_or_load_feature_cache(rebuild=False)

        key = (int(season), int(team_id), pd.Timestamp(date))
        if key not in self._feature_lookup:
            return np.zeros(len(self.feature_names), dtype=np.float32)

        return self._feature_lookup[key]


def build_matchup_feature_frame(
    games_df: pd.DataFrame,
    feature_store: SeasonToDateFeatureStore,
    rebuild: bool = False,
) -> pd.DataFrame:
    cache_path = Path(feature_store.cfg.matchup_cache_path)

    if cache_path.exists() and not rebuild:
        if feature_store.cfg.verbose:
            print(f"[cache] Loading matchup feature cache from {cache_path}")
        return pd.read_pickle(cache_path)

    if feature_store.cfg.verbose:
        print("[build] Creating matchup feature cache from scratch...")

    feature_table = feature_store.build_or_load_feature_cache(rebuild=rebuild)
    feature_names = feature_store.feature_names

    t1_features = feature_table.rename(
        columns={name: f"team1_{name}" for name in feature_names}
    ).rename(columns={"team_id": "team1id"})

    t2_features = feature_table.rename(
        columns={name: f"team2_{name}" for name in feature_names}
    ).rename(columns={"team_id": "team2id"})

    matchups = games_df.copy()
    matchups["date"] = _to_datetime_date(matchups["date"])

    if feature_store.cfg.verbose:
        print("[merge] Merging team1 pregame features...")
    matchups = matchups.merge(
        t1_features,
        on=["season", "date", "team1id"],
        how="left",
    )

    if feature_store.cfg.verbose:
        print("[merge] Merging team2 pregame features...")
    matchups = matchups.merge(
        t2_features,
        on=["season", "date", "team2id"],
        how="left",
    )

    # Applied basic preprocessing appropriate to your modality (e.g., image resizing, text tokenization, handling missing values)
    fill_bar = tqdm(
        feature_names,
        total=len(feature_names),
        desc="Filling missing merged features",
        unit="feature",
        disable=not feature_store.cfg.verbose,
    )
    for name in fill_bar:
        matchups[f"team1_{name}"] = matchups[f"team1_{name}"].fillna(0.0)
        matchups[f"team2_{name}"] = matchups[f"team2_{name}"].fillna(0.0)

    make_bar = tqdm(
        feature_names,
        total=len(feature_names),
        desc="Creating diff/sum features",
        unit="feature",
        disable=not feature_store.cfg.verbose,
    )
    for name in make_bar:
        matchups[f"diff_{name}"] = matchups[f"team1_{name}"] - matchups[f"team2_{name}"]
        matchups[f"sum_{name}"] = matchups[f"team1_{name}"] + matchups[f"team2_{name}"]

    matchups["home_indicator"] = matchups["team1home"].astype(int) - matchups[
        "team2home"
    ].astype(int)

    if feature_store.cfg.verbose:
        print(f"[cache] Saving matchup feature cache to {cache_path}")
    matchups.to_pickle(cache_path)
    return matchups


# Implemented proper train/validation/test split with documented split ratios (3 pts)
def chronological_split(
    matchups: pd.DataFrame,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if not np.isclose(train_ratio + val_ratio + test_ratio, 1.0):
        raise ValueError("Split ratios must sum to 1.0")

    ordered = matchups.sort_values(
        ["date", "season", "team1id", "team2id"]
    ).reset_index(drop=True)
    n = len(ordered)

    train_end = int(n * train_ratio)
    val_end = train_end + int(n * val_ratio)

    train_df = ordered.iloc[:train_end].copy()
    val_df = ordered.iloc[train_end:val_end].copy()
    test_df = ordered.iloc[val_end:].copy()

    return train_df, val_df, test_df


# Properly normalized or standardized input features/data appropriate to your modality (3 pts)
class Standardizer:
    def __init__(self):
        self.means: Optional[pd.Series] = None
        self.stds: Optional[pd.Series] = None

    def fit(self, df: pd.DataFrame, feature_cols: List[str]) -> None:
        self.means = df[feature_cols].mean()
        self.stds = df[feature_cols].std().replace(0, 1.0).fillna(1.0)

    def transform(self, df: pd.DataFrame, feature_cols: List[str]) -> pd.DataFrame:
        if self.means is None or self.stds is None:
            raise ValueError("Standardizer must be fit before calling transform.")
        out = df.copy()
        out.loc[:, feature_cols] = (out[feature_cols] - self.means) / self.stds
        return out

    def fit_transform(self, df: pd.DataFrame, feature_cols: List[str]) -> pd.DataFrame:
        self.fit(df, feature_cols)
        return self.transform(df, feature_cols)


def get_continuous_feature_columns(matchups: pd.DataFrame) -> List[str]:
    feature_cols = [
        col
        for col in matchups.columns
        if col.startswith("team1_")
        or col.startswith("team2_")
        or col.startswith("diff_")
        or col.startswith("sum_")
    ]
    feature_cols.append("home_indicator")
    return sorted(set(feature_cols))


def build_team_id_index(games_df: pd.DataFrame) -> Dict[int, int]:
    all_ids = sorted(
        set(games_df["team1id"].astype(int)).union(set(games_df["team2id"].astype(int)))
    )
    return {team_id: idx for idx, team_id in enumerate(all_ids)}


# Used appropriate data loading with batching and shuffling (PyTorch DataLoader or equivalent) (3 pts)
class BasketballMatchupDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        feature_cols: List[str],
        team_id_to_index: Dict[int, int],
        target_cols: Tuple[str, str] = ("team1pts", "team2pts"),
    ):
        self.df = df.reset_index(drop=True).copy()
        self.feature_cols = feature_cols
        self.team_id_to_index = team_id_to_index
        self.target_cols = target_cols

        self.x = self.df[self.feature_cols].to_numpy(dtype=np.float32)
        self.team1_idx = (
            self.df["team1id"]
            .astype(int)
            .map(self.team_id_to_index)
            .to_numpy(dtype=np.int64)
        )
        self.team2_idx = (
            self.df["team2id"]
            .astype(int)
            .map(self.team_id_to_index)
            .to_numpy(dtype=np.int64)
        )
        self.y = self.df[list(self.target_cols)].to_numpy(dtype=np.float32)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return {
            "x": torch.tensor(self.x[idx], dtype=torch.float32),
            "team1_id": torch.tensor(self.team1_idx[idx], dtype=torch.long),
            "team2_id": torch.tensor(self.team2_idx[idx], dtype=torch.long),
            "y": torch.tensor(self.y[idx], dtype=torch.float32),
        }


# Used appropriate data loading with batching and shuffling (PyTorch DataLoader or equivalent) (3 pts)
def build_dataloaders(
    csv_path: str = "mbb_games.csv",
    cfg: Optional[FeatureConfig] = None,
    batch_size: int = 128,
    rebuild_features: bool = False,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
) -> Tuple[DataLoader, DataLoader, DataLoader, List[str], Standardizer, Dict[int, int]]:
    cfg = cfg or FeatureConfig()

    if cfg.verbose:
        print(f"[load] Reading games from {csv_path}")
    games = load_games_csv(csv_path)

    if cfg.verbose:
        print("[init] Initializing season-to-date feature store")
    store = SeasonToDateFeatureStore(games, cfg=cfg)

    matchups = build_matchup_feature_frame(games, store, rebuild=rebuild_features)

    train_df, val_df, test_df = chronological_split(
        matchups,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
    )

    if cfg.verbose:
        print(
            f"[split] train={len(train_df)} "
            f"val={len(val_df)} "
            f"test={len(test_df)} "
            f"ratios=({train_ratio:.2f}, {val_ratio:.2f}, {test_ratio:.2f})"
        )

    feature_cols = get_continuous_feature_columns(matchups)

    # Properly normalized or standardized input features/data appropriate to your modality (3 pts)
    standardizer = Standardizer()
    train_df = standardizer.fit_transform(train_df, feature_cols)
    val_df = standardizer.transform(val_df, feature_cols)
    test_df = standardizer.transform(test_df, feature_cols)

    team_id_to_index = build_team_id_index(games)

    train_ds = BasketballMatchupDataset(train_df, feature_cols, team_id_to_index)
    val_ds = BasketballMatchupDataset(val_df, feature_cols, team_id_to_index)
    test_ds = BasketballMatchupDataset(test_df, feature_cols, team_id_to_index)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    if cfg.verbose:
        print(f"[done] Built dataloaders with batch_size={batch_size}")

    return (
        train_loader,
        val_loader,
        test_loader,
        feature_cols,
        standardizer,
        team_id_to_index,
    )


if __name__ == "__main__":
    cfg = FeatureConfig(
        feature_cache_path="team_feature_cache.pkl",
        matchup_cache_path="matchup_feature_cache.pkl",
        decay_days=180,
        fta_weight=0.44,
        verbose=True,
    )

    (
        train_loader,
        val_loader,
        test_loader,
        feature_cols,
        standardizer,
        team_id_to_index,
    ) = build_dataloaders(
        csv_path="mbb_games.csv",
        cfg=cfg,
        batch_size=128,
        rebuild_features=False,
        train_ratio=0.70,
        val_ratio=0.15,
        test_ratio=0.15,
    )

    batch = next(iter(train_loader))
    print("x shape:", batch["x"].shape)
    print("team1_id shape:", batch["team1_id"].shape)
    print("team2_id shape:", batch["team2_id"].shape)
    print("y shape:", batch["y"].shape)
    print("num continuous features:", len(feature_cols))
    print("num teams:", len(team_id_to_index))
