"""
Build leakage-safe training representations for:
- team-season strength/state inputs
- context inputs
- low-rank (factorized) rivalry inputs

Input CSV schema (one row per game):
season,date,team1id,team1home,team1pts,team1fgm,team1fga,team13ptm,team13pta,team1ftm,team1fta,
team1oreb,team1dreb,team1ast,team1stl,team1blk,team1to,team1pf,
team2id,team2home,team2pts,team2fgm,team2fga,team23ptm,team23pta,team2ftm,team2fta,
team2oreb,team2dreb,team2ast,team2stl,team2blk,team2to,team2pf

Output (example parquet/csv):
Each row = one directed training example (A vs B for an actual game),
with features computed ONLY from games before that game in same season.
Also includes team IDs for low-rank rivalry:
- rivalry_off_id = teamA_base_id
- rivalry_def_id = teamB_base_id
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from dataclasses import dataclass
from tqdm import tqdm
from typing import Dict, List, Tuple, Optional

from globals import *


# -----------------------------
# Config
# -----------------------------
RAW_STAT_FIELDS = [
    "pts", "fgm", "fga", "3ptm", "3pta", "ftm", "fta",
    "oreb", "dreb", "ast", "stl", "blk", "to", "pf"
]

# rolling windows for recent form
ROLL_WINDOWS = [5, 10]

# epsilon for safe division
EPS = 1e-9


@dataclass
class BuildConfig:
    csv_path: str
    out_path: Optional[str] = None  # e.g. "training_examples.parquet"
    parse_dates: bool = True
    sort_tiebreaker_cols: Tuple[str, ...] = ("season", "date", "team1id", "team2id")
    keep_both_directions: bool = True  # True -> creates A vs B and B vs A rows


# -----------------------------
# Utilities
# -----------------------------
def safe_div(a: pd.Series | np.ndarray, b: pd.Series | np.ndarray, eps: float = EPS):
    return a / (b + eps)


def possessions_estimate(fga, fta, oreb, to):
    # Common proxy: FGA - OREB + TO + 0.44*FTA
    return fga - oreb + to + 0.44 * fta


def validate_and_load(cfg: BuildConfig) -> pd.DataFrame:
    df = pd.read_csv(cfg.csv_path)
    required_cols = {
        "season", "date",
        "team1id", "team1home",
        "team2id", "team2home",
    }
    for s in RAW_STAT_FIELDS:
        required_cols.add(f"team1{s}")
        required_cols.add(f"team2{s}")
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    if cfg.parse_dates:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        if df["date"].isna().any():
            bad = df[df["date"].isna()].head()
            raise ValueError(f"Could not parse some dates. Sample bad rows:\n{bad}")

    # booleans can come as 0/1 or strings
    df["team1home"] = df["team1home"].astype(bool)
    df["team2home"] = df["team2home"].astype(bool)

    # neutral/home sanity
    both_home = df["team1home"] & df["team2home"]
    if both_home.any():
        n = both_home.sum()
        raise ValueError(f"Found {n} rows where both team1home and team2home are True. Here's the first few:\n{df[both_home].head()}")

    # unique game id for stable ordering
    df = df.reset_index(drop=True)
    df["game_id"] = np.arange(len(df), dtype=np.int64)

    # deterministic sort
    df = df.sort_values(list(cfg.sort_tiebreaker_cols) + ["game_id"]).reset_index(drop=True)
    return df


def expand_to_team_game_rows(df_games: pd.DataFrame) -> pd.DataFrame:
    """
    Convert game-level rows to team-game rows (long format):
    one row per (game, team perspective).
    """
    recs = []

    for _, g in tqdm(df_games.iterrows(), total=len(df_games), desc="Expanding to team-game rows"):
        # Team1 perspective
        rec1 = {
            "game_id": g["game_id"],
            "season": g["season"],
            "date": g["date"],
            "team_id": g["team1id"],
            "opp_id": g["team2id"],
            "is_home": bool(g["team1home"]),
            "is_away": bool(g["team2home"]),
            "is_neutral": (not bool(g["team1home"])) and (not bool(g["team2home"])),
            "pts_for": g["team1pts"],
            "pts_against": g["team2pts"],
        }
        for s in RAW_STAT_FIELDS:
            rec1[f"{s}_for"] = g[f"team1{s}"]
            rec1[f"{s}_against"] = g[f"team2{s}"]

        # Team2 perspective
        rec2 = {
            "game_id": g["game_id"],
            "season": g["season"],
            "date": g["date"],
            "team_id": g["team2id"],
            "opp_id": g["team1id"],
            "is_home": bool(g["team2home"]),
            "is_away": bool(g["team1home"]),
            "is_neutral": (not bool(g["team1home"])) and (not bool(g["team2home"])),
            "pts_for": g["team2pts"],
            "pts_against": g["team1pts"],
        }
        for s in RAW_STAT_FIELDS:
            rec2[f"{s}_for"] = g[f"team2{s}"]
            rec2[f"{s}_against"] = g[f"team1{s}"]

        recs.append(rec1)
        recs.append(rec2)

    tdf = pd.DataFrame.from_records(recs)
    tdf = tdf.sort_values(["season", "date", "game_id", "team_id"]).reset_index(drop=True)

    # Derived per-game efficiencies/features (single-game, not rolling yet)
    tdf["poss_for"] = possessions_estimate(
        tdf["fga_for"], tdf["fta_for"], tdf["oreb_for"], tdf["to_for"]
    )
    tdf["poss_against"] = possessions_estimate(
        tdf["fga_against"], tdf["fta_against"], tdf["oreb_against"], tdf["to_against"]
    )
    tdf["pace_game"] = (tdf["poss_for"] + tdf["poss_against"]) / 2.0

    tdf["off_eff_game"] = 100.0 * safe_div(tdf["pts_for"], tdf["poss_for"])
    tdf["def_eff_game"] = 100.0 * safe_div(tdf["pts_against"], tdf["poss_against"])
    tdf["net_eff_game"] = tdf["off_eff_game"] - tdf["def_eff_game"]

    tdf["efg_for_game"] = safe_div(tdf["fgm_for"] + 0.5 * tdf["3ptm_for"], tdf["fga_for"])
    tdf["efg_against_game"] = safe_div(tdf["fgm_against"] + 0.5 * tdf["3ptm_against"], tdf["fga_against"])
    tdf["tov_rate_for_game"] = safe_div(tdf["to_for"], tdf["poss_for"])
    tdf["tov_rate_against_game"] = safe_div(tdf["to_against"], tdf["poss_against"])
    tdf["orb_rate_for_game"] = safe_div(tdf["oreb_for"], tdf["oreb_for"] + tdf["dreb_against"])
    tdf["orb_rate_against_game"] = safe_div(tdf["oreb_against"], tdf["oreb_against"] + tdf["dreb_for"])
    tdf["ftr_for_game"] = safe_div(tdf["fta_for"], tdf["fga_for"])
    tdf["ftr_against_game"] = safe_div(tdf["fta_against"], tdf["fga_against"])

    tdf["margin_game"] = tdf["pts_for"] - tdf["pts_against"]
    tdf["win_game"] = (tdf["margin_game"] > 0).astype(np.int8)

    return tdf


def add_season_to_date_features(team_games: pd.DataFrame) -> pd.DataFrame:
    """
    Add leakage-safe team-season rolling features:
    everything shifted by 1 game so current game's label is never in its own features.
    """
    tdf = team_games.copy()
    group = tdf.groupby(["season", "team_id"], sort=False)

    # games played before current
    tdf["games_played_before"] = group.cumcount()

    # List of base single-game metrics to aggregate
    metric_cols = [
        "pts_for", "pts_against", "margin_game",
        "poss_for", "poss_against", "pace_game",
        "off_eff_game", "def_eff_game", "net_eff_game",
        "efg_for_game", "efg_against_game",
        "tov_rate_for_game", "tov_rate_against_game",
        "orb_rate_for_game", "orb_rate_against_game",
        "ftr_for_game", "ftr_against_game",
        "ast_for", "stl_for", "blk_for", "pf_for", "to_for",
        "ast_against", "stl_against", "blk_against", "pf_against", "to_against",
        "win_game",
    ]

    # Season-to-date mean (before game)
    for c in metric_cols:
        tdf[f"{c}_s2d_mean"] = group[c].transform(lambda s: s.shift(1).expanding().mean())

    # Season-to-date sum (before game) for a few
    sum_cols = ["pts_for", "pts_against", "poss_for", "poss_against", "win_game"]
    for c in sum_cols:
        tdf[f"{c}_s2d_sum"] = group[c].transform(lambda s: s.shift(1).cumsum())

    # Recent form windows
    for w in tqdm(ROLL_WINDOWS, desc="Adding season-to-date features"):
        for c in metric_cols:
            tdf[f"{c}_last{w}_mean"] = group[c].transform(lambda s: s.shift(1).rolling(w, min_periods=1).mean())

    # rest days
    tdf["days_since_prev_game"] = group["date"].transform(lambda s: s.diff().dt.days).fillna(-1)

    # Early-season missingness fill (before first game)
    fill_zero_cols = [c for c in tdf.columns if c.endswith("_s2d_mean") or c.endswith("_s2d_sum") or "_last" in c]
    tdf[fill_zero_cols] = tdf[fill_zero_cols].fillna(0.0)

    return tdf


def build_id_maps(df_games: pd.DataFrame, team_games: pd.DataFrame) -> Dict[str, Dict]:
    # base team id map (for rivalry factorization u_team, v_team)
    team_ids = pd.Index(sorted(pd.unique(pd.concat([df_games["team1id"], df_games["team2id"]], axis=0))))
    base_team_to_idx = {tid: i for i, tid in enumerate(team_ids)}

    # season-team map (for team-season embeddings)
    team_season_pairs = (
        team_games[["season", "team_id"]]
        .drop_duplicates()
        .sort_values(["season", "team_id"])
        .itertuples(index=False, name=None)
    )
    team_season_to_idx = {pair: i for i, pair in enumerate(team_season_pairs)}

    return {
        "base_team_to_idx": base_team_to_idx,
        "team_season_to_idx": team_season_to_idx,
    }


def join_game_level_examples(
    df_games: pd.DataFrame,
    team_games_feat: pd.DataFrame,
    id_maps: Dict[str, Dict],
    keep_both_directions: bool = True
) -> pd.DataFrame:
    """
    Build final training table:
    one row per game direction A->B (and optional B->A).
    Features for A and B are from their own team-game rows for that same game_id,
    where each feature was built using ONLY prior games in season.
    """
    # extract feature columns
    exclude_cols = {
        "season", "date", "game_id", "team_id", "opp_id",
        "is_home", "is_away", "is_neutral",
        # labels/outcomes from current game (keep separately)
        "pts_for", "pts_against", "margin_game", "win_game",
    }
    feat_cols = [c for c in team_games_feat.columns if c not in exclude_cols]

    # A-side = team1 perspective row
    a = team_games_feat.rename(
        columns={
            "team_id": "teamA_id",
            "opp_id": "teamB_id",
            "is_home": "A_is_home",
            "is_away": "A_is_away",
            "is_neutral": "is_neutral",
            "pts_for": "A_pts",
            "pts_against": "B_pts",
            "margin_game": "A_margin",
            "win_game": "A_win",
        }
    )

    # pick rows where perspective corresponds to team1 in df_games
    # merge key ensures correct perspective
    gk = df_games[["game_id", "season", "date", "team1id", "team2id", "team1home", "team2home"]].copy()
    a = a.merge(
        gk,
        left_on=["game_id", "season", "date", "teamA_id", "teamB_id"],
        right_on=["game_id", "season", "date", "team1id", "team2id"],
        how="inner"
    ).drop(columns=["team1id", "team2id"])

    # B-side features for same game, team2 perspective
    b = team_games_feat.rename(columns={"team_id": "teamB_id", "opp_id": "teamA_id"})
    b_small = b[["game_id", "teamA_id", "teamB_id"] + feat_cols].copy()
    b_small = b_small.rename(columns={c: f"B_{c}" for c in feat_cols})

    a_small = a[["game_id", "season", "date", "teamA_id", "teamB_id", "A_pts", "B_pts", "A_margin", "A_win",
                 "team1home", "team2home"] + feat_cols].copy()
    a_small = a_small.rename(columns={c: f"A_{c}" for c in feat_cols})

    ex = a_small.merge(
        b_small,
        on=["game_id", "teamA_id", "teamB_id"],
        how="left",
        validate="one_to_one"
    )

    # context
    ex["is_neutral"] = (~ex["team1home"].astype(bool)) & (~ex["team2home"].astype(bool))
    ex["A_home"] = ex["team1home"].astype(np.int8)
    ex["B_home"] = ex["team2home"].astype(np.int8)

    # targets
    ex["target_A_score"] = ex["A_pts"]
    ex["target_B_score"] = ex["B_pts"]
    ex["target_margin"] = ex["A_margin"]               # regression target
    ex["target_A_win"] = ex["A_win"].astype(np.int8)   # classification target

    # IDs for embeddings
    base_map = id_maps["base_team_to_idx"]
    ts_map = id_maps["team_season_to_idx"]

    ex["teamA_base_idx"] = ex["teamA_id"].map(base_map).astype(np.int64)
    ex["teamB_base_idx"] = ex["teamB_id"].map(base_map).astype(np.int64)
    ex["teamA_season_idx"] = ex.apply(lambda r: ts_map[(r["season"], r["teamA_id"])], axis=1).astype(np.int64)
    ex["teamB_season_idx"] = ex.apply(lambda r: ts_map[(r["season"], r["teamB_id"])], axis=1).astype(np.int64)

    # low-rank rivalry factorization IDs (u_A^T v_B)
    ex["rivalry_off_id"] = ex["teamA_base_idx"]
    ex["rivalry_def_id"] = ex["teamB_base_idx"]

    # optional reverse direction rows (B vs A)
    if keep_both_directions:
        rev = ex.copy()
        # swap team roles
        swap_cols = [
            ("teamA_id", "teamB_id"),
            ("teamA_base_idx", "teamB_base_idx"),
            ("teamA_season_idx", "teamB_season_idx"),
            ("rivalry_off_id", "rivalry_def_id"),
            ("A_home", "B_home"),
            ("target_A_score", "target_B_score"),
        ]
        for c1, c2 in swap_cols:
            rev[c1], rev[c2] = ex[c2].values, ex[c1].values

        # margin/win invert
        rev["target_margin"] = -ex["target_margin"].values
        rev["target_A_win"] = (rev["target_margin"] > 0).astype(np.int8)

        # swap A_* and B_* feature prefixes
        a_cols = [c for c in ex.columns if c.startswith("A_") and c not in {"A_home"}]
        b_cols = [c for c in ex.columns if c.startswith("B_") and c not in {"B_home"}]
        # build mapping by suffix
        a_suffix_to_col = {c[2:]: c for c in a_cols}
        b_suffix_to_col = {c[2:]: c for c in b_cols}
        common_suffixes = sorted(set(a_suffix_to_col).intersection(b_suffix_to_col))
        for suf in common_suffixes:
            ac = a_suffix_to_col[suf]
            bc = b_suffix_to_col[suf]
            rev[ac], rev[bc] = ex[bc].values, ex[ac].values

        ex = pd.concat([ex, rev], axis=0, ignore_index=True)

    # drop raw per-game outcomes from features table
    ex = ex.drop(columns=["A_pts", "B_pts", "A_margin", "A_win", "team1home", "team2home"])

    # deterministic order
    ex = ex.sort_values(["season", "date", "game_id", "teamA_id", "teamB_id"]).reset_index(drop=True)
    return ex


def build_representations(cfg: BuildConfig) -> Tuple[pd.DataFrame, Dict[str, Dict]]:
    # 1) Load + validate game rows
    for i in tqdm(range(1), desc="Loading and validating data"):
        games = validate_and_load(cfg)

    # 2) Long format team-game rows
    team_games = expand_to_team_game_rows(games)

    # 3) Leakage-safe season-to-date features
    team_games_feat = add_season_to_date_features(team_games)

    # 4) ID maps for embeddings/factorization
    for i in tqdm(range(1), desc="Building ID maps"):
        id_maps = build_id_maps(games, team_games_feat)

    # 5) Final game-level supervised examples
    for i in tqdm(range(1), desc="Building final game-level examples"):
        examples = join_game_level_examples(
            games,
            team_games_feat,
            id_maps,
            keep_both_directions=cfg.keep_both_directions
        )

    # 6) Save optionally
    if cfg.out_path:
        if cfg.out_path.endswith(".parquet"):
            examples.to_parquet(cfg.out_path, index=False)
        else:
            examples.to_csv(cfg.out_path, index=False)

    return examples, id_maps


if __name__ == "__main__":
    # Example usage:
    cfg = BuildConfig(
        csv_path=historicalDataFile,
        out_path=historicalTableFile,
        keep_both_directions=True
    )
    examples_df, id_maps = build_representations(cfg)

    print("Built examples:", len(examples_df))
    print("Columns:", len(examples_df.columns))
    print("Num base teams:", len(id_maps["base_team_to_idx"]))
    print("Num team-season entities:", len(id_maps["team_season_to_idx"]))

    # Minimal set the model will definitely need:
    core_cols = [
        "season", "date", "game_id",
        "teamA_id", "teamB_id",
        "teamA_base_idx", "teamB_base_idx",
        "teamA_season_idx", "teamB_season_idx",
        "rivalry_off_id", "rivalry_def_id",
        "A_home", "B_home", "is_neutral",
        "target_A_score", "target_B_score", "target_margin", "target_A_win",
    ]
    print("Core preview:")
    print(examples_df[core_cols].head(10))
# -----------------------------