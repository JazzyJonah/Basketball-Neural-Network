from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

import numpy as np
import pandas as pd
import sportsdataverse.mbb as mbb
from tqdm.auto import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

try:
    from src.season_features import FeatureConfig, SeasonToDateFeatureStore, load_games_csv
except ImportError:
    from season_features import FeatureConfig, SeasonToDateFeatureStore, load_games_csv


DEFAULT_SEASON = 2027
DEFAULT_START = date(2026, 11, 1)
DEFAULT_END = date(2027, 4, 30)


def _date_range(start: date, end: date) -> Iterable[date]:
    current = start
    while current <= end:
        yield current
        current += timedelta(days=1)


def _first_existing(row: pd.Series, *columns: str, default: Any = None) -> Any:
    for column in columns:
        if column in row.index:
            value = row[column]
            if pd.notna(value):
                return value
    return default


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return False
    if isinstance(value, (int, np.integer, float, np.floating)):
        return bool(value)
    return str(value).strip().lower() in {"1", "true", "yes", "y", "final", "completed"}


def _is_completed_schedule_row(row: pd.Series) -> bool:
    direct = _first_existing(
        row,
        "status_type_completed",
        "status_completed",
        "completed",
        default=None,
    )
    if direct is not None and _as_bool(direct):
        return True

    status = str(
        _first_existing(
            row,
            "status_type_name",
            "status_type_description",
            "status_type_detail",
            "status_type_short_detail",
            "status",
            default="",
        )
    ).lower()
    return "final" in status or "completed" in status


def _extract_game_date(row: pd.Series) -> Optional[date]:
    value = _first_existing(row, "game_date", "date", "start_date", "start_time", default=None)
    if value is None:
        return None
    parsed = pd.to_datetime(value, errors="coerce")
    if pd.isna(parsed):
        return None
    return pd.Timestamp(parsed).date()


def _fetch_day(day: date) -> tuple[date, Optional[pd.DataFrame], Optional[Exception]]:
    try:
        data = mbb.espn_mbb_schedule(
            dates=int(day.strftime("%Y%m%d")),
            groups=50,
            season_type=None,
            limit=1000,
            return_as_pandas=True,
        )
        if data is None:
            return day, None, None
        if not isinstance(data, pd.DataFrame):
            data = pd.DataFrame(data)
        if data.empty:
            return day, None, None
        return day, data, None
    except Exception as exc:
        return day, None, exc


def fetch_schedule(start: date, end: date, max_workers: int) -> pd.DataFrame:
    days = list(_date_range(start, end))
    frames: list[pd.DataFrame] = []
    errors: list[tuple[date, Exception]] = []

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(_fetch_day, day): day for day in days}
        for future in tqdm(
            as_completed(futures),
            total=len(futures),
            desc="Fetching upcoming schedule",
            unit="day",
        ):
            _, frame, error = future.result()
            if error is not None:
                errors.append((futures[future], error))
            elif frame is not None:
                frames.append(frame)

    if errors:
        print(f"[warn] {len(errors)} schedule dates failed to fetch.")
        for failed_day, error in errors[:10]:
            print(f"       {failed_day}: {error}")
        if len(errors) > 10:
            print(f"       ... plus {len(errors) - 10} more")

    if not frames:
        return pd.DataFrame()

    schedule = pd.concat(frames, ignore_index=True, sort=False)
    id_col = "game_id" if "game_id" in schedule.columns else "id" if "id" in schedule.columns else None
    if id_col is not None:
        schedule = schedule.drop_duplicates(subset=[id_col]).reset_index(drop=True)
    return schedule


def load_team_lookup(path: Path) -> Dict[int, str]:
    if not path.exists():
        return {}

    with path.open("r", encoding="utf-8") as f:
        raw = json.load(f)

    values = raw.values() if isinstance(raw, dict) else raw if isinstance(raw, list) else []
    lookup: Dict[int, str] = {}
    for value in values:
        if not isinstance(value, dict) or "id" not in value:
            continue
        try:
            team_id = int(value["id"])
        except (TypeError, ValueError):
            continue
        lookup[team_id] = (
            value.get("displayName")
            or value.get("display_name")
            or value.get("shortName")
            or value.get("short_display_name")
            or value.get("abbreviation")
            or f"Team {team_id}"
        )
    return lookup


def load_model_team_ids(model_meta_path: Path) -> set[int]:
    with model_meta_path.open("r", encoding="utf-8") as f:
        meta = json.load(f)
    mapping = meta.get("teamIdToIndex") or meta.get("team_id_to_index")
    if not mapping:
        raise ValueError(f"{model_meta_path} has no teamIdToIndex/team_id_to_index mapping")
    return {int(team_id) for team_id in mapping.keys()}


def infer_team1_role(completed_games: pd.DataFrame) -> str:
    if not {"team1home", "team2home"}.issubset(completed_games.columns):
        print("[warn] Could not infer historical team ordering; defaulting team1 to home.")
        return "home"

    non_neutral = completed_games.loc[
        completed_games["team1home"].astype(bool) != completed_games["team2home"].astype(bool)
    ]
    if non_neutral.empty:
        print("[warn] No non-neutral historical games found; defaulting team1 to home.")
        return "home"

    team1_home_rate = float(non_neutral["team1home"].astype(bool).mean())
    if team1_home_rate >= 0.75:
        role = "home"
    elif team1_home_rate <= 0.25:
        role = "away"
    else:
        role = "home"
        print(
            f"[warn] Historical team1 ordering is mixed (team1 home rate={team1_home_rate:.3f}). "
            "Using team1=home for future games."
        )

    print(f"[info] Future matchup orientation: team1={role} (historical team1 home rate={team1_home_rate:.3f})")
    return role


def schedule_row_to_game(
    row: pd.Series,
    season: int,
    team1_role: str,
    team_lookup: Dict[int, str],
) -> Optional[Dict[str, Any]]:
    home_id_raw = _first_existing(row, "home_id", "home_team_id", default=None)
    away_id_raw = _first_existing(row, "away_id", "away_team_id", default=None)
    if home_id_raw is None or away_id_raw is None:
        return None

    try:
        home_id = int(home_id_raw)
        away_id = int(away_id_raw)
    except (TypeError, ValueError):
        return None

    game_day = _extract_game_date(row)
    if game_day is None:
        return None

    neutral = _as_bool(_first_existing(row, "neutral_site", "neutral", default=False))

    home_name = team_lookup.get(home_id) or str(
        _first_existing(
            row,
            "home_display_name",
            "home_name",
            "home_short_display_name",
            "home_location",
            default=f"Team {home_id}",
        )
    )
    away_name = team_lookup.get(away_id) or str(
        _first_existing(
            row,
            "away_display_name",
            "away_name",
            "away_short_display_name",
            "away_location",
            default=f"Team {away_id}",
        )
    )

    if team1_role == "away":
        team1_id, team2_id = away_id, home_id
        team1_name, team2_name = away_name, home_name
        team1_home, team2_home = False, not neutral
    else:
        team1_id, team2_id = home_id, away_id
        team1_name, team2_name = home_name, away_name
        team1_home, team2_home = not neutral, False

    if neutral:
        team1_home = False
        team2_home = False

    raw_game_id = _first_existing(row, "game_id", "id", default=None)
    game_id = str(raw_game_id) if raw_game_id is not None else f"{season}_{game_day:%Y%m%d}_{team1_id}_{team2_id}"

    return {
        "id": game_id,
        "season": int(season),
        "date": game_day.isoformat(),
        "team1Id": int(team1_id),
        "team1Name": team1_name,
        "team2Id": int(team2_id),
        "team2Name": team2_name,
        "team1Home": bool(team1_home),
        "team2Home": bool(team2_home),
    }


def build_upcoming_games(
    completed_csv: Path,
    teams_json: Path,
    model_meta_path: Path,
    output_path: Path,
    season: int,
    season_start: date,
    season_end: date,
    max_workers: int,
) -> None:
    print(f"[info] Loading completed-game database from {completed_csv}")
    all_completed = load_games_csv(str(completed_csv))
    team1_role = infer_team1_role(all_completed)

    season_completed = all_completed.loc[all_completed["season"].astype(int) == int(season)].copy()
    if season_completed.empty:
        data_through = None
        print(f"[info] Completed {season} games currently in database: 0")
    else:
        data_through = pd.Timestamp(season_completed["date"].max()).date().isoformat()
        print(
            f"[info] Completed {season} games currently in database: {len(season_completed):,} "
            f"(through {data_through})"
        )

    # Only complete box-score rows enter SeasonToDateFeatureStore. Therefore future
    # schedule rows can never accidentally become part of a team's statistical history.
    cfg = FeatureConfig(verbose=False)
    feature_store = SeasonToDateFeatureStore(season_completed, cfg=cfg)

    today = date.today()
    fetch_start = max(today, season_start)
    if fetch_start > season_end:
        raise ValueError(f"No remaining dates in season window {season_start} through {season_end}")

    print(f"[info] Fetching schedule from {fetch_start} through {season_end}")
    schedule = fetch_schedule(fetch_start, season_end, max_workers=max_workers)
    if schedule.empty:
        raise RuntimeError("ESPN returned no schedule rows for the requested future date range")

    if "season" in schedule.columns:
        numeric_season = pd.to_numeric(schedule["season"], errors="coerce")
        schedule = schedule.loc[numeric_season == int(season)].copy()

    team_lookup = load_team_lookup(teams_json)
    model_team_ids = load_model_team_ids(model_meta_path)

    upcoming_games: list[Dict[str, Any]] = []
    skipped_completed = 0
    skipped_unknown_team = 0
    skipped_malformed = 0

    for _, row in tqdm(
        schedule.iterrows(),
        total=len(schedule),
        desc="Building future-game features",
        unit="game",
    ):
        game_day = _extract_game_date(row)
        if game_day is None or game_day < today or _is_completed_schedule_row(row):
            skipped_completed += 1
            continue

        game = schedule_row_to_game(row, season, team1_role, team_lookup)
        if game is None:
            skipped_malformed += 1
            continue

        if game["team1Id"] not in model_team_ids or game["team2Id"] not in model_team_ids:
            skipped_unknown_team += 1
            continue

        query_date = pd.Timestamp(game["date"])
        team1_features = feature_store.get_team_feature_vector_as_of(
            game["team1Id"], query_date, season
        )
        team2_features = feature_store.get_team_feature_vector_as_of(
            game["team2Id"], query_date, season
        )

        game["team1Features"] = [float(np.float32(v)) for v in team1_features]
        game["team2Features"] = [float(np.float32(v)) for v in team2_features]

        wins_idx = feature_store.feature_names.index("wins")
        losses_idx = feature_store.feature_names.index("losses")
        game["team1WinsBefore"] = int(round(float(team1_features[wins_idx])))
        game["team1LossesBefore"] = int(round(float(team1_features[losses_idx])))
        game["team2WinsBefore"] = int(round(float(team2_features[wins_idx])))
        game["team2LossesBefore"] = int(round(float(team2_features[losses_idx])))

        upcoming_games.append(game)

    upcoming_games.sort(key=lambda g: (g["date"], g["team1Name"], g["team2Name"], g["id"]))

    payload = {
        "generatedAt": datetime.now(timezone.utc).isoformat(),
        "dataThrough": data_through,
        "season": int(season),
        "featureNames": feature_store.feature_names,
        "games": upcoming_games,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False)

    print(f"\n[success] Wrote {len(upcoming_games):,} predictable upcoming games to {output_path}")
    print(f"[info] Skipped completed/past rows: {skipped_completed:,}")
    print(f"[info] Skipped games with teams absent from model embeddings: {skipped_unknown_team:,}")
    print(f"[info] Skipped malformed schedule rows: {skipped_malformed:,}")
    if data_through is None:
        print(
            "[info] No completed games from this season are currently in mbb_games.csv, "
            "so all season-to-date statistical features are zero for now."
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Build the static upcoming-games artifact used by the GitHub Pages frontend. "
            "Future features use only completed games already present in mbb_games.csv."
        )
    )
    parser.add_argument("--completed-csv", default="data/mbb_games.csv")
    parser.add_argument("--teams-json", default="web/public/data/teams.json")
    parser.add_argument("--model-meta", default="web/public/model/model_meta.json")
    parser.add_argument("--output", default="web/public/data/upcoming_games.json")
    parser.add_argument("--season", type=int, default=DEFAULT_SEASON)
    parser.add_argument("--start", default=DEFAULT_START.isoformat())
    parser.add_argument("--end", default=DEFAULT_END.isoformat())
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()

    def repo_path(value: str) -> Path:
        path = Path(value)
        return path if path.is_absolute() else REPO_ROOT / path

    build_upcoming_games(
        completed_csv=repo_path(args.completed_csv),
        teams_json=repo_path(args.teams_json),
        model_meta_path=repo_path(args.model_meta),
        output_path=repo_path(args.output),
        season=args.season,
        season_start=date.fromisoformat(args.start),
        season_end=date.fromisoformat(args.end),
        max_workers=args.workers,
    )


if __name__ == "__main__":
    main()
