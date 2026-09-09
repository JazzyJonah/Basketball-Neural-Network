from __future__ import annotations

import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, timedelta
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Tuple
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import sportsdataverse.mbb as mbb
from tqdm.auto import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = REPO_ROOT / "data" / "mbb_games.csv"

SEASON = 2027
SEASON_START = date(2026, 11, 1)
SEASON_END = date(2027, 4, 30)

# Recheck the last week each day. This catches API hiccups and late-posted box scores.
LOOKBACK_DAYS = 7
MAX_WORKERS = 8

CSV_COLUMNS = [
    "season",
    "date",
    "team1id",
    "team1home",
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
    "team2home",
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

MINIMUMS = {
    "pts": 20,
    "fgm": 5,
    "fga": 15,
    "tpa": 5,
    "ftm": 2,
    "fta": 2,
    "oreb": 3,
    "dreb": 5,
    "ast": 1,
    "to": 1,
    "pf": 1,
}


def daterange(start: date, end: date) -> Iterable[date]:
    current = start
    while current <= end:
        yield current
        current += timedelta(days=1)


def _first(mapping: Dict[str, Any], names: Iterable[str], default: Any = None) -> Any:
    for name in names:
        if name in mapping:
            value = mapping[name]
            if value is not None and not (isinstance(value, float) and np.isnan(value)):
                return value
    return default


def _to_int(value: Any) -> Optional[int]:
    if value is None:
        return None
    try:
        if isinstance(value, str):
            value = value.strip()
            if not value:
                return None
        return int(float(value))
    except (TypeError, ValueError):
        return None


def _to_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    text = str(value).strip().lower()
    return text in {"1", "true", "t", "yes", "y"}


def _parse_made_attempted(value: Any) -> Tuple[Optional[int], Optional[int]]:
    if value is None:
        return None, None

    text = str(value).strip()
    if "-" not in text:
        return None, None

    left, right = text.split("-", 1)
    return _to_int(left), _to_int(right)


def _stat_map(team_box: Dict[str, Any]) -> Dict[str, Any]:
    result: Dict[str, Any] = {}
    stats = team_box.get("statistics", [])

    if isinstance(stats, dict):
        return stats

    if isinstance(stats, list):
        for item in stats:
            if not isinstance(item, dict):
                continue
            name = item.get("name")
            if name:
                result[str(name)] = item.get("displayValue", item.get("value"))

    return result


def _parse_team_box(team_box: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    team = team_box.get("team") or {}
    team_id = _to_int(team.get("id"))
    if team_id is None:
        return None

    stats = _stat_map(team_box)

    fgm, fga = _parse_made_attempted(stats.get("fieldGoalsMade-fieldGoalsAttempted"))
    tpm, tpa = _parse_made_attempted(
        stats.get("threePointFieldGoalsMade-threePointFieldGoalsAttempted")
    )
    ftm, fta = _parse_made_attempted(stats.get("freeThrowsMade-freeThrowsAttempted"))

    values = {
        "id": team_id,
        "homeAway": str(team_box.get("homeAway", "")).lower(),
        "fgm": fgm,
        "fga": fga,
        "tpm": tpm,
        "tpa": tpa,
        "ftm": ftm,
        "fta": fta,
        "oreb": _to_int(stats.get("offensiveRebounds")),
        "dreb": _to_int(stats.get("defensiveRebounds")),
        "ast": _to_int(stats.get("assists")),
        "stl": _to_int(stats.get("steals")),
        "blk": _to_int(stats.get("blocks")),
        "to": _to_int(stats.get("turnovers")),
        "pf": _to_int(stats.get("fouls")),
    }

    required = [
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
    if any(values[name] is None for name in required):
        return None

    # Score from the box score statistics. If an official score is available
    # from the schedule, that score is used instead and validated later.
    values["calculated_pts"] = (
        2 * (values["fgm"] - values["tpm"]) + 3 * values["tpm"] + values["ftm"]
    )

    return values


def _schedule_game_id(row: Dict[str, Any]) -> Optional[str]:
    value = _first(row, ["game_id", "id", "gameId", "event_id"])
    if value is None:
        return None
    return str(value).strip()


def _schedule_score(row: Dict[str, Any], side: str) -> Optional[int]:
    candidates = [
        f"{side}_score",
        f"{side}_points",
        f"{side}_pts",
    ]
    return _to_int(_first(row, candidates))


def _fetch_schedule_day(
    day: date,
) -> Tuple[date, Optional[pd.DataFrame], Optional[str]]:
    try:
        df = mbb.espn_mbb_schedule(
            dates=int(day.strftime("%Y%m%d")),
            groups=50,
            season_type=None,
            limit=1000,
            return_as_pandas=True,
        )
        if df is None:
            return day, None, None
        if not isinstance(df, pd.DataFrame):
            df = pd.DataFrame(df)
        if df.empty:
            return day, None, None
        return day, df, None
    except Exception as exc:
        return day, None, repr(exc)


def _fetch_game(
    game_id: str,
    game_date: date,
    schedule_row: Dict[str, Any],
) -> Tuple[str, Optional[Dict[str, Any]], Optional[str]]:
    try:
        pbp = mbb.espn_mbb_pbp(game_id, raw=False)
        if not isinstance(pbp, dict):
            return game_id, None, "PBP response was not a dict"

        boxscore = pbp.get("boxscore") or {}
        teams = boxscore.get("teams") or []
        if len(teams) != 2:
            return game_id, None, f"Expected 2 team boxes, found {len(teams)}"

        parsed = [_parse_team_box(team_box) for team_box in teams]
        if any(team is None for team in parsed):
            return game_id, None, "Missing required team box statistics"

        team_a, team_b = parsed  # type: ignore[misc]

        # Keep the same away/home orientation used by the schedule whenever possible.
        if team_a["homeAway"] == "away" and team_b["homeAway"] == "home":
            away, home = team_a, team_b
        elif team_b["homeAway"] == "away" and team_a["homeAway"] == "home":
            away, home = team_b, team_a
        else:
            # ESPN normally supplies homeAway even for neutral-site games.
            away, home = team_a, team_b

        neutral = _to_bool(
            _first(
                schedule_row,
                ["neutral_site", "neutralSite", "neutral", "is_neutral"],
                False,
            )
        )

        away_score = _schedule_score(schedule_row, "away")
        home_score = _schedule_score(schedule_row, "home")

        if away_score is None:
            away_score = int(away["calculated_pts"])
        if home_score is None:
            home_score = int(home["calculated_pts"])

        row = {
            "season": SEASON,
            "date": game_date.isoformat(),
            "team1id": int(away["id"]),
            "team1home": False,
            "team1pts": away_score,
            "team1fgm": int(away["fgm"]),
            "team1fga": int(away["fga"]),
            "team13ptm": int(away["tpm"]),
            "team13pta": int(away["tpa"]),
            "team1ftm": int(away["ftm"]),
            "team1fta": int(away["fta"]),
            "team1oreb": int(away["oreb"]),
            "team1dreb": int(away["dreb"]),
            "team1ast": int(away["ast"]),
            "team1stl": int(away["stl"]),
            "team1blk": int(away["blk"]),
            "team1to": int(away["to"]),
            "team1pf": int(away["pf"]),
            "team2id": int(home["id"]),
            "team2home": False if neutral else True,
            "team2pts": home_score,
            "team2fgm": int(home["fgm"]),
            "team2fga": int(home["fga"]),
            "team23ptm": int(home["tpm"]),
            "team23pta": int(home["tpa"]),
            "team2ftm": int(home["ftm"]),
            "team2fta": int(home["fta"]),
            "team2oreb": int(home["oreb"]),
            "team2dreb": int(home["dreb"]),
            "team2ast": int(home["ast"]),
            "team2stl": int(home["stl"]),
            "team2blk": int(home["blk"]),
            "team2to": int(home["to"]),
            "team2pf": int(home["pf"]),
        }

        valid, reason = validate_game_row(row)
        if not valid:
            return game_id, None, reason

        return game_id, row, None

    except Exception as exc:
        return game_id, None, repr(exc)


def _validate_team(row: Dict[str, Any], prefix: str) -> Tuple[bool, str]:
    if prefix == "team1":
        values = {
            "pts": row["team1pts"],
            "fgm": row["team1fgm"],
            "fga": row["team1fga"],
            "tpm": row["team13ptm"],
            "tpa": row["team13pta"],
            "ftm": row["team1ftm"],
            "fta": row["team1fta"],
            "oreb": row["team1oreb"],
            "dreb": row["team1dreb"],
            "ast": row["team1ast"],
            "stl": row["team1stl"],
            "blk": row["team1blk"],
            "to": row["team1to"],
            "pf": row["team1pf"],
        }
    else:
        values = {
            "pts": row["team2pts"],
            "fgm": row["team2fgm"],
            "fga": row["team2fga"],
            "tpm": row["team23ptm"],
            "tpa": row["team23pta"],
            "ftm": row["team2ftm"],
            "fta": row["team2fta"],
            "oreb": row["team2oreb"],
            "dreb": row["team2dreb"],
            "ast": row["team2ast"],
            "stl": row["team2stl"],
            "blk": row["team2blk"],
            "to": row["team2to"],
            "pf": row["team2pf"],
        }

    if any(value is None for value in values.values()):
        return False, f"{prefix}: missing statistic"

    if any(float(value) < 0 for value in values.values()):
        return False, f"{prefix}: negative statistic"

    for stat, minimum in MINIMUMS.items():
        if values[stat] < minimum:
            return False, f"{prefix}: {stat}={values[stat]} below minimum {minimum}"

    if values["fgm"] > values["fga"]:
        return False, f"{prefix}: FGM > FGA"
    if values["tpm"] > values["tpa"]:
        return False, f"{prefix}: 3PM > 3PA"
    if values["tpm"] > values["fgm"]:
        return False, f"{prefix}: 3PM > FGM"
    if values["ftm"] > values["fta"]:
        return False, f"{prefix}: FTM > FTA"
    if values["ast"] > values["fgm"]:
        return False, f"{prefix}: AST > FGM"

    calculated_pts = (
        2 * (values["fgm"] - values["tpm"]) + 3 * values["tpm"] + values["ftm"]
    )
    tolerance = max(1.0, 0.05 * float(values["pts"]))
    if abs(float(values["pts"]) - float(calculated_pts)) > tolerance:
        return (
            False,
            f"{prefix}: points inconsistent with shooting stats "
            f"({values['pts']} vs calculated {calculated_pts})",
        )

    return True, ""


def validate_game_row(row: Dict[str, Any]) -> Tuple[bool, str]:
    try:
        game_date = pd.Timestamp(row["date"]).date()
    except Exception:
        return False, "invalid date"

    if not (SEASON_START <= game_date <= SEASON_END):
        return False, f"date {game_date} outside season {SEASON}"

    if bool(row["team1home"]) and bool(row["team2home"]):
        return False, "both teams marked home"

    ok, reason = _validate_team(row, "team1")
    if not ok:
        return ok, reason

    ok, reason = _validate_team(row, "team2")
    if not ok:
        return ok, reason

    # Across the whole game, total rebounds should roughly match missed field
    # goals plus roughly half of missed free throws. Keep the original broad
    # 25% tolerance so obviously broken box scores are rejected.
    total_rebounds = (
        row["team1oreb"] + row["team1dreb"] + row["team2oreb"] + row["team2dreb"]
    )
    rebound_opportunities = (
        (row["team1fga"] - row["team1fgm"])
        + (row["team2fga"] - row["team2fgm"])
        + 0.5
        * ((row["team1fta"] - row["team1ftm"]) + (row["team2fta"] - row["team2ftm"]))
    )

    if rebound_opportunities > 0:
        rebound_error = (
            abs(total_rebounds - rebound_opportunities) / rebound_opportunities
        )
        if rebound_error > 0.25:
            return (
                False,
                f"rebound sanity check failed "
                f"(rebounds={total_rebounds}, opportunities={rebound_opportunities:.1f})",
            )

    return True, ""


def _canonical_key(
    season: int,
    game_date: Any,
    team1id: int,
    team2id: int,
) -> Tuple[int, str, int, int]:
    date_text = pd.Timestamp(game_date).strftime("%Y-%m-%d")
    low, high = sorted((int(team1id), int(team2id)))
    return int(season), date_text, low, high


def determine_fetch_window(
    existing: pd.DataFrame,
) -> Tuple[Optional[date], Optional[date]]:
    eastern_today = pd.Timestamp.now(tz=ZoneInfo("America/New_York")).date()

    # Never ingest today's games. The workflow runs in the morning, so through
    # yesterday is guaranteed to exclude games that are currently in progress.
    end = min(eastern_today - timedelta(days=1), SEASON_END)

    if end < SEASON_START:
        return None, None

    season_rows = existing[
        pd.to_numeric(existing["season"], errors="coerce") == SEASON
    ].copy()

    if season_rows.empty:
        start = SEASON_START
    else:
        dates = pd.to_datetime(season_rows["date"], errors="coerce").dropna()
        if dates.empty:
            start = SEASON_START
        else:
            last_date = dates.max().date()
            start = max(SEASON_START, last_date - timedelta(days=LOOKBACK_DAYS))

    return start, end


def main() -> None:
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Missing historical database: {DATA_PATH}")

    existing = pd.read_csv(DATA_PATH)

    missing_cols = [col for col in CSV_COLUMNS if col not in existing.columns]
    if missing_cols:
        raise ValueError(f"{DATA_PATH} is missing columns: {missing_cols}")

    start, end = determine_fetch_window(existing)
    if start is None or end is None:
        print(
            f"[info] No completed {SEASON} games can exist yet. "
            f"Season begins {SEASON_START}."
        )
        return

    days = list(daterange(start, end))
    print(f"[info] Checking completed-game window: {start} through {end}")
    print(f"[info] Rechecking {LOOKBACK_DAYS} trailing days to catch late API updates.")

    schedule_rows: Dict[str, Tuple[date, Dict[str, Any]]] = {}
    schedule_errors = []

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(_fetch_schedule_day, day): day for day in days}

        for future in tqdm(
            as_completed(futures),
            total=len(futures),
            desc="Fetching completed schedules",
            unit="day",
        ):
            day, df, error = future.result()

            if error:
                schedule_errors.append((day, error))
                continue
            if df is None:
                continue

            for _, series in df.iterrows():
                row = series.to_dict()

                if "season" in row:
                    row_season = _to_int(row.get("season"))
                    if row_season is not None and row_season != SEASON:
                        continue

                game_id = _schedule_game_id(row)
                if game_id:
                    schedule_rows[game_id] = (day, row)

    print(f"[info] Found {len(schedule_rows):,} schedule entries in the fetch window.")

    if schedule_errors:
        print(f"[warn] {len(schedule_errors)} schedule dates failed.")
        for day, error in schedule_errors[:5]:
            print(f"       {day}: {error}")

    if not schedule_rows:
        print("[done] No schedule entries to inspect.")
        return

    existing_keys = {
        _canonical_key(
            int(row.season),
            row.date,
            int(row.team1id),
            int(row.team2id),
        )
        for row in existing.itertuples(index=False)
    }

    new_rows = []
    skipped_existing = 0
    rejected = []

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {
            executor.submit(_fetch_game, game_id, game_date, schedule_row): game_id
            for game_id, (game_date, schedule_row) in schedule_rows.items()
        }

        bar = tqdm(
            as_completed(futures),
            total=len(futures),
            desc="Fetching completed box scores",
            unit="game",
        )

        for future in bar:
            game_id, row, error = future.result()

            if row is None:
                rejected.append((game_id, error or "unknown error"))
                bar.set_postfix(new=len(new_rows), rejected=len(rejected))
                continue

            key = _canonical_key(
                row["season"],
                row["date"],
                row["team1id"],
                row["team2id"],
            )

            if key in existing_keys:
                skipped_existing += 1
                bar.set_postfix(
                    new=len(new_rows),
                    existing=skipped_existing,
                    rejected=len(rejected),
                )
                continue

            existing_keys.add(key)
            new_rows.append(row)

            bar.set_postfix(
                new=len(new_rows),
                existing=skipped_existing,
                rejected=len(rejected),
            )

    if not new_rows:
        print(
            f"[done] No new completed games. "
            f"Already present={skipped_existing:,}, rejected/unavailable={len(rejected):,}."
        )
        if rejected:
            print("[info] First few unavailable/rejected games:")
            for game_id, reason in rejected[:10]:
                print(f"       {game_id}: {reason}")
        return

    new_df = pd.DataFrame(new_rows, columns=CSV_COLUMNS)
    combined = pd.concat([existing[CSV_COLUMNS], new_df], ignore_index=True)

    combined["date"] = pd.to_datetime(combined["date"], errors="raise").dt.strftime(
        "%Y-%m-%d"
    )
    combined = combined.sort_values(
        ["season", "date", "team1id", "team2id"]
    ).reset_index(drop=True)

    # Atomic write: do not risk corrupting the historical database if the job
    # is interrupted during the CSV save.
    tmp_path = DATA_PATH.with_suffix(".csv.tmp")
    combined.to_csv(tmp_path, index=False)
    os.replace(tmp_path, DATA_PATH)

    print(f"\n[success] Added {len(new_df):,} completed games.")
    print(f"[success] Database now contains {len(combined):,} games.")
    print(f"[success] Updated {DATA_PATH}")

    if rejected:
        print(f"[info] {len(rejected):,} schedule entries were unavailable/rejected.")
        for game_id, reason in rejected[:10]:
            print(f"       {game_id}: {reason}")


if __name__ == "__main__":
    main()
