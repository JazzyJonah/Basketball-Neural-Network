from __future__ import annotations

import json
from pathlib import Path
import sys

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from season_features import load_games_csv


def load_team_lookup(teams_path: Path) -> dict[int, str]:
    with open(teams_path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    lookup = {}
    for _, value in raw.items():
        team_id = int(value["id"])
        display_name = (
            value.get("displayName")
            or value.get("shortName")
            or value.get("abbreviation")
            or str(team_id)
        )
        lookup[team_id] = display_name
    return lookup


def add_pregame_records(games: pd.DataFrame) -> pd.DataFrame:
    games = games.copy()
    games["team1WinsBefore"] = 0
    games["team1LossesBefore"] = 0
    games["team2WinsBefore"] = 0
    games["team2LossesBefore"] = 0

    for season, season_df in games.groupby("season", sort=False):
        record_map: dict[int, list[int]] = {}

        ordered = season_df.sort_values(["date", "team1id", "team2id"])
        for idx, row in ordered.iterrows():
            t1 = int(row["team1id"])
            t2 = int(row["team2id"])

            if t1 not in record_map:
                record_map[t1] = [0, 0]
            if t2 not in record_map:
                record_map[t2] = [0, 0]

            games.at[idx, "team1WinsBefore"] = record_map[t1][0]
            games.at[idx, "team1LossesBefore"] = record_map[t1][1]
            games.at[idx, "team2WinsBefore"] = record_map[t2][0]
            games.at[idx, "team2LossesBefore"] = record_map[t2][1]

            t1_win = int(row["team1pts"] > row["team2pts"])
            t2_win = 1 - t1_win

            record_map[t1][0] += t1_win
            record_map[t1][1] += (1 - t1_win)
            record_map[t2][0] += t2_win
            record_map[t2][1] += (1 - t2_win)

    return games


def build_games_json(csv_path: str, teams_path: str, out_path: str) -> None:
    games = load_games_csv(csv_path).copy()
    games = add_pregame_records(games)
    team_lookup = load_team_lookup(Path(teams_path))

    games["id"] = (
        games["season"].astype(int).astype(str)
        + "_"
        + pd.to_datetime(games["date"]).dt.strftime("%Y%m%d")
        + "_"
        + games["team1id"].astype(int).astype(str)
        + "_"
        + games["team2id"].astype(int).astype(str)
    )

    records = []
    for row in games.itertuples(index=False):
        records.append({
            "id": row.id,
            "season": int(row.season),
            "date": pd.Timestamp(row.date).strftime("%Y-%m-%d"),
            "team1Id": int(row.team1id),
            "team1Name": team_lookup.get(int(row.team1id), f"Team {int(row.team1id)}"),
            "team2Id": int(row.team2id),
            "team2Name": team_lookup.get(int(row.team2id), f"Team {int(row.team2id)}"),
            "team1Home": bool(row.team1home),
            "team2Home": bool(row.team2home),
            "team1Score": int(row.team1pts),
            "team2Score": int(row.team2pts),
            "team1WinsBefore": int(row.team1WinsBefore),
            "team1LossesBefore": int(row.team1LossesBefore),
            "team2WinsBefore": int(row.team2WinsBefore),
            "team2LossesBefore": int(row.team2LossesBefore),
        })

    out_file = Path(out_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False)

    print(f"[done] Wrote {len(records)} games to {out_file}")


if __name__ == "__main__":
    build_games_json(
        csv_path=str(REPO_ROOT / "mbb_games.csv"),
        teams_path=str(REPO_ROOT / "web" / "public" / "data" / "teams.json"),
        out_path=str(REPO_ROOT / "web" / "public" / "data" / "games" / "all_games.json"),
    )