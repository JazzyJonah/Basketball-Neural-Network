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
        display_name = value.get("displayName") or value.get("shortName") or value.get("abbreviation") or str(team_id)
        lookup[team_id] = display_name
    return lookup


def build_games_artifact(csv_path: str, teams_path: str, out_path: str) -> None:
    games = load_games_csv(csv_path).copy()
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
            "team1WinsBefore": None,
            "team1LossesBefore": None,
            "team2WinsBefore": None,
            "team2LossesBefore": None,
        })

    out_file = Path(out_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False)

    print(f"[done] Wrote {len(records)} games to {out_file}")


if __name__ == "__main__":
    build_games_artifact(
        csv_path=str(REPO_ROOT / "mbb_games.csv"),
        teams_path=str(REPO_ROOT / "web" / "public" / "data" / "teams.json"),
        out_path=str(REPO_ROOT / "web" / "public" / "data" / "games" / "all_games.json"),
    )