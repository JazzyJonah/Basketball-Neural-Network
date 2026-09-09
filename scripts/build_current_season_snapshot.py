from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from tqdm.auto import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.season_features import (
    FeatureConfig,
    SeasonToDateFeatureStore,
    load_games_csv,
)


def build_current_season_snapshot(
    season: int = 2027,
    csv_path: str = "data/mbb_games.csv",
    output_dir: str = "web/public/data/snapshots",
) -> None:
    csv_file = REPO_ROOT / csv_path
    output = REPO_ROOT / output_dir
    output.mkdir(parents=True, exist_ok=True)

    games = load_games_csv(str(csv_file))
    season_games = games[games["season"] == season].copy()

    if season_games.empty:
        payload = {
            "season": int(season),
            "featureNames": [],
            "rows": [],
        }
        output_path = output / f"{season}.json"
        with output_path.open("w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False)
        print(
            f"[info] No completed season {season} games yet; wrote empty {output_path}"
        )
        return

    # Build only this season. This avoids rebuilding every historical season
    # during the daily GitHub Action.
    temp_cache = REPO_ROOT / "data" / f".team_feature_cache_{season}.pkl.gz"
    cfg = FeatureConfig(
        feature_cache_path=str(temp_cache),
        verbose=True,
    )

    store = SeasonToDateFeatureStore(season_games, cfg=cfg)
    feature_table = store.build_or_load_feature_cache(rebuild=True)

    feature_table = feature_table.sort_values(["date", "team_id"]).reset_index(
        drop=True
    )

    rows = []
    for row in tqdm(
        feature_table.itertuples(index=False),
        total=len(feature_table),
        desc=f"Writing {season} snapshots",
        unit="team-game",
    ):
        features = [
            float(np.float32(getattr(row, feature_name)))
            for feature_name in store.feature_names
        ]
        rows.append(
            {
                "date": str(getattr(row, "date").date()),
                "teamId": int(getattr(row, "team_id")),
                "features": features,
            }
        )

    payload = {
        "season": int(season),
        "featureNames": store.feature_names,
        "rows": rows,
    }

    output_path = output / f"{season}.json"
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False)

    try:
        temp_cache.unlink(missing_ok=True)
    except OSError:
        pass

    print(f"[success] Wrote {len(rows):,} team-date snapshots to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Build only the current season snapshot for the historical web viewer."
    )
    parser.add_argument("--season", type=int, default=2027)
    parser.add_argument("--csv", default="data/mbb_games.csv")
    parser.add_argument("--output-dir", default="web/public/data/snapshots")
    args = parser.parse_args()

    build_current_season_snapshot(
        season=args.season,
        csv_path=args.csv,
        output_dir=args.output_dir,
    )
