from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
from tqdm.auto import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
from build_games import build_games_json
from season_features import FeatureConfig, SeasonToDateFeatureStore, load_games_csv


def build_snapshots(csv_path: str, output_dir: str, feature_cache_path: str, verbose: bool = True) -> None:
    games = load_games_csv(csv_path)
    cfg = FeatureConfig(feature_cache_path=feature_cache_path, verbose=verbose)
    store = SeasonToDateFeatureStore(games, cfg=cfg)
    feature_table = store.build_or_load_feature_cache(rebuild=False)

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    grouped = feature_table.groupby('season', sort=True)
    for season, season_df in tqdm(grouped, total=len(grouped), desc='Writing snapshot files', unit='season'):
        season_df = season_df.sort_values(['date', 'team_id']).reset_index(drop=True)
        rows = []
        for row in season_df.itertuples(index=False):
            features = [float(np.float32(getattr(row, feature_name))) for feature_name in store.feature_names]
            rows.append({
                'date': str(getattr(row, 'date').date()),
                'teamId': int(getattr(row, 'team_id')),
                'features': features,
            })

        payload = {
            'season': int(season),
            'featureNames': store.feature_names,
            'rows': rows,
        }

        output_path = output / f'{int(season)}.json'
        with output_path.open('w', encoding='utf-8') as f:
            json.dump(payload, f, ensure_ascii=False)
        print(f'[success] Wrote {len(rows)} team-date snapshots to {output_path}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Export per-season historical team snapshots for browser inference.')
    parser.add_argument('--csv', default='mbb_games.csv')
    parser.add_argument('--output-dir', default='web/public/data/snapshots')
    parser.add_argument('--feature-cache', default='team_feature_cache.pkl.gz')
    parser.add_argument('--quiet', action='store_true')
    args = parser.parse_args()
    build_snapshots(args.csv, args.output_dir, args.feature_cache, verbose=not args.quiet)
