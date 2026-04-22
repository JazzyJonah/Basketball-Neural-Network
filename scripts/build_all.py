from __future__ import annotations

import argparse
from pathlib import Path

from build_games import build_games_json
from build_snapshots import build_snapshots
from build_teams import build_teams_json
from export_model_to_onnx import export_model


def main() -> None:
    parser = argparse.ArgumentParser(description='Build all static artifacts for the historical viewer.')
    parser.add_argument('--csv', default='mbb_games.csv')
    parser.add_argument('--web-root', default='web/public')
    parser.add_argument('--feature-cache', default='team_feature_cache.pkl.gz')
    parser.add_argument('--model-dir', default='model_outputs')
    args = parser.parse_args()

    web_root = Path(args.web_root)
    data_root = web_root / 'data'
    model_root = web_root / 'model'

    build_teams_json(str(data_root / 'teams.json'))
    build_games_json(args.csv, str(data_root / 'teams.json'), str(data_root / 'games'))
    build_snapshots(args.csv, str(data_root / 'snapshots'), args.feature_cache, verbose=True)
    export_model(args.model_dir, str(model_root))
    print('[success] All artifacts built.')


if __name__ == '__main__':
    main()
