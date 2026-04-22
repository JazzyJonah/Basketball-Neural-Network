# Historical Basketball Predictor Scaffold

## What is included

- `scripts/build_teams.py` — one-time SportsDataverse export for `teams.json`
- `scripts/build_games.py` — season game index builder from `mbb_games.csv`
- `scripts/build_snapshots.py` — exports per-season historical team snapshots from `team_feature_cache.pkl.gz`
- `scripts/export_model_to_onnx.py` — exports the trained PyTorch model to ONNX plus browser metadata
- `scripts/build_all.py` — convenience entry point that runs all of the above
- `web/` — a Vite + React + TypeScript historical viewer with:
  - past-game prediction only
  - soft browser-side rate limiting (15 requests per minute)
  - on-page logging panel
  - clear error banner

## Expected local files before running the build scripts

Place these in the same root directory where you run the scripts:

- `mbb_games.csv`
- `season_features.py`
- `basketball_model.py`
- `model_outputs/best_score_model.pt`
- `model_outputs/best_score_model_meta.json`
- `model_outputs/best_standardizer.pt`
- `team_feature_cache.pkl.gz`

## Build the static artifacts

From the project root:

```bash
python scripts/build_all.py --csv mbb_games.csv --feature-cache team_feature_cache.pkl.gz --model-dir model_outputs --web-root web/public
```

## Run the frontend locally

```bash
cd web
npm install
npm run dev
```

## Deploy to GitHub Pages later

Once the static artifacts are inside `web/public`, a normal Vite build will bundle the site.
Then you can deploy `web/dist` with GitHub Pages.

## Notes

- The current scaffold is **historical viewer only**.
- The upcoming-game viewer can reuse the same ONNX model and team snapshot logic later.
- The rate limiting in this version is a **browser-only soft limit**. It is good enough for a demo, but it is not tamper-proof.
