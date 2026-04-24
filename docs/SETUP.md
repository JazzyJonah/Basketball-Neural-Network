# Setup

This should be rather simple to set up, so here are the steps!

1. Pull the code from GitHub. You can download it as a zip, clone the repository, or any other way of getting the code from GitHub onto your computer.
2. Install all requirements. This includes everything listed in `requirements.txt`, as well as npm (in order to locally run the website).
3. Run the setup scripts. Many of the setup steps have been prerun and their associated files are already in the repository. This includes the data collection in `data/` (for example, `data/mbb_games.csv`), the trained model artifacts in `models/`, and the website data under `web/public/data/`.
4. To regenerate the web artifacts, run the build helper from the repository root:

```bash
python scripts/build_all.py
```

With the default configuration, that command uses:
- `data/mbb_games.csv`
- `data/team_feature_cache.pkl.gz`
- `models/`
- `web/public/data/`
- `web/public/model/`

5. To run the website locally, change into the `web` directory and run:

```bash
npm install
npm run dev
```

6. To rerun training or feature generation locally, use the source scripts in `src/`:

```bash
python -m src.train_basketball_model
python -m src.season_features
python -m src.find_best_game
```
