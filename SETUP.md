# Setup

This should be rather simple to set up, so here are the steps!

1. Pull the code from GitHub. You can download it as a zip, clone the repository, or any other wayt of getting the code from GitHub onto your computer.
2. Install all requirements. This includes everything listed in requirements.txt, as well as npm (in order to locally run the website).
3. Run the setup scripts. Many of the setup steps have been prerun and their associated files are already in the repository. This includes: the data collection (`mbb_games.csv`, `web/public/data/games/`, `web/public/data/snapshots/`, and `weeb/public/data/teams.json`), the models (`model_outputs/best_score_model.pt`, `model_outputs/best_standardizer.pt`, and `web/public/data/model.onnx`), and the metadata (`web/public/model/model_meta.json`). However, you'll still have to generate the cahced features, by running `season_features.py.` This should be all the necessary setup. To regenerate all files, run `train_basketball_model.py`, `season_features.py`, `scripts/build_all.py`, `scripts/export_model_to_onnx.py`, `generate_plots.py`, and `find_best_game.py`.
4. Navigate to the `web` directory (in the Command Prompt or other), and then do `npm install` and `npm run dev`. This will create a local version of the website that you can use.
5. To generate your own 