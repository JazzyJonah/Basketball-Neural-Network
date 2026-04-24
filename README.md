# Basketball Neural Network

## What it Does
Project Goal: predict basketball games as accurately as possible! It uses a multi-layer perceptron with three hidden linear layers to convert its inputs (including advanced statistics of each team) into score outputs. It uses snapshots from the past to predict basketball games based on the information available at the time of the snapshot. Currently, only a "historical predictor" is available, since we're in the off-season. However, it has the capabilities to predict future scheduled games, too.

## Quick start
To use the model, just head to the [website](https://jazzyjonah.github.io/Basketball-Neural-Network/)! To copy the model locally, follow the instructions in `docs/SETUP.md`.

## Video Links
Project Demo: https://youtu.be/6jdgDRJym2s, or `videos/Project Demo.mp4`
Technical Walkthrough: https://youtu.be/7cvt6b_UrHI, or `videos/Technical Walkthrough.mp4`

## Evaluation
The best model achieved a total accuracy of 72.8%. For reference, a model that simply chose the team with the better record had an accuracy of only 66.8%. Furthermore, the RMSE was just 10.8. Lastly, the model has a plethora of amazing guesses (available in `models/best_game_by_score.csv`)--it correctly guessed the exact score in over a hundred games. Notably, it exactly predicted the score of Notre Dame's overtime win against Butler in the 2015 Second Round. (The reason this is notable is that Duke won in 2015)!

## Individual Contributions
I was the sole contributor to this project.