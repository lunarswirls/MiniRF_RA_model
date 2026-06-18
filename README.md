# MiniRF_RA_model

This repository contains a Python translation of a Mini-RF RA prediction workflow previously developed in MATLAB. 
The current code focuses on rebuilding the 2D radar maps, running a two-pass RA prediction pipeline, and tuning the resulting maps.

## Project summary

The code uses six Mini-RF-derived input channels plus a reference `ra` map from the Giordano Bruno dataset:

- `miniRF_1` through `miniRF_6`
- `ra`

From those inputs, the pipeline derives two radar features used by the translated model:

- `CPR`
- `m-chi G (volume scattering)`

It then predicts an RA map with:

1. A first pass that estimates a local MAP RA value from weighted spatial neighborhoods.
2. A second pass that refines each pixel with a Bayesian inversion that uses the first-pass map as a spatial prior.
3. An evaluation step that reports global, regime-wise, high-RA, uncertainty, and spatial roughness metrics.

## What the code currently does

### `src/parallel_ra_prediction.py`

This is the main pipeline. It currently:

- Loads analysis data from a CSV file.
- Reconstructs the original 2D arrays from a flattened table with `row`, `col`, and variable columns.
- Applies the Mini-RF preprocessing used by the source workflow.
- Uses hardcoded scattering-model fits translated from the earlier workflow.
- Runs the first-pass local inversion and the second-pass spatially regularized inversion.
- Evaluates the final prediction against the reference `ra` map.

The script is configured through lowercase variables at the top of the file instead of command-line arguments:

- `data_path`
- `workers`

Running it from the repository root:

```bash
python3 src/parallel_ra_prediction.py
```

### `src/hyperparameter_search.py`

This script wraps the same translated prediction pipeline in a focused grid search. It currently:

- Builds a candidate grid from top-of-file parameter lists.
- Reuses the default pipeline options from `parallel_ra_prediction.py`.
- Splits the reference map into spatial tiles.
- Scores each candidate on blocked spatial folds.
- Ranks candidates by a scalar objective, currently `balancedRMSE`.
- Optionally writes ranked results to a CSV file.

The search is centered on manually tuned values discussed in the earlier workflow, especially:

- `s`
- `sigmaSpatial`
- `nu`
- `RA0`

Running it from the repository root:

```bash
python3 src/hyperparameter_search.py
```

By default, the current search space expands to 180 candidates.

## Data layout

The current Python code expects a flattened CSV with these columns:

```text
row,col,miniRF_1,miniRF_2,miniRF_3,miniRF_4,miniRF_5,miniRF_6,ra
```

The repository currently includes:

- `data/GiordanoBruno_analysis.csv` as the main analysis input used by the scripts
- `data/GiordanoBruno.mat` as the original source file kept for reference
- `data/ManualTuning_OptParam_RAFromRadarMdl.rtf` as manual tuning notes
- `data/test_small.csv` as a smaller dataset artifact from the earlier workflow

## Evaluation outputs

The translated evaluator currently reports metrics such as:

- RMSE and MAE
- balanced RMSE across RA regimes
- Pearson and Spearman correlation
- high-RA precision/recall/F1
- optional 68% interval coverage and width
- spatial roughness summaries

## Repo structure

```text
README.md
src/
  parallel_ra_prediction.py
  hyperparameter_search.py
  mat_to_csv.py
data/
  GiordanoBruno_analysis.csv
  GiordanoBruno.mat
  ManualTuning_OptParam_RAFromRadarMdl.rtf
  test_small.csv
```
