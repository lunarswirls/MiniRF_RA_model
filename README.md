# MiniRF_RA_model

This repository contains Python translations of Mini-RF RA prediction workflows previously developed in MATLAB.
The current code covers both the earlier CSV-based translation and a newer `v2` workflow that uses saved scattering fits plus a CPR-G joint covariance model in the second pass.

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

### `src/parallel_ra_prediction_v2.py`

This is the newer `v2` translation. It currently:

- Loads `data/GiordanoBruno.mat` directly from MATLAB format.
- Loads the exported covariance model from `data/covModel_global_CPR_Green_2026-07-09_085116.npz`.
- Uses the saved `v2` CPR and Green scattering-fit coefficients.
- Runs the first-pass local inversion with the same low-RA swap/clamp logic used in the MATLAB first pass.
- Runs the second pass with `likelihoodMode = jointGaussian` and `covarianceMode = smooth` by default.
- Uses the posterior median as the final `RApred`, while also keeping the posterior mean plus the 68 percent `low` and `high` maps.
- Always computes a continuous Diviner sigma map and a floored sigma-envelope map.
- Optionally compares the final prediction to the saved reference map in `data/TestData_Bruno_RApred.npy`.
- Writes two output figures:
  - `output/ra_prediction_map_sigma.png`
  - `output/ra_prediction_histograms.png`

Running it from the repository root:

```bash
python src/parallel_ra_prediction_v2.py
```

### `src/hyperparameter_search.py`

This script wraps the same translated prediction pipeline in a bounded continuous optimizer. It currently:

- Reuses the default pipeline options from `parallel_ra_prediction.py`.
- Optimizes the main continuous parameters `s`, `sigmaSpatial`, `nu`, and `RA0`.
- Uses `scipy.optimize.direct` for the global pass and bounded `Powell` for local polish.
- Reuses cached first-pass maps so candidates with the same first-pass settings do not recompute the whole first pass.
- Splits the reference map into spatial tiles and scores each candidate on blocked spatial folds.
- Ranks candidates by a scalar objective, currently `balancedRMSE`.
- Penalizes dropped predictions with a completeness term based on `1 - pred_valid_fraction`.
- Optionally writes ranked results to a CSV file in `output/`.

Running it from the repository root:

```bash
./.venv/bin/python src/hyperparameter_search.py
```

The progress output now prints both score and `predValid`, so it is easier to spot candidates that improve RMSE by NaN-ing difficult pixels.

### `src/hyperparameter_search_v2.py`

This script wraps the new `v2` Python pipeline in a bounded continuous optimizer. It currently:

- Reuses the default `v2` pipeline options from `parallel_ra_prediction_v2.py`.
- Optimizes the continuous parameters that still matter in the joint-Gaussian workflow: `s`, `sigmaSpatial`, and `RA0`.
- Uses `scipy.optimize.direct` for the global pass with `maxiter = 100`, then bounded `Powell` for local polish.
- Reuses cached first-pass maps so candidates with the same first-pass settings do not recompute the whole first pass.
- Reuses cached whole-candidate results when `direct` and `Powell` revisit the same parameter vector.
- Splits the reference map into spatial tiles and scores each candidate on blocked spatial folds.
- Penalizes dropped predictions with a completeness term based on `1 - pred_valid_fraction`.
- Hard-rejects candidates if the prediction NaN count on the finite Diviner mask grows by more than 10 percent relative to the baseline `v2` run.
- Optionally writes ranked results to `output/GiordanoBruno_v2_rankedresults.csv`.

Running it from the repository root:

```bash
python src/hyperparameter_search_v2.py
```

The progress output prints score, `predValid`, and NaN-growth rejection messages with `flush=True`, so long searches stay readable in the terminal.

### `src/covariance_diagnostics.py`

This script tests whether CPR and Green residuals co-vary after standardizing by the current fitted mean and scale curves. It currently:

- Loads the same CSV dataset and preprocessing used by the prediction pipeline.
- Computes standardized CPR and Green residuals:
  - `z1 = (CPR - mu1) / sigma1`
  - `z2 = (G - mu2) / sigma2`
- Saves a diagnostic figure with:
  - a residual-cloud plot with covariance ellipses
  - an RA-binned plot of CPR-G slope and residual correlation
- Fits three smooth covariance models, `rho = tanh(eta)`, with constant, `rho(ra)`, and `rho(ra,inc)` variants.
- Prints summary numbers such as global residual `rho`, validation negative log-likelihood, and whether the incidence effect appears weak.

Running it from the repository root:

```bash
python src/covariance_diagnostics.py
```

The default output plot is:

- `output/cpr_green_covariance_diagnostics.png`

### `src/compare_ra_distribution_lookup_vs_smooth_v2.py`

This script compares how the `v2` second pass behaves when the CPR-G covariance is driven by the saved lookup table versus the smooth fitted `rho(ra,inc)` surface. It currently:

- Reuses one shared first-pass result for a fair comparison.
- Runs two second passes with identical priors:
  - `covarianceMode = lookup`
  - `covarianceMode = smooth`
- Compares the resulting RA distributions against the Diviner reference.
- Compares posterior median versus posterior mean distributions for both covariance modes.
- Saves two diagnostic figures:
  - `output/lookup_vs_smooth_ra_distribution.png`
  - `output/posterior_mean_vs_median_ra_distribution.png`

Running it from the repository root:

```bash
python src/compare_ra_distribution_lookup_vs_smooth_v2.py
```

## Data layout

The earlier Python translation expects a flattened CSV with these columns:

```text
row,col,miniRF_1,miniRF_2,miniRF_3,miniRF_4,miniRF_5,miniRF_6,ra
```

The repository currently includes:

- `data/GiordanoBruno_analysis.csv` as the main CSV analysis input used by the earlier workflow
- `data/ManualTuning_OptParam_RAFromRadarMdl.rtf` as manual tuning notes
- `data/test_small.csv` as a smaller dataset artifact from the earlier workflow
- `data/GiordanoBruno.mat` as the MATLAB-format input used by the active `v2` translation
- `data/covModel_global_CPR_Green_2026-07-09_085116.npz` as the exported NumPy covariance grid used by the Python `v2` code
- `data/TestData_Bruno_RApred.npy` as the saved reference RA map used for Python-side `v2` comparison

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
  parallel_ra_prediction_v2.py
  hyperparameter_search.py
  hyperparameter_search_v2.py
  covariance_diagnostics.py
  compare_ra_distribution_lookup_vs_smooth_v2.py
data/
  GiordanoBruno.mat
  GiordanoBruno_analysis.csv
  ManualTuning_OptParam_RAFromRadarMdl.rtf
  TestData_Bruno_RApred.npy
  covModel_global_CPR_Green_2026-07-09_085116.npz
  test_small.csv
output/
  cpr_green_covariance_diagnostics.png
  lookup_vs_smooth_ra_distribution.png
  posterior_mean_vs_median_ra_distribution.png
  ra_prediction_histograms.png
  ra_prediction_map_sigma.png
```
