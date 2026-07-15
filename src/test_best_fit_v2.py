#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
test one fixed v2 best-fit candidate and print the same metrics used in search
"""

import os
from pathlib import Path

repo_root = Path(__file__).resolve().parents[1]
mpl_cache_dir = repo_root / "output/.mplcache"
cache_home_dir = repo_root / "output/.cache"
mpl_cache_dir.mkdir(parents=True, exist_ok=True)
cache_home_dir.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", str(mpl_cache_dir))
os.environ.setdefault("XDG_CACHE_HOME", str(cache_home_dir))

from typing import Any
import numpy as np
import parallel_ra_prediction_v2 as pr
import hyperparameter_search_v2 as hs


Array = np.ndarray

# path to the v2 input dataset
data_path = repo_root / "data/GiordanoBruno.mat"

# number of worker threads for the prediction pipeline
n_workers = 20

# output path for the candidate-specific map and sigma plot
map_plot_output_path = repo_root / "output/best_fit_v2_ra_prediction_map_sigma.png"

# output path for the candidate-specific histogram plot
hist_plot_output_path = repo_root / "output/best_fit_v2_ra_prediction_histograms.png"

# best-fit candidate from eval 427 of the v2 direct plus powell search
best_fit_candidate = {
    "sigmaFloor": 0.02,
    "windowSize": 3,
    "spatialPriorMode": "direct",
    "secondPassWindowSize": 5,
    "s": 1.0720164609053497,
    "sigmaSpatial": 1.2484567901234567,
    "RA0": 0.9502057613168723,
}


def print_search_summary(label: str, result: dict[str, Any]) -> None:
    """print the blockwise search summary for one candidate"""
    summary = result["summary"]
    print(f"\n{label}", flush=True)
    print(f"candidate={result['candidate']}", flush=True)
    print(f"mean_score={summary['mean_score']:.4f}", flush=True)
    print(f"mean_score_unpenalized={summary['mean_score_unpenalized']:.4f}", flush=True)
    print(f"mean_balancedRMSE={summary['mean_balancedRMSE']:.4f}", flush=True)
    print(f"pred_valid_fraction={summary['pred_valid_fraction']:.4f}", flush=True)
    print(f"pred_valid_count={summary['pred_valid_count']}", flush=True)
    print(f"pred_nan_count={summary['pred_nan_count']}", flush=True)
    print(f"valid_fraction_penalty={summary['valid_fraction_penalty']:.4f}", flush=True)
    print(f"hard_rejected_nan_growth={summary['hard_rejected_nan_growth']}", flush=True)
    print(f"reference_mean_diff={summary['reference_mean_diff']:.6f}", flush=True)


def print_candidate_delta(baseline_result: dict[str, Any], best_result: dict[str, Any]) -> None:
    """print the change from the default v2 settings to the tested best-fit candidate"""
    baseline_summary = baseline_result["summary"]
    best_summary = best_result["summary"]
    print("\ndelta relative to default v2 settings", flush=True)
    print(f"delta_mean_score={best_summary['mean_score'] - baseline_summary['mean_score']:.4f}", flush=True)
    print(
        f"delta_balancedRMSE={best_summary['mean_balancedRMSE'] - baseline_summary['mean_balancedRMSE']:.4f}",
        flush=True,
    )
    print(
        f"delta_pred_valid_fraction={best_summary['pred_valid_fraction'] - baseline_summary['pred_valid_fraction']:.4f}",
        flush=True,
    )
    print(f"delta_pred_nan_count={best_summary['pred_nan_count'] - baseline_summary['pred_nan_count']}", flush=True)


def print_global_metrics(metrics: dict[str, Any]) -> None:
    """print a compact global metric summary after the full candidate run"""
    global_metrics = metrics["global"]
    highra_metrics = metrics["highRA"]
    print("\nfull-map global metrics", flush=True)
    print(f"RMSE={global_metrics['RMSE']:.4f}", flush=True)
    print(f"balancedRMSE={global_metrics['balancedRMSE']:.4f}", flush=True)
    print(f"MAE_mean={global_metrics['MAE_mean']:.4f}", flush=True)
    print(f"MAE_median={global_metrics['MAE_median']:.4f}", flush=True)
    print(f"bias_mean={global_metrics['bias_mean']:.4f}", flush=True)
    print(f"bias_median={global_metrics['bias_median']:.4f}", flush=True)
    print(f"coverage68={global_metrics['coverage68']:.4f}", flush=True)
    print(f"width68_median={global_metrics['width68_median']:.4f}", flush=True)
    print(f"corrPearson={global_metrics['corrPearson']:.4f}", flush=True)
    print(f"corrSpearman={global_metrics['corrSpearman']:.4f}", flush=True)
    print(f"highRA_F1={highra_metrics['F1']:.4f}", flush=True)


if __name__ == "__main__":
    print("Loading v2 models and data.", flush=True)
    models = pr.load_parallel_scattering_models()
    cov_model = pr.load_covariance_model()
    data = pr.load_analysis_dataset(data_path, list(pr.ANALYSIS_VARIABLES))
    s = pr.preprocess_minirf_data(data)
    base_options = pr.build_default_pipeline_options(models, cov_model)
    blocks = hs.make_spatial_blocks(s["ra"], hs.tile_rows, hs.tile_cols, min_valid_pixels=hs.min_valid_pixels)
    fixed_values = hs.build_fixed_candidate_values()
    baseline_candidate = hs.build_baseline_candidate(base_options, fixed_values)
    firstpass_cache: dict[tuple[Any, ...], tuple[Array, Array]] = {}

    print("Evaluating default v2 candidate.", flush=True)
    baseline_result = hs.evaluate_candidate_on_preprocessed(
        s,
        baseline_candidate,
        base_options,
        blocks,
        firstpass_cache,
        n_workers=n_workers,
        objective=hs.objective,
        roughness_weight=hs.roughness_weight,
        coverage_weight=hs.coverage_weight,
        coverage_target=hs.coverage_target,
        valid_fraction_weight=hs.valid_fraction_weight,
    )
    baseline_nan_count = int(baseline_result["summary"]["pred_nan_count"])

    print("Evaluating requested best-fit candidate.", flush=True)
    best_result = hs.evaluate_candidate_on_preprocessed(
        s,
        best_fit_candidate,
        base_options,
        blocks,
        firstpass_cache,
        n_workers=n_workers,
        objective=hs.objective,
        roughness_weight=hs.roughness_weight,
        coverage_weight=hs.coverage_weight,
        coverage_target=hs.coverage_target,
        valid_fraction_weight=hs.valid_fraction_weight,
        baseline_nan_count=baseline_nan_count,
        max_nan_growth_fraction=hs.max_nan_growth_fraction,
    )

    print_search_summary("default v2 candidate", baseline_result)
    print_search_summary("tested best-fit candidate", best_result)
    print_candidate_delta(baseline_result, best_result)

    pipeline_options = hs.apply_candidate_to_pipeline_options(base_options, best_fit_candidate)
    firstpass_key = hs.make_firstpass_cache_key(pipeline_options["firstpass"])
    ra_pred_map, flag_map = firstpass_cache[firstpass_key]

    print("\nRunning the full-map prediction for plots and detailed metrics.", flush=True)
    prediction = pr.predict_ra_from_preprocessed_with_firstpass(
        s,
        ra_pred_map,
        flag_map,
        pipeline_options["secondpass"],
        n_workers=n_workers,
    )

    eval_options = dict(pipeline_options["eval"])
    eval_options["RA_low"] = prediction["RApredMaps"]["low"]
    eval_options["RA_high"] = prediction["RApredMaps"]["high"]
    metrics = pr.evaluate_ra_prediction(s["ra"], prediction["RApred"], eval_options)
    print_global_metrics(metrics)

    if pr.reference_path.exists():
        reference = np.load(pr.reference_path)
        reference_mean_diff = float(np.nanmean(reference - prediction["RApred"]))
        print(f"\nmean difference to saved v2 reference={reference_mean_diff:.6f}", flush=True)

    map_plot_path = pr.make_ra_map_sigma_plot(prediction["RApred"], prediction.get("sigma_map"), map_plot_output_path)
    hist_plot_path = pr.make_ra_histogram_plot(
        s["ra"],
        prediction["RApredMaps"]["med"],
        prediction["RApredMaps"]["mean"],
        hist_plot_output_path,
    )

    print(f"\nsaved map and sigma plot to {map_plot_path}", flush=True)
    print(f"saved histogram plot to {hist_plot_path}", flush=True)
