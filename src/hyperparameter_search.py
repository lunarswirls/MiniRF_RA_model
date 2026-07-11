#!/usr/bin/env python
# -*- coding: utf-8 -*-
import csv
import itertools
from pathlib import Path
from typing import Any
import numpy as np
from scipy.optimize import direct, minimize
import parallel_ra_prediction as pr


Array = np.ndarray

# path to the input dataset
data_path = Path(__file__).resolve().parents[1] / "data/GiordanoBruno_analysis.csv"

# number of worker threads for the prediction pipeline
n_workers = 20

# number of row-wise spatial blocks
tile_rows = 4

# number of column-wise spatial blocks
tile_cols = 4

# minimum finite reference pixels required per block
min_valid_pixels = 1000

# scalar metric used to rank candidates
objective = "balancedRMSE"

# penalty weight for roughness mismatch
roughness_weight = 0.0

# penalty weight for interval coverage mismatch
coverage_weight = 0.0

# penalty weight for dropped predictions on the finite reference mask
valid_fraction_weight = 1.0

# target empirical coverage for the 68 percent interval
coverage_target = 0.68

# number of ranked candidates to print
top_k = 5

# optional csv path for ranked results
output_csv = Path(__file__).resolve().parents[1] / "output/GiordanoBruno_rankedresults.csv"

# manual tuning notes favored searches centered on s = 1.5 for the mixed CPR+Green workflow
s_values = [1.0, 1.25, 1.5, 1.75, 2.0]

# manual tuning tested bandwidths near 0.5 and showed balanced-RMSE gains at somewhat broader priors
sigma_spatial_values = [0.4, 0.5, 0.6, 0.7]

# tune locally around the default Student-t degrees of freedom
nu_values = [2.0, 3.0, 4.0]

# hold non-manual-tuned parameters at the translated defaults
sigma_floor_values = [0.02]

# hold non-manual-tuned parameters at the translated defaults
window_size_values = [3]

# tune locally around the default first-pass prior transition
ra0_values = [0.95, 1.0, 1.05]

# candidate values for the second-pass prior-center mode
spatial_prior_modes = ["direct"]  # matlab default direct

# hold non-manual-tuned parameters at the translated defaults
second_pass_window_size_values = [5]

# eps tradeoff between more global and more local direct search
direct_eps = 1e-4

# optional cap on direct objective evaluations
direct_maxfun = None

# maximum number of direct iterations
direct_maxiter = 100

# whether direct should prefer locally biased subdivision
direct_locally_biased = True

# maximum number of powell iterations during local polish
powell_maxiter = 100

# relative step tolerance for powell polish
powell_xtol = 1e-3

# relative objective tolerance for powell polish
powell_ftol = 1e-3


def build_candidate_grid(search_space: dict[str, list[Any]]) -> list[dict[str, Any]]:
    """Expand a dict of parameter value lists into a full candidate grid"""
    names = list(search_space)
    value_lists = [search_space[name] for name in names]
    return [dict(zip(names, combo)) for combo in itertools.product(*value_lists)]


def make_spatial_blocks(ra_true: Array, tile_rows: int, tile_cols: int, min_valid_pixels: int = 1000) -> list[dict[str, Any]]:
    """Split the reference map into valid spatial tiles for blocked scoring"""
    ra_true = np.asarray(ra_true, dtype=float)
    nrows, ncols = ra_true.shape
    row_edges = np.linspace(0, nrows, tile_rows + 1, dtype=int)
    col_edges = np.linspace(0, ncols, tile_cols + 1, dtype=int)

    blocks: list[dict[str, Any]] = []
    for i in range(tile_rows):
        for j in range(tile_cols):
            r0, r1 = row_edges[i], row_edges[i + 1]
            c0, c1 = col_edges[j], col_edges[j + 1]
            block = ra_true[r0:r1, c0:c1]
            n_valid = int(np.sum(np.isfinite(block)))
            if n_valid < min_valid_pixels:
                continue
            blocks.append(
                {
                    "row_block": i + 1,
                    "col_block": j + 1,
                    "r0": r0,
                    "r1": r1,
                    "c0": c0,
                    "c1": c1,
                    "n_valid": n_valid,
                }
            )
    return blocks


def apply_candidate_to_pipeline_options(base_options: dict[str, dict[str, Any]], candidate: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Overlay one hyperparameter candidate onto the default pipeline options"""
    firstpass = dict(base_options["firstpass"])
    secondpass = dict(base_options["secondpass"])
    eval_options = dict(base_options["eval"])

    if "s" in candidate:
        firstpass["s"] = float(candidate["s"])
        secondpass["s"] = float(candidate["s"])

    if "sigmaFloor" in candidate:
        firstpass["sigmaFloor"] = float(candidate["sigmaFloor"])

    if "sigmaSpatial" in candidate:
        secondpass["sigmaSpatial"] = float(candidate["sigmaSpatial"])

    if "nu" in candidate:
        secondpass["nu"] = float(candidate["nu"])

    if "RA0" in candidate:
        firstpass["RA0"] = float(candidate["RA0"])

    if "windowSize" in candidate:
        window_size = int(candidate["windowSize"])
        firstpass["halfWin"] = window_size // 2
        firstpass["sigmaDist"] = window_size / 3.0

    if "firstPassWindowSize" in candidate:
        window_size = int(candidate["firstPassWindowSize"])
        firstpass["halfWin"] = window_size // 2
        firstpass["sigmaDist"] = window_size / 3.0

    if "spatialPriorMode" in candidate:
        secondpass["spatialPriorMode"] = str(candidate["spatialPriorMode"])

    if "secondPassWindowSize" in candidate:
        secondpass["windowSize"] = int(candidate["secondPassWindowSize"])

    if "smoothWindow" in candidate:
        eval_options["smoothWindow"] = int(candidate["smoothWindow"])

    return {"firstpass": firstpass, "secondpass": secondpass, "eval": eval_options}


def make_firstpass_cache_key(firstpass_options: dict[str, Any]) -> tuple[Any, ...]:
    """Build a stable cache key for a first-pass parameter combination."""
    ra_grid = np.asarray(firstpass_options["RA_grid"], dtype=float).reshape(-1)
    return (
        float(firstpass_options["s"]),
        float(firstpass_options["sigmaFloor"]),
        float(firstpass_options["RA0"]),
        int(firstpass_options["halfWin"]),
        float(firstpass_options["sigmaDist"]),
        ra_grid.shape,
        ra_grid.tobytes(),
    )


def make_candidate_result_cache_key(candidate: dict[str, Any]) -> tuple[Any, ...]:
    """Build a stable cache key for one full hyperparameter candidate."""
    return (
        float(candidate["s"]),
        float(candidate["sigmaSpatial"]),
        float(candidate["nu"]),
        float(candidate["sigmaFloor"]),
        int(candidate["windowSize"]),
        float(candidate["RA0"]),
        str(candidate["spatialPriorMode"]),
        int(candidate["secondPassWindowSize"]),
    )


def build_optimization_parameters() -> list[tuple[str, float, float]]:
    """Return the ordered continuous parameter bounds for direct and Powell."""
    return [
        ("s", float(min(s_values)), float(max(s_values))),
        ("sigmaSpatial", float(min(sigma_spatial_values)), float(max(sigma_spatial_values))),
        ("nu", float(min(nu_values)), float(max(nu_values))),
        ("RA0", float(min(ra0_values)), float(max(ra0_values))),
    ]


def build_fixed_candidate_values() -> dict[str, Any]:
    """Return the fixed non-optimized hyperparameters used by the optimizer."""
    return {
        "sigmaFloor": float(sigma_floor_values[0]),
        "windowSize": int(window_size_values[0]),
        "spatialPriorMode": str(spatial_prior_modes[0]),
        "secondPassWindowSize": int(second_pass_window_size_values[0]),
    }


def candidate_from_vector(x: Array, parameter_specs: list[tuple[str, float, float]], fixed_values: dict[str, Any]) -> dict[str, Any]:
    """Convert an ordered parameter vector into one candidate dict."""
    candidate = dict(fixed_values)
    x = np.asarray(x, dtype=float).reshape(-1)
    for value, (name, _, _) in zip(x, parameter_specs):
        candidate[name] = float(value)
    return candidate


def clone_search_result(result: dict[str, Any]) -> dict[str, Any]:
    """Shallow-copy one search result for stage-specific reporting."""
    cloned = dict(result)
    cloned["candidate"] = dict(result["candidate"])
    cloned["summary"] = dict(result["summary"])
    return cloned


def score_metrics(metrics: dict[str, Any], objective: str = "balancedRMSE", roughness_weight: float = 0.0,
                  coverage_weight: float = 0.0, coverage_target: float = 0.68) -> float:
    """Reduce evaluation metrics to one scalar score for ranking candidates"""
    objective_key = objective.lower()
    if objective_key == "balancedrmse":
        score = float(metrics["global"]["balancedRMSE"])
    elif objective_key == "rmse":
        score = float(metrics["global"]["RMSE"])
    elif objective_key == "balancedmae":
        score = float(metrics["global"]["balancedMAE"])
    elif objective_key == "mae":
        score = float(metrics["global"]["MAE_mean"])
    else:
        raise ValueError(f"Unknown objective {objective!r}.")

    if roughness_weight != 0.0:
        roughness_ratio = float(metrics["spatial"]["roughness_ratio_pred_true"])
        if np.isfinite(roughness_ratio):
            score += roughness_weight * abs(roughness_ratio - 1.0)

    if coverage_weight != 0.0:
        coverage = float(metrics["global"]["coverage68"])
        if np.isfinite(coverage):
            score += coverage_weight * abs(coverage - coverage_target)

    return score


def apply_valid_fraction_penalty(score: float, pred_valid_fraction: float, valid_fraction_weight: float) -> float:
    """Add a penalty for predictions dropped from the finite reference mask."""
    if valid_fraction_weight != 0.0 and np.isfinite(pred_valid_fraction):
        score += valid_fraction_weight * (1.0 - pred_valid_fraction)
    return score


def evaluate_prediction_on_blocks(ra_true: Array, ra_pred: Array, ra_low: Array | None, ra_high: Array | None,
                                  blocks: list[dict[str, Any]], eval_options: dict[str, Any],
                                  objective: str, roughness_weight: float,
                                  coverage_weight: float, coverage_target: float) -> list[dict[str, Any]]:
    """Evaluate one prediction on each blocked fold and score every fold"""
    fold_results: list[dict[str, Any]] = []

    for block in blocks:
        r0 = block["r0"]
        r1 = block["r1"]
        c0 = block["c0"]
        c1 = block["c1"]

        block_eval_options = dict(eval_options)
        block_eval_options["report"] = False
        block_eval_options["mapMode"] = True
        if ra_low is not None and ra_high is not None:
            block_eval_options["RA_low"] = ra_low[r0:r1, c0:c1]
            block_eval_options["RA_high"] = ra_high[r0:r1, c0:c1]
        else:
            block_eval_options["RA_low"] = None
            block_eval_options["RA_high"] = None

        metrics = pr.evaluate_ra_prediction(
            ra_true[r0:r1, c0:c1],
            ra_pred[r0:r1, c0:c1],
            block_eval_options,
        )
        score = score_metrics(
            metrics,
            objective=objective,
            roughness_weight=roughness_weight,
            coverage_weight=coverage_weight,
            coverage_target=coverage_target,
        )
        fold_results.append(
            {
                "block": block,
                "metrics": metrics,
                "score": score,
            }
        )

    return fold_results


def summarize_fold_results(fold_results: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate blocked-fold scores into a compact summary"""
    scores = np.array([fold["score"] for fold in fold_results], dtype=float)
    balanced_rmse = np.array([fold["metrics"]["global"]["balancedRMSE"] for fold in fold_results], dtype=float)
    rmse = np.array([fold["metrics"]["global"]["RMSE"] for fold in fold_results], dtype=float)
    coverage = np.array([fold["metrics"]["global"]["coverage68"] for fold in fold_results], dtype=float)
    roughness = np.array([fold["metrics"]["spatial"]["roughness_ratio_pred_true"] for fold in fold_results], dtype=float)

    return {
        "mean_score": float(np.nanmean(scores)),
        "std_score": float(np.nanstd(scores)),
        "mean_balancedRMSE": float(np.nanmean(balanced_rmse)),
        "mean_RMSE": float(np.nanmean(rmse)),
        "mean_coverage68": float(np.nanmean(coverage)),
        "mean_roughness_ratio": float(np.nanmean(roughness)),
        "n_blocks": int(len(fold_results)),
    }


def evaluate_candidate_on_preprocessed(s: dict[str, Array], candidate: dict[str, Any], base_options: dict[str, dict[str, Any]],
                                       blocks: list[dict[str, Any]], firstpass_cache: dict[tuple[Any, ...], Array],
                                       n_workers: int | None = None, objective: str = "balancedRMSE",
                                       roughness_weight: float = 0.0, coverage_weight: float = 0.0,
                                       coverage_target: float = 0.68, valid_fraction_weight: float = 1.0,
                                       candidate_index: int | None = None) -> dict[str, Any]:
    """Evaluate one hyperparameter candidate on the preprocessed dataset."""
    pipeline_options = apply_candidate_to_pipeline_options(base_options, candidate)
    firstpass_key = make_firstpass_cache_key(pipeline_options["firstpass"])
    if firstpass_key in firstpass_cache:
        ra_pred_map = firstpass_cache[firstpass_key]
    else:
        ra_pred_map = pr.invert_ra_2rad_first_pass_spatial(
            s["Rad1"],
            s["Rad2"],
            s["RAMap"],
            s["IncidenceMap"],
            pipeline_options["firstpass"],
            n_workers=n_workers,
        )
        firstpass_cache[firstpass_key] = ra_pred_map

    prediction = pr.predict_ra_from_preprocessed_with_firstpass(
        s,
        ra_pred_map,
        pipeline_options["secondpass"],
        n_workers=n_workers,
    )

    eval_options = dict(pipeline_options["eval"])
    eval_options["report"] = False
    eval_options["RA_low"] = prediction["RApredMaps"]["low"]
    eval_options["RA_high"] = prediction["RApredMaps"]["high"]

    full_metrics = pr.evaluate_ra_prediction(s["ra"], prediction["RApred"], eval_options)
    fold_results = evaluate_prediction_on_blocks(
        s["ra"],
        prediction["RApred"],
        prediction["RApredMaps"]["low"],
        prediction["RApredMaps"]["high"],
        blocks,
        pipeline_options["eval"],
        objective=objective,
        roughness_weight=roughness_weight,
        coverage_weight=coverage_weight,
        coverage_target=coverage_target,
    )
    summary = summarize_fold_results(fold_results)
    pred_valid_fraction = float(full_metrics["global"]["pred_valid_fraction"])
    summary["mean_score_unpenalized"] = float(summary["mean_score"])
    summary["pred_valid_fraction"] = pred_valid_fraction
    summary["pred_valid_count"] = int(full_metrics["global"]["pred_valid_count"])
    summary["ref_valid_count"] = int(full_metrics["global"]["ref_valid_count"])
    summary["valid_fraction_penalty"] = (
        float(valid_fraction_weight * (1.0 - pred_valid_fraction))
        if valid_fraction_weight != 0.0 and np.isfinite(pred_valid_fraction)
        else 0.0
    )
    summary["mean_score"] = apply_valid_fraction_penalty(
        float(summary["mean_score"]),
        pred_valid_fraction,
        valid_fraction_weight,
    )
    return {
        "candidate_index": candidate_index,
        "candidate": dict(candidate),
        "full_metrics": full_metrics,
        "fold_results": fold_results,
        "summary": summary,
    }


def search_hyperparameters_on_preprocessed(s: dict[str, Array], candidates: list[dict[str, Any]], n_workers: int | None = None,
                                           tile_rows: int = 4, tile_cols: int = 4, min_valid_pixels: int = 1000,
                                           objective: str = "balancedRMSE", roughness_weight: float = 0.0,
                                           coverage_weight: float = 0.0, coverage_target: float = 0.68,
                                           valid_fraction_weight: float = 1.0) -> dict[str, Any]:
    """Search candidate settings on an already loaded and preprocessed dataset"""
    base_options = pr.build_default_pipeline_options()
    blocks = make_spatial_blocks(s["ra"], tile_rows=tile_rows, tile_cols=tile_cols, min_valid_pixels=min_valid_pixels)
    if not blocks:
        raise ValueError("No valid spatial blocks were created. Lower min_valid_pixels or change tile settings.")

    results: list[dict[str, Any]] = []
    firstpass_cache: dict[tuple[Any, ...], Array] = {}

    for idx, candidate in enumerate(candidates, start=1):
        result = evaluate_candidate_on_preprocessed(
            s,
            candidate,
            base_options,
            blocks,
            firstpass_cache,
            n_workers=n_workers,
            objective=objective,
            roughness_weight=roughness_weight,
            coverage_weight=coverage_weight,
            coverage_target=coverage_target,
            valid_fraction_weight=valid_fraction_weight,
            candidate_index=idx,
        )
        results.append(result)

        print(
            f"[{idx}/{len(candidates)}] score={result['summary']['mean_score']:.4f} "
            f"balancedRMSE={result['summary']['mean_balancedRMSE']:.4f} "
            f"predValid={result['summary']['pred_valid_fraction']:.4f} candidate={candidate}",
            flush=True,
        )

    results.sort(key=lambda item: item["summary"]["mean_score"])
    return {"blocks": blocks, "results": results}


def search_hyperparameters(data_path: str | Path, candidates: list[dict[str, Any]], n_workers: int | None = None,
                           tile_rows: int = 4, tile_cols: int = 4, min_valid_pixels: int = 1000,
                           objective: str = "balancedRMSE", roughness_weight: float = 0.0,
                           coverage_weight: float = 0.0, coverage_target: float = 0.68,
                           valid_fraction_weight: float = 1.0) -> dict[str, Any]:
    """Load a dataset, preprocess it, and run the blocked hyperparameter search"""
    data = pr.load_analysis_dataset(data_path, pr.ANALYSIS_VARIABLES)
    s = pr.preprocess_minirf_data(data)
    return search_hyperparameters_on_preprocessed(
        s,
        candidates,
        n_workers=n_workers,
        tile_rows=tile_rows,
        tile_cols=tile_cols,
        min_valid_pixels=min_valid_pixels,
        objective=objective,
        roughness_weight=roughness_weight,
        coverage_weight=coverage_weight,
        coverage_target=coverage_target,
        valid_fraction_weight=valid_fraction_weight,
    )


def optimize_hyperparameters_on_preprocessed(s: dict[str, Array], n_workers: int | None = None,
                                             tile_rows: int = 4, tile_cols: int = 4, min_valid_pixels: int = 1000,
                                             objective: str = "balancedRMSE", roughness_weight: float = 0.0,
                                             coverage_weight: float = 0.0, coverage_target: float = 0.68,
                                             valid_fraction_weight: float = 1.0) -> dict[str, Any]:
    """Optimize the main continuous hyperparameters with direct plus Powell polish."""
    base_options = pr.build_default_pipeline_options()
    blocks = make_spatial_blocks(s["ra"], tile_rows=tile_rows, tile_cols=tile_cols, min_valid_pixels=min_valid_pixels)
    if not blocks:
        raise ValueError("No valid spatial blocks were created. Lower min_valid_pixels or change tile settings.")

    parameter_specs = build_optimization_parameters()
    bounds = [(lower, upper) for _, lower, upper in parameter_specs]
    fixed_values = build_fixed_candidate_values()
    firstpass_cache: dict[tuple[Any, ...], Array] = {}
    result_cache: dict[tuple[Any, ...], dict[str, Any]] = {}
    eval_state = {"count": 0, "best_score": float("inf")}

    def get_result_for_vector(x: Array) -> tuple[dict[str, Any], dict[str, Any]]:
        candidate = candidate_from_vector(x, parameter_specs, fixed_values)
        candidate_key = make_candidate_result_cache_key(candidate)
        if candidate_key not in result_cache:
            result_cache[candidate_key] = evaluate_candidate_on_preprocessed(
                s,
                candidate,
                base_options,
                blocks,
                firstpass_cache,
                n_workers=n_workers,
                objective=objective,
                roughness_weight=roughness_weight,
                coverage_weight=coverage_weight,
                coverage_target=coverage_target,
                valid_fraction_weight=valid_fraction_weight,
            )
        return candidate, result_cache[candidate_key]

    def objective_fun(x: Array) -> float:
        eval_state["count"] += 1
        candidate, result = get_result_for_vector(x)
        score = float(result["summary"]["mean_score"])
        if score < eval_state["best_score"]:
            eval_state["best_score"] = score
            print(
                f"[eval {eval_state['count']}] new best score={score:.4f} "
                f"balancedRMSE={result['summary']['mean_balancedRMSE']:.4f} "
                f"predValid={result['summary']['pred_valid_fraction']:.4f} candidate={candidate}",
                flush=True,
            )
        return score

    direct_result_raw = direct(
        objective_fun,
        bounds=bounds,
        eps=direct_eps,
        maxfun=direct_maxfun,
        maxiter=direct_maxiter,
        locally_biased=direct_locally_biased,
    )
    direct_candidate, direct_result_base = get_result_for_vector(direct_result_raw.x)
    direct_result = clone_search_result(direct_result_base)
    direct_result["candidate_index"] = 1
    direct_result["search_stage"] = "direct"
    direct_result["optimizer_success"] = bool(getattr(direct_result_raw, "success", True))
    direct_result["optimizer_message"] = str(getattr(direct_result_raw, "message", ""))
    direct_result["optimizer_nfev"] = int(getattr(direct_result_raw, "nfev", 0))
    direct_result["candidate"] = dict(direct_candidate)

    powell_options = {
        "maxiter": powell_maxiter,
        "xtol": powell_xtol,
        "ftol": powell_ftol,
        "disp": False,
    }
    powell_result_raw = minimize(
        objective_fun,
        np.asarray(direct_result_raw.x, dtype=float),
        method="Powell",
        bounds=bounds,
        options=powell_options,
    )
    powell_candidate, powell_result_base = get_result_for_vector(powell_result_raw.x)
    powell_result = clone_search_result(powell_result_base)
    powell_result["candidate_index"] = 2
    powell_result["search_stage"] = "powell"
    powell_result["optimizer_success"] = bool(getattr(powell_result_raw, "success", True))
    powell_result["optimizer_message"] = str(getattr(powell_result_raw, "message", ""))
    powell_result["optimizer_nfev"] = int(getattr(powell_result_raw, "nfev", 0))
    powell_result["candidate"] = dict(powell_candidate)

    results = [direct_result, powell_result]
    results.sort(key=lambda item: item["summary"]["mean_score"])
    return {"blocks": blocks, "results": results, "evaluation_count": eval_state["count"]}


def optimize_hyperparameters(data_path: str | Path, n_workers: int | None = None,
                             tile_rows: int = 4, tile_cols: int = 4, min_valid_pixels: int = 1000,
                             objective: str = "balancedRMSE", roughness_weight: float = 0.0,
                             coverage_weight: float = 0.0, coverage_target: float = 0.68,
                             valid_fraction_weight: float = 1.0) -> dict[str, Any]:
    """Load a dataset, preprocess it, and optimize the main continuous hyperparameters."""
    data = pr.load_analysis_dataset(data_path, pr.ANALYSIS_VARIABLES)
    s = pr.preprocess_minirf_data(data)
    return optimize_hyperparameters_on_preprocessed(
        s,
        n_workers=n_workers,
        tile_rows=tile_rows,
        tile_cols=tile_cols,
        min_valid_pixels=min_valid_pixels,
        objective=objective,
        roughness_weight=roughness_weight,
        coverage_weight=coverage_weight,
        coverage_target=coverage_target,
        valid_fraction_weight=valid_fraction_weight,
    )


def write_search_results_csv(search_output: dict[str, Any], csv_path: str | Path) -> Path:
    """Write the ranked search summary to a CSV file"""
    csv_path = Path(csv_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "rank",
                "search_stage",
                "candidate_index",
                "mean_score",
                "mean_score_unpenalized",
                "std_score",
                "mean_balancedRMSE",
                "mean_RMSE",
                "pred_valid_fraction",
                "pred_valid_count",
                "ref_valid_count",
                "valid_fraction_penalty",
                "mean_coverage68",
                "mean_roughness_ratio",
                "n_blocks",
                "optimizer_success",
                "optimizer_nfev",
                "optimizer_message",
                "candidate",
            ]
        )
        for rank, result in enumerate(search_output["results"], start=1):
            summary = result["summary"]
            writer.writerow(
                [
                    rank,
                    result.get("search_stage", ""),
                    result["candidate_index"],
                    summary["mean_score"],
                    summary.get("mean_score_unpenalized", ""),
                    summary["std_score"],
                    summary["mean_balancedRMSE"],
                    summary["mean_RMSE"],
                    summary.get("pred_valid_fraction", ""),
                    summary.get("pred_valid_count", ""),
                    summary.get("ref_valid_count", ""),
                    summary.get("valid_fraction_penalty", ""),
                    summary["mean_coverage68"],
                    summary["mean_roughness_ratio"],
                    summary["n_blocks"],
                    result.get("optimizer_success", ""),
                    result.get("optimizer_nfev", ""),
                    result.get("optimizer_message", ""),
                    repr(result["candidate"]),
                ]
            )

    return csv_path


def main() -> None:
    parameter_specs = build_optimization_parameters()
    bounds_text = ", ".join([f"{name}=[{lower}, {upper}]" for name, lower, upper in parameter_specs])
    print(f"Running direct global search with Powell polish on {bounds_text}.", flush=True)

    search_output = optimize_hyperparameters(
        data_path,
        n_workers=n_workers,
        tile_rows=tile_rows,
        tile_cols=tile_cols,
        min_valid_pixels=min_valid_pixels,
        objective=objective,
        roughness_weight=roughness_weight,
        coverage_weight=coverage_weight,
        coverage_target=coverage_target,
        valid_fraction_weight=valid_fraction_weight,
    )

    print("\nTop candidates:", flush=True)
    for rank, result in enumerate(search_output["results"][:top_k], start=1):
        summary = result["summary"]
        print(
            f"{rank}. stage={result.get('search_stage', 'search')} "
            f"score={summary['mean_score']:.4f} "
            f"balancedRMSE={summary['mean_balancedRMSE']:.4f} "
            f"predValid={summary['pred_valid_fraction']:.4f} "
            f"RMSE={summary['mean_RMSE']:.4f} candidate={result['candidate']}",
            flush=True,
        )

    if output_csv:
        out_path = write_search_results_csv(search_output, output_csv)
        print(f"\nWrote ranked results to {out_path}", flush=True)


if __name__ == "__main__":
    main()
