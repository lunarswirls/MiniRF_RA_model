#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Hyperparameter search wrapper for the v2 RA prediction pipeline
"""

import csv
from pathlib import Path
from typing import Any
import numpy as np
from scipy.optimize import direct, minimize
import parallel_ra_prediction_v2 as pr


Array = np.ndarray

# path to the input dataset
data_path = Path(__file__).resolve().parents[1] / "data/GiordanoBruno.mat"

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

# hard rejection threshold for growth in prediction nans relative to the baseline run
max_nan_growth_fraction = 0.10

# large finite score used for hard-rejected candidates
hard_reject_score = 1e12

# target empirical coverage for the 68 percent interval
coverage_target = 0.68

# number of ranked candidates to print
top_k = 5

# optional csv path for ranked results
output_csv = Path(__file__).resolve().parents[1] / "output/GiordanoBruno_v2_rankedresults.csv"

# tune around the matlab v2 defaults
s_values = [1.0, 1.25, 1.5, 1.75, 2.0]
sigma_spatial_values = [0.5, 0.75, 1.0, 1.25]
ra0_values = [0.95, 1.0, 1.05]

# fixed v2 defaults
sigma_floor_values = [0.02]
window_size_values = [3]
second_pass_window_size_values = [5]
spatial_prior_modes = ["direct"]

# direct settings
direct_eps = 1e-4
direct_maxfun = None
direct_maxiter = 100
direct_locally_biased = True

# powell settings
powell_maxiter = 100
powell_xtol = 1e-3
powell_ftol = 1e-3


def make_spatial_blocks(ra_true: Array, tile_rows: int, tile_cols: int, min_valid_pixels: int = 1000) -> list[dict[str, Any]]:
    """split the reference ra map into evaluation blocks with enough valid pixels"""
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


def apply_candidate_to_pipeline_options(base_options: dict[str, dict[str, Any]],
                                        candidate: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """inject one candidate vector into the shared first and second pass options"""
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

    if "RA0" in candidate:
        firstpass["RA0"] = float(candidate["RA0"])

    if "windowSize" in candidate:
        window_size = int(candidate["windowSize"])
        firstpass["halfWin"] = window_size // 2
        firstpass["sigmaDist"] = window_size / 3.0

    if "spatialPriorMode" in candidate:
        secondpass["spatialPriorMode"] = str(candidate["spatialPriorMode"])

    if "secondPassWindowSize" in candidate:
        secondpass["windowSize"] = int(candidate["secondPassWindowSize"])

    return {"firstpass": firstpass, "secondpass": secondpass, "eval": eval_options}


def make_firstpass_cache_key(firstpass_options: dict[str, Any]) -> tuple[Any, ...]:
    """build a hashable key for reusing first-pass spatial predictions"""
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
    """build a cache key for the full candidate evaluation result"""
    return (
        float(candidate["s"]),
        float(candidate["sigmaSpatial"]),
        float(candidate["RA0"]),
        float(candidate["sigmaFloor"]),
        int(candidate["windowSize"]),
        str(candidate["spatialPriorMode"]),
        int(candidate["secondPassWindowSize"]),
    )


def build_optimization_parameters() -> list[tuple[str, float, float]]:
    """return the continuous parameters searched by direct and powell"""
    return [
        ("s", float(min(s_values)), float(max(s_values))),
        ("sigmaSpatial", float(min(sigma_spatial_values)), float(max(sigma_spatial_values))),
        ("RA0", float(min(ra0_values)), float(max(ra0_values))),
    ]


def build_fixed_candidate_values() -> dict[str, Any]:
    """return the fixed v2 settings that are not part of the search"""
    return {
        "sigmaFloor": float(sigma_floor_values[0]),
        "windowSize": int(window_size_values[0]),
        "spatialPriorMode": str(spatial_prior_modes[0]),
        "secondPassWindowSize": int(second_pass_window_size_values[0]),
    }


def build_baseline_candidate(base_options: dict[str, dict[str, Any]], fixed_values: dict[str, Any]) -> dict[str, Any]:
    """build the default v2 candidate used as the nan-growth baseline"""
    candidate = dict(fixed_values)
    candidate["s"] = float(base_options["firstpass"]["s"])
    candidate["sigmaSpatial"] = float(base_options["secondpass"]["sigmaSpatial"])
    candidate["RA0"] = float(base_options["firstpass"]["RA0"])
    return candidate


def candidate_from_vector(x: Array, parameter_specs: list[tuple[str, float, float]],
                          fixed_values: dict[str, Any]) -> dict[str, Any]:
    """convert an optimizer vector back into the candidate dict format"""
    candidate = dict(fixed_values)
    x = np.asarray(x, dtype=float).reshape(-1)
    for value, (name, _, _) in zip(x, parameter_specs):
        candidate[name] = float(value)
    return candidate


def clone_search_result(result: dict[str, Any]) -> dict[str, Any]:
    """copy the nested pieces we mutate when ranking search results"""
    cloned = dict(result)
    cloned["candidate"] = dict(result["candidate"])
    cloned["summary"] = dict(result["summary"])
    return cloned


def score_metrics(metrics: dict[str, Any], objective: str = "balancedRMSE", roughness_weight: float = 0.0,
                  coverage_weight: float = 0.0, coverage_target: float = 0.68) -> float:
    """collapse the evaluation metrics into one scalar optimization score"""
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
        coverage68 = float(metrics["global"]["coverage68"])
        if np.isfinite(coverage68):
            score += coverage_weight * abs(coverage68 - coverage_target)

    return score


def apply_valid_fraction_penalty(score: float, pred_valid_fraction: float,
                                 valid_fraction_weight: float) -> float:
    """penalize candidates that drop predictions on the valid truth mask"""
    if valid_fraction_weight != 0.0 and np.isfinite(pred_valid_fraction):
        score += valid_fraction_weight * (1.0 - pred_valid_fraction)
    return score


def evaluate_candidate_on_preprocessed(s: dict[str, Array], candidate: dict[str, Any],
                                       base_options: dict[str, dict[str, Any]], blocks: list[dict[str, Any]],
                                       firstpass_cache: dict[tuple[Any, ...], tuple[Array, Array]],
                                       n_workers: int | None = None, objective: str = "balancedRMSE",
                                       roughness_weight: float = 0.0, coverage_weight: float = 0.0,
                                       coverage_target: float = 0.68, valid_fraction_weight: float = 1.0,
                                       baseline_nan_count: int | None = None,
                                       max_nan_growth_fraction: float | None = None) -> dict[str, Any]:
    """run one candidate end to end and score it over the spatial blocks"""
    pipeline_options = apply_candidate_to_pipeline_options(base_options, candidate)
    firstpass_options = pipeline_options["firstpass"]
    secondpass_options = pipeline_options["secondpass"]
    eval_options = dict(pipeline_options["eval"])
    eval_options["report"] = False

    firstpass_key = make_firstpass_cache_key(firstpass_options)
    # reuse the expensive first pass whenever only second-pass settings changed
    if firstpass_key not in firstpass_cache:
        firstpass_cache[firstpass_key] = pr.invert_ra_2rad_first_pass_spatial(
            s["Rad1"],
            s["Rad2"],
            s["RAMap"],
            s["IncidenceMap"],
            firstpass_options,
            n_workers=n_workers,
        )
    ra_pred_map, flag_map = firstpass_cache[firstpass_key]

    prediction = pr.predict_ra_from_preprocessed_with_firstpass(
        s,
        ra_pred_map,
        flag_map,
        secondpass_options,
        n_workers=n_workers,
    )

    eval_options["RA_low"] = prediction["RApredMaps"]["low"]
    eval_options["RA_high"] = prediction["RApredMaps"]["high"]

    global_metrics = pr.evaluate_ra_prediction(s["ra"], prediction["RApred"], eval_options)
    pred_valid_fraction = float(global_metrics["global"]["pred_valid_fraction"])
    pred_nan_count = int(global_metrics["global"]["ref_valid_count"] - global_metrics["global"]["pred_valid_count"])
    valid_fraction_penalty = (
        float(valid_fraction_weight * (1.0 - pred_valid_fraction))
        if valid_fraction_weight != 0.0 and np.isfinite(pred_valid_fraction)
        else 0.0
    )

    block_scores: list[float] = []
    block_balanced_rmse: list[float] = []
    for block in blocks:
        r0 = block["r0"]
        r1 = block["r1"]
        c0 = block["c0"]
        c1 = block["c1"]

        block_eval_options = dict(eval_options)
        block_eval_options["RA_low"] = prediction["RApredMaps"]["low"][r0:r1, c0:c1]
        block_eval_options["RA_high"] = prediction["RApredMaps"]["high"][r0:r1, c0:c1]

        metrics = pr.evaluate_ra_prediction(
            s["ra"][r0:r1, c0:c1],
            prediction["RApred"][r0:r1, c0:c1],
            block_eval_options,
        )
        block_scores.append(
            score_metrics(
                metrics,
                objective=objective,
                roughness_weight=roughness_weight,
                coverage_weight=coverage_weight,
                coverage_target=coverage_target,
            )
        )
        block_balanced_rmse.append(float(metrics["global"]["balancedRMSE"]))

    mean_score_unpenalized = float(np.nanmean(block_scores)) if block_scores else float("nan")
    mean_score = apply_valid_fraction_penalty(mean_score_unpenalized, pred_valid_fraction, valid_fraction_weight)

    hard_rejected_nan_growth = False
    max_allowed_nan_count = float("nan")
    if baseline_nan_count is not None and max_nan_growth_fraction is not None:
        max_allowed_nan_count = float(baseline_nan_count) * (1.0 + float(max_nan_growth_fraction))
        if pred_nan_count > max_allowed_nan_count:
            hard_rejected_nan_growth = True
            mean_score = float(hard_reject_score)

    summary = {
        "mean_score_unpenalized": mean_score_unpenalized,
        "mean_score": mean_score,
        "mean_balancedRMSE": float(np.nanmean(block_balanced_rmse)) if block_balanced_rmse else float("nan"),
        "pred_valid_fraction": pred_valid_fraction,
        "pred_valid_count": int(global_metrics["global"]["pred_valid_count"]),
        "pred_nan_count": pred_nan_count,
        "ref_valid_count": int(global_metrics["global"]["ref_valid_count"]),
        "valid_fraction_penalty": valid_fraction_penalty,
        "baseline_nan_count": int(baseline_nan_count) if baseline_nan_count is not None else None,
        "max_allowed_nan_count": max_allowed_nan_count,
        "hard_rejected_nan_growth": hard_rejected_nan_growth,
        "reference_mean_diff": float(prediction.get("reference_mean_diff", np.nan)),
    }

    return {
        "candidate": dict(candidate),
        "summary": summary,
        "global_metrics": global_metrics,
    }


def optimize_hyperparameters_on_preprocessed(s: dict[str, Array], n_workers: int | None = None,
                                             objective: str = "balancedRMSE", roughness_weight: float = 0.0,
                                             coverage_weight: float = 0.0, coverage_target: float = 0.68,
                                             valid_fraction_weight: float = 1.0,
                                             max_nan_growth_fraction: float | None = 0.10) -> dict[str, Any]:
    """search the v2 hyperparameters on one already preprocessed dataset"""
    models = pr.load_parallel_scattering_models()
    cov_model = pr.load_covariance_model()
    base_options = pr.build_default_pipeline_options(models, cov_model)
    blocks = make_spatial_blocks(s["ra"], tile_rows, tile_cols, min_valid_pixels=min_valid_pixels)
    if not blocks:
        raise ValueError("No valid spatial blocks were found for v2 hyperparameter search.")

    parameter_specs = build_optimization_parameters()
    fixed_values = build_fixed_candidate_values()
    baseline_candidate = build_baseline_candidate(base_options, fixed_values)
    bounds = [(lower, upper) for _, lower, upper in parameter_specs]

    firstpass_cache: dict[tuple[Any, ...], tuple[Array, Array]] = {}
    result_cache: dict[tuple[Any, ...], dict[str, Any]] = {}
    eval_state = {"count": 0, "best_score": float("inf")}

    baseline_result = evaluate_candidate_on_preprocessed(
        s,
        baseline_candidate,
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
    baseline_nan_count = int(baseline_result["summary"]["pred_nan_count"])
    baseline_result["summary"]["baseline_nan_count"] = baseline_nan_count
    baseline_result["summary"]["max_allowed_nan_count"] = (
        float("nan")
        if max_nan_growth_fraction is None
        else float(baseline_nan_count) * (1.0 + float(max_nan_growth_fraction))
    )
    baseline_result["summary"]["hard_rejected_nan_growth"] = False
    result_cache[make_candidate_result_cache_key(baseline_candidate)] = baseline_result
    if max_nan_growth_fraction is None:
        print(f"Baseline NaN count on finite Diviner mask: {baseline_nan_count}", flush=True)
    else:
        print(
            f"Baseline NaN count on finite Diviner mask: {baseline_nan_count} "
            f"(max allowed during search: {baseline_nan_count * (1.0 + float(max_nan_growth_fraction)):.1f})",
            flush=True,
        )

    def get_result_for_vector(x: Array) -> tuple[dict[str, Any], dict[str, Any]]:
        candidate = candidate_from_vector(x, parameter_specs, fixed_values)
        candidate_key = make_candidate_result_cache_key(candidate)
        # cache whole candidate evaluations so direct and powell can revisit points cheaply
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
                baseline_nan_count=baseline_nan_count,
                max_nan_growth_fraction=max_nan_growth_fraction,
            )
        return candidate, result_cache[candidate_key]

    def objective_fun(x: Array) -> float:
        eval_state["count"] += 1
        candidate, result = get_result_for_vector(x)
        score = float(result["summary"]["mean_score"])
        if result["summary"]["hard_rejected_nan_growth"]:
            print(
                f"[eval {eval_state['count']}] rejected for NaN growth "
                f"predNaNs={result['summary']['pred_nan_count']} "
                f"baselineNaNs={result['summary']['baseline_nan_count']} "
                f"limit={result['summary']['max_allowed_nan_count']:.1f} "
                f"candidate={candidate}",
                flush=True,
            )
            return score
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
    # use direct for the global first pass, then let powell locally polish it
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

    return {
        "results": results,
        "parameter_specs": parameter_specs,
        "blocks": blocks,
    }


def optimize_hyperparameters(data_path: str | Path, n_workers: int | None = None, objective: str = "balancedRMSE",
                             roughness_weight: float = 0.0, coverage_weight: float = 0.0,
                             coverage_target: float = 0.68, valid_fraction_weight: float = 1.0,
                             max_nan_growth_fraction: float | None = 0.10) -> dict[str, Any]:
    """load, preprocess, and optimize the v2 pipeline on one dataset"""
    data = pr.load_analysis_dataset(data_path, list(pr.ANALYSIS_VARIABLES))
    s = pr.preprocess_minirf_data(data)
    return optimize_hyperparameters_on_preprocessed(
        s,
        n_workers=n_workers,
        objective=objective,
        roughness_weight=roughness_weight,
        coverage_weight=coverage_weight,
        coverage_target=coverage_target,
        valid_fraction_weight=valid_fraction_weight,
        max_nan_growth_fraction=max_nan_growth_fraction,
    )


def write_results_csv(results: list[dict[str, Any]], out_path: str | Path) -> Path:
    """write the ranked search results to a flat csv file"""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    for result in results:
        row = {
            "search_stage": result["search_stage"],
            "optimizer_success": result["optimizer_success"],
            "optimizer_message": result["optimizer_message"],
            "optimizer_nfev": result["optimizer_nfev"],
            "s": float(result["candidate"]["s"]),
            "sigmaSpatial": float(result["candidate"]["sigmaSpatial"]),
            "RA0": float(result["candidate"]["RA0"]),
            "mean_score": float(result["summary"]["mean_score"]),
            "mean_score_unpenalized": float(result["summary"]["mean_score_unpenalized"]),
            "mean_balancedRMSE": float(result["summary"]["mean_balancedRMSE"]),
            "pred_valid_fraction": float(result["summary"]["pred_valid_fraction"]),
            "pred_valid_count": int(result["summary"]["pred_valid_count"]),
            "pred_nan_count": int(result["summary"]["pred_nan_count"]),
            "ref_valid_count": int(result["summary"]["ref_valid_count"]),
            "valid_fraction_penalty": float(result["summary"]["valid_fraction_penalty"]),
            "baseline_nan_count": result["summary"]["baseline_nan_count"],
            "max_allowed_nan_count": float(result["summary"]["max_allowed_nan_count"]),
            "hard_rejected_nan_growth": bool(result["summary"]["hard_rejected_nan_growth"]),
            "reference_mean_diff": float(result["summary"]["reference_mean_diff"]),
        }
        rows.append(row)

    if not rows:
        return out_path

    with out_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    return out_path


if __name__ == "__main__":
    bounds_text = ", ".join(f"{name} in [{lower:.3g}, {upper:.3g}]" for name, lower, upper in build_optimization_parameters())
    print(f"Running v2 direct global search with Powell polish on {bounds_text}.", flush=True)
    search_output = optimize_hyperparameters(
        data_path,
        n_workers=n_workers,
        objective=objective,
        roughness_weight=roughness_weight,
        coverage_weight=coverage_weight,
        coverage_target=coverage_target,
        valid_fraction_weight=valid_fraction_weight,
        max_nan_growth_fraction=max_nan_growth_fraction,
    )

    print("\nTop candidates:", flush=True)
    for rank, result in enumerate(search_output["results"][:top_k], start=1):
        summary = result["summary"]
        print(
            f"{rank:>2}. stage={result['search_stage']} score={summary['mean_score']:.4f} "
            f"balancedRMSE={summary['mean_balancedRMSE']:.4f} "
            f"predValid={summary['pred_valid_fraction']:.4f} "
            f"candidate={result['candidate']}",
            flush=True,
        )

    if output_csv is not None:
        out_path = write_results_csv(search_output["results"], output_csv)
        print(f"\nWrote ranked results to {out_path}", flush=True)
