#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Translated v2 RA prediction workflow with smooth residual covariance
"""

import math
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from dataclasses import dataclass
import numpy as np
from numba import njit
from scipy.io import loadmat
import matplotlib.pyplot as plt
from typing import Any, Callable


Array = np.ndarray
REALMIN = np.finfo(float).tiny
ANALYSIS_VARIABLES = ["miniRF_1", "miniRF_2", "miniRF_3", "miniRF_4", "miniRF_5", "miniRF_6", "ra"]
MODEL_SIGMOID_COSD_EXP_RA = 0
MODEL_SIGMOID_COSD_LINEAR_RA = 1

# path to the v2 input dataset
data_path = Path(__file__).resolve().parents[1] / "data/GiordanoBruno.mat"

# path to the saved covariance grid exported from the matlab v2 file
covmodel_path = Path(__file__).resolve().parents[1] / "data/covModel_global_CPR_Green_2026-07-09_085116.npz"

# saved matlab reference prediction exported for comparison
reference_path = Path(__file__).resolve().parents[1] / "data/TestData_Bruno_RApred.npy"

# output path for the predicted RA and sigma maps
map_plot_output_path = Path(__file__).resolve().parents[1] / "output/ra_prediction_map_sigma.png"

# output path for the predicted RA histogram comparison
hist_plot_output_path = Path(__file__).resolve().parents[1] / "output/ra_prediction_histograms.png"

# color used for masked nan pixels in saved figures
nan_plot_color = "#ffffff"

# number of worker threads used by the spatial passes
workers = None

# compare the final prediction to the saved matlab reference map
compare_saved_reference = True

# lower bound of the posterior RA grid in percent
ra_grid_min = 0.01

# upper bound of the posterior RA grid in percent
ra_grid_max = 10.0

# number of points in the posterior RA grid
ra_grid_points = 500

# first-pass CPR and green mixing sharpness
firstpass_s = 1.5

# minimum model sigma used in the first pass
firstpass_sigma_floor = 0.02

# power-law pivot RA used in the first pass prior
firstpass_ra0 = 1.0

# square window size for the first-pass local neighborhood
firstpass_window_size = 3

# square window size for the second-pass spatial prior smoothing modes
secondpass_window_size = 5

# spatial prior mode used in the second pass
secondpass_spatial_prior_mode = "direct"

# covariance model flavor used in the second pass
secondpass_covariance_mode = "smooth"

# fractional width of the second-pass spatial prior
secondpass_sigma_spatial = 0.75

# minimum allowed width of the second-pass spatial prior
secondpass_min_sigma_spatial = 0.1

# maximum allowed width of the second-pass spatial prior
secondpass_max_sigma_spatial = 5.0

# regime edges used in the evaluation summary
eval_regime_edges = [0.0, 0.35, 1.0, 3.0, 5.0, 12.0]

# regime names used in the evaluation summary
eval_regime_names = ["Low RA", "Transition", "High RA1", "High RA2", "High RA3"]

# smoothing window used by the roughness diagnostic
eval_smooth_window = 5

# evaluate predictions in map mode
eval_map_mode = True

# print the evaluation summary to stdout
eval_report = True


@dataclass(frozen=True)
class ModelFit:
    model_fun: Callable[[Array, float | Array, Array], Array]
    params: Array
    model_kind: int


def default_ra_grid(ra_min: float = 0.01, ra_max: float = 10.0, n_grid_points: int = 500) -> Array:
    """Build default logarithmic RA grid"""
    epsilon = math.log10(ra_max / ra_min) / n_grid_points
    return np.power(10.0, math.log10(ra_min) + np.arange(n_grid_points + 1, dtype=float) * epsilon)


def normpdf(x: Array, mu: float | Array, sigma: float | Array) -> Array:
    """normal probability density function"""
    sigma = np.asarray(sigma, dtype=float)
    x = np.asarray(x, dtype=float)
    mu = np.asarray(mu, dtype=float)
    return np.exp(-0.5 * ((x - mu) / sigma) ** 2) / (sigma * math.sqrt(2.0 * math.pi))


def nanmedian_window(arr: Array, r0: int, r1: int, c0: int, c1: int) -> float:
    """Return the NaN-aware median of one 2D window"""
    window = arr[r0:r1, c0:c1]
    if np.all(np.isnan(window)):
        return np.nan
    return float(np.nanmedian(window))


def boxcar_median(arr: Array, window_size: int) -> Array:
    """Apply a 2D NaN-aware moving median with square windows"""
    arr = np.asarray(arr, dtype=float)
    nrows, ncols = arr.shape
    half = window_size // 2
    out = np.full_like(arr, np.nan, dtype=float)

    for r in range(nrows):
        r0 = max(0, r - half)
        r1 = min(nrows, r + half + 1)
        for c in range(ncols):
            c0 = max(0, c - half)
            c1 = min(ncols, c + half + 1)
            out[r, c] = nanmedian_window(arr, r0, r1, c0, c1)
    return out


def moving_median_1d(arr: Array, window_size: int, axis: int) -> Array:
    """Apply a 1D NaN-aware moving median along one axis"""
    arr = np.asarray(arr, dtype=float)
    out = np.full_like(arr, np.nan, dtype=float)
    half = window_size // 2

    if axis == 0:
        n = arr.shape[0]
        for i in range(n):
            i0 = max(0, i - half)
            i1 = min(n, i + half + 1)
            chunk = arr[i0:i1, :]
            out[i, :] = np.nanmedian(chunk, axis=0)
    elif axis == 1:
        n = arr.shape[1]
        for i in range(n):
            i0 = max(0, i - half)
            i1 = min(n, i + half + 1)
            chunk = arr[:, i0:i1]
            out[:, i] = np.nanmedian(chunk, axis=1)
    else:
        raise ValueError("axis must be 0 or 1")

    return out


def _nanpercentile(values: Array, q: float | list[float]) -> float | Array:
    """Compute percentiles after dropping non-finite values"""
    values = np.asarray(values, dtype=float)
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        if np.isscalar(q):
            return float("nan")
        return np.full(len(q), np.nan, dtype=float)
    return np.nanpercentile(finite, q)


def _rankdata_average(values: Array) -> Array:
    """Assign average ranks, matching the behavior needed for Spearman correlation"""
    values = np.asarray(values, dtype=float)
    order = np.argsort(values, kind="mergesort")
    sorted_values = values[order]
    ranks = np.empty(values.size, dtype=float)

    i = 0
    while i < values.size:
        j = i + 1
        while j < values.size and sorted_values[j] == sorted_values[i]:
            j += 1
        rank = 0.5 * (i + j - 1) + 1.0
        ranks[order[i:j]] = rank
        i = j
    return ranks


def _pearson_corr(x: Array, y: Array) -> float:
    """Compute finite-only Pearson correlation"""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    valid = np.isfinite(x) & np.isfinite(y)
    x = x[valid]
    y = y[valid]
    if x.size < 2:
        return float("nan")
    if np.allclose(x, x[0]) or np.allclose(y, y[0]):
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def _spearman_corr(x: Array, y: Array) -> float:
    """Compute finite-only Spearman correlation"""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    valid = np.isfinite(x) & np.isfinite(y)
    x = x[valid]
    y = y[valid]
    if x.size < 2:
        return float("nan")
    return _pearson_corr(_rankdata_average(x), _rankdata_average(y))


def _print_metrics_summary(metrics: dict[str, Any]) -> None:
    """Print a nice evaluation report"""
    threshold = metrics["highRA"]["threshold"]
    print("\nRA prediction evaluation summary")
    print("--------------------------------")
    print(f"N valid pixels: {metrics['N']}")
    print(f"RMSE: {metrics['global']['RMSE']:.4f}")
    print(f"Median absolute error: {metrics['global']['MAE_median']:.4f}")
    print(f"Median bias: {metrics['global']['bias_median']:.4f}")
    print(f"P90 absolute error: {metrics['global']['absErr_prctile_50_68_90_95'][2]:.4f}")
    print(f"Balanced RMSE: {metrics['global']['balancedRMSE']:.4f}")
    print(f"High-RA F1 at threshold {threshold:.2f}%: {metrics['highRA']['F1']:.4f}")
    print(f"Global Spearman: {metrics['global']['corrSpearman']:.2f}")
    print(f"Global Pearson: {metrics['global']['corrPearson']:.2f}")
    print("\nSpatial roughness metrics:")
    print(f"True median roughness: {metrics['spatial']['roughness_true_median']:.4f}")
    print(f"Pred median roughness: {metrics['spatial']['roughness_pred_median']:.4f}")
    print(f"Pred/true median roughness ratio: {metrics['spatial']['roughness_ratio_pred_true']:.4f}")
    print("True roughness percentiles [50 75 90 95]:", *[f"{x:.4f}" for x in metrics["spatial"]["roughness_true_prctile_50_75_90_95"]])
    print("Pred roughness percentiles [50 75 90 95]:", *[f"{x:.4f}" for x in metrics["spatial"]["roughness_pred_prctile_50_75_90_95"]])
    print(
        "Pred/true roughness percentile ratios [50 75 90 95]:",
        *[f"{x:.4f}" for x in metrics["spatial"]["roughness_ratio_prctile_50_75_90_95"]],
    )

    if np.isfinite(metrics["global"]["coverage68"]):
        print(f"68% coverage: {metrics['global']['coverage68']:.4f}")
        print(f"Median 68% width: {metrics['global']['width68_median']:.4f}")

    print("\nRegime metrics:")
    for row in metrics["regimeTable"]:
        print(row)


def evaluate_ra_prediction(ra_true: Array, ra_pred: Array, argopt: dict[str, Any] | None = None) -> dict[str, Any]:
    """Evaluate predicted RA map against reference map"""
    if argopt is None:
        argopt = {}
    argopt = dict(argopt)

    argopt.setdefault("plot", False)
    argopt.setdefault("RA_low", None)
    argopt.setdefault("RA_high", None)
    argopt.setdefault("regimeEdges", [0.0, 0.35, 1.5, math.inf])
    argopt.setdefault("regimeNames", ["Low RA", "Transition RA", "High RA"])
    argopt.setdefault("highThreshold", 1.5)
    argopt.setdefault("smoothWindow", 5)
    argopt.setdefault("mapMode", ra_true.ndim != 1)
    argopt.setdefault("report", True)

    if ra_true.shape != ra_pred.shape:
        raise ValueError("ra_true and ra_pred must have the same shape.")

    orig_size = ra_true.shape
    ra_true_flat = np.asarray(ra_true, dtype=float).reshape(-1, order="F")
    ra_pred_flat = np.asarray(ra_pred, dtype=float).reshape(-1, order="F")

    ref_valid = np.isfinite(ra_true_flat)
    pred_valid_on_ref = ref_valid & np.isfinite(ra_pred_flat)
    n_ref_valid = int(np.sum(ref_valid))
    n_pred_valid = int(np.sum(pred_valid_on_ref))
    pred_valid_fraction = float(n_pred_valid / n_ref_valid) if n_ref_valid > 0 else float("nan")

    valid = pred_valid_on_ref
    ra_true_v = ra_true_flat[valid]
    ra_pred_v = ra_pred_flat[valid]

    resid = ra_true_v - ra_pred_v
    abs_err = np.abs(resid)

    metrics: dict[str, Any] = {"N": int(ra_true_v.size), "originalSize": orig_size, "global": {}, "highRA": {}, "spatial": {}}
    metrics["global"]["ref_valid_count"] = n_ref_valid
    metrics["global"]["pred_valid_count"] = n_pred_valid
    metrics["global"]["pred_valid_fraction"] = pred_valid_fraction

    metrics["global"]["RMSE"] = float(np.sqrt(np.nanmean(resid**2)))
    metrics["global"]["MAE_median"] = float(np.nanmedian(abs_err))
    metrics["global"]["MAE_mean"] = float(np.nanmean(abs_err))
    metrics["global"]["bias_median"] = float(np.nanmedian(resid))
    metrics["global"]["bias_mean"] = float(np.nanmean(resid))
    metrics["global"]["resid_prctile_25_50_75"] = np.asarray(_nanpercentile(resid, [25, 50, 75]), dtype=float)
    metrics["global"]["absErr_prctile_50_68_90_95"] = np.asarray(_nanpercentile(abs_err, [50, 68, 90, 95]), dtype=float)
    metrics["global"]["resid_IQR"] = float(np.subtract(*np.nanpercentile(resid, [75, 25])))
    metrics["global"]["robustSigma"] = metrics["global"]["resid_IQR"] / 1.349
    metrics["global"]["corrPearson"] = _pearson_corr(ra_true_v, ra_pred_v)
    metrics["global"]["corrSpearman"] = _spearman_corr(ra_true_v, ra_pred_v)

    sse = float(np.nansum((ra_true_v - ra_pred_v) ** 2))
    sst = float(np.nansum((ra_true_v - np.nanmean(ra_true_v)) ** 2))
    metrics["global"]["R2"] = float("nan") if sst == 0.0 else 1.0 - sse / sst

    eps_ra = 0.1
    frac_err = abs_err / (ra_true_v + eps_ra)
    metrics["global"]["fracErr_median"] = float(np.nanmedian(frac_err))
    metrics["global"]["fracErr_p90"] = float(_nanpercentile(frac_err, 90))

    has_intervals = argopt["RA_low"] is not None and argopt["RA_high"] is not None
    if has_intervals:
        ra_low = np.asarray(argopt["RA_low"], dtype=float)
        ra_high = np.asarray(argopt["RA_high"], dtype=float)
        if ra_low.shape != orig_size or ra_high.shape != orig_size:
            raise ValueError("RA_low and RA_high must have same shape as ra_true.")
        ra_low_v = ra_low.reshape(-1, order="F")[valid]
        ra_high_v = ra_high.reshape(-1, order="F")[valid]
        covered68 = (ra_true_v >= ra_low_v) & (ra_true_v <= ra_high_v)
        width68 = ra_high_v - ra_low_v
        metrics["global"]["coverage68"] = float(np.nanmean(covered68))
        metrics["global"]["width68_median"] = float(np.nanmedian(width68))
        metrics["global"]["width68_p25_p50_p75"] = np.asarray(_nanpercentile(width68, [25, 50, 75]), dtype=float)
    else:
        metrics["global"]["coverage68"] = float("nan")
        metrics["global"]["width68_median"] = float("nan")
        metrics["global"]["width68_p25_p50_p75"] = np.array([np.nan, np.nan, np.nan], dtype=float)
        ra_low_v = None
        ra_high_v = None

    edges = list(argopt["regimeEdges"])
    names = list(argopt["regimeNames"])
    regime_table: list[dict[str, Any]] = []

    for k in range(len(edges) - 1):
        idx = (ra_true_v >= edges[k]) & (ra_true_v < edges[k + 1])
        true_reg = ra_true_v[idx]
        pred_reg = ra_pred_v[idx]
        row: dict[str, Any] = {
            "Regime": names[k],
            "N": int(np.sum(idx)),
            "RAmin": float(edges[k]),
            "RAmax": float(edges[k + 1]),
            "RMSE": float("nan"),
            "MAE_median": float("nan"),
            "Bias_median": float("nan"),
            "Bias_mean": float("nan"),
            "P90_absErr": float("nan"),
            "RobustSigma": float("nan"),
            "Coverage68": float("nan"),
            "Width68_median": float("nan"),
            "FracErr_median": float("nan"),
            "Pearson": float("nan"),
            "Spearman": float("nan"),
        }
        if np.any(idx):
            r = resid[idx]
            ae = abs_err[idx]
            row["RMSE"] = float(np.sqrt(np.nanmean(r**2)))
            row["MAE_median"] = float(np.nanmedian(ae))
            row["Bias_median"] = float(np.nanmedian(r))
            row["Bias_mean"] = float(np.nanmean(r))
            row["P90_absErr"] = float(_nanpercentile(ae, 90))
            row["RobustSigma"] = float(np.subtract(*np.nanpercentile(r, [75, 25])) / 1.349)
            row["FracErr_median"] = float(np.nanmedian(frac_err[idx]))

            if has_intervals and ra_low_v is not None and ra_high_v is not None:
                covered = (ra_true_v[idx] >= ra_low_v[idx]) & (ra_true_v[idx] <= ra_high_v[idx])
                width = ra_high_v[idx] - ra_low_v[idx]
                row["Coverage68"] = float(np.nanmean(covered))
                row["Width68_median"] = float(np.nanmedian(width))

            if true_reg.size > 2:
                row["Pearson"] = _pearson_corr(true_reg, pred_reg)
                row["Spearman"] = _spearman_corr(true_reg, pred_reg)

        regime_table.append(row)

    metrics["regimeTable"] = regime_table
    rmse_values = [row["RMSE"] for row in regime_table if np.isfinite(row["RMSE"])]
    mae_values = [row["MAE_median"] for row in regime_table if np.isfinite(row["MAE_median"])]
    metrics["global"]["balancedRMSE"] = float(np.nanmean(rmse_values)) if rmse_values else float("nan")
    metrics["global"]["balancedMAE"] = float(np.nanmean(mae_values)) if mae_values else float("nan")

    threshold = float(argopt["highThreshold"])
    true_high = ra_true_v >= threshold
    pred_high = ra_pred_v >= threshold
    tp = int(np.sum(true_high & pred_high))
    fp = int(np.sum(~true_high & pred_high))
    fn = int(np.sum(true_high & ~pred_high))
    tn = int(np.sum(~true_high & ~pred_high))
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2.0 * precision * recall / max(precision + recall, np.finfo(float).eps)
    metrics["highRA"] = {
        "threshold": threshold,
        "TP": tp,
        "FP": fp,
        "FN": fn,
        "TN": tn,
        "precision": precision,
        "recall": recall,
        "F1": f1,
    }

    if argopt["mapMode"]:
        try:
            ra_true_smooth = boxcar_median(np.asarray(ra_true, dtype=float), int(argopt["smoothWindow"]))
            ra_pred_smooth = boxcar_median(np.asarray(ra_pred, dtype=float), int(argopt["smoothWindow"]))
            rough_true = np.abs(np.asarray(ra_true, dtype=float) - ra_true_smooth)
            rough_pred = np.abs(np.asarray(ra_pred, dtype=float) - ra_pred_smooth)
            rough_true_v = rough_true.reshape(-1, order="F")
            rough_pred_v = rough_pred.reshape(-1, order="F")

            metrics["spatial"]["roughness_true_median"] = float(np.nanmedian(rough_true_v))
            metrics["spatial"]["roughness_pred_median"] = float(np.nanmedian(rough_pred_v))
            metrics["spatial"]["roughness_ratio_pred_true"] = (
                metrics["spatial"]["roughness_pred_median"] / metrics["spatial"]["roughness_true_median"]
            )
            metrics["spatial"]["roughness_true_prctile_50_75_90_95"] = np.asarray(
                _nanpercentile(rough_true_v, [50, 75, 90, 95]), dtype=float
            )
            metrics["spatial"]["roughness_pred_prctile_50_75_90_95"] = np.asarray(
                _nanpercentile(rough_pred_v, [50, 75, 90, 95]), dtype=float
            )
            metrics["spatial"]["roughness_ratio_prctile_50_75_90_95"] = (
                metrics["spatial"]["roughness_pred_prctile_50_75_90_95"]
                / metrics["spatial"]["roughness_true_prctile_50_75_90_95"]
            )
        except Exception:
            metrics["spatial"]["roughness_true_median"] = float("nan")
            metrics["spatial"]["roughness_pred_median"] = float("nan")
            metrics["spatial"]["roughness_ratio_pred_true"] = float("nan")
            metrics["spatial"]["roughness_true_prctile_50_75_90_95"] = np.array([np.nan, np.nan, np.nan, np.nan])
            metrics["spatial"]["roughness_pred_prctile_50_75_90_95"] = np.array([np.nan, np.nan, np.nan, np.nan])
            metrics["spatial"]["roughness_ratio_prctile_50_75_90_95"] = np.array([np.nan, np.nan, np.nan, np.nan])
    else:
        metrics["spatial"]["roughness_true_median"] = float("nan")
        metrics["spatial"]["roughness_pred_median"] = float("nan")
        metrics["spatial"]["roughness_ratio_pred_true"] = float("nan")
        metrics["spatial"]["roughness_true_prctile_50_75_90_95"] = np.array([np.nan, np.nan, np.nan, np.nan])
        metrics["spatial"]["roughness_pred_prctile_50_75_90_95"] = np.array([np.nan, np.nan, np.nan, np.nan])
        metrics["spatial"]["roughness_ratio_prctile_50_75_90_95"] = np.array([np.nan, np.nan, np.nan, np.nan])

    if argopt["report"]:
        _print_metrics_summary(metrics)

    return metrics


def cosd(x: float | Array) -> Array:
    """Return the cosine of angles expressed in degrees"""
    return np.cos(np.deg2rad(x))


def _sigmoid_cosd_exp_ra(params: Array, inc: float | Array, ra: Array) -> Array:
    """Evaluate the saved sigmoid-plus-cosine model with exponential RA incidence scaling"""
    p = np.asarray(params, dtype=float)
    ra = np.asarray(ra, dtype=float)
    inc_term = np.power(cosd(inc), p[4] * np.exp(p[5] * ra))
    sigmoid = p[0] + (p[1] / (1.0 + np.exp(-p[2] * (np.log10(ra) + p[3]))))
    linear = p[6] + p[7] * ra
    return sigmoid * inc_term + linear


def _sigmoid_cosd_linear_ra(params: Array, inc: float | Array, ra: Array) -> Array:
    """Evaluate the saved sigmoid-plus-cosine model with linear RA incidence scaling"""
    p = np.asarray(params, dtype=float)
    ra = np.asarray(ra, dtype=float)
    inc_term = np.power(cosd(inc), p[4] + p[5] * ra)
    sigmoid = p[0] + (p[1] / (1.0 + np.exp(-p[2] * (np.log10(ra) + p[3]))))
    linear = p[6] + p[7] * ra
    return sigmoid * inc_term + linear


@njit(cache=True)
def _trapz_1d(y: Array, x: Array) -> float:
    """Integrate a sampled 1D curve with the trapezoid rule"""
    total = 0.0
    for i in range(y.size - 1):
        total += 0.5 * (y[i] + y[i + 1]) * (x[i + 1] - x[i])
    return total


@njit(cache=True)
def normalize_density_inplace(values: Array, grid: Array) -> None:
    """Clamp invalid density values and normalize them in place"""
    has_positive = False
    for i in range(values.size):
        value = values[i]
        if not math.isfinite(value) or value < 0.0:
            value = 0.0
        values[i] = value
        if value > 0.0:
            has_positive = True

    if not has_positive:
        for i in range(values.size):
            values[i] = 1.0

    area = _trapz_1d(values, grid)
    if not math.isfinite(area) or area <= 0.0:
        for i in range(values.size):
            values[i] = 1.0
        area = _trapz_1d(values, grid)

    if not math.isfinite(area) or area <= 0.0:
        fill = 1.0 / max(values.size, 1)
        for i in range(values.size):
            values[i] = fill
        return

    for i in range(values.size):
        values[i] /= area


@njit(cache=True)
def model_value(model_kind: int, params: Array, inc_deg: float, ra: float) -> float:
    """Evaluate one hardcoded scattering-model fit at one RA value"""
    inc_cos = math.cos((math.pi / 180.0) * inc_deg)
    if model_kind == MODEL_SIGMOID_COSD_EXP_RA:
        inc_term = inc_cos ** (params[4] * math.exp(params[5] * ra))
    else:
        inc_term = inc_cos ** (params[4] + params[5] * ra)

    sigmoid = params[0] + (params[1] / (1.0 + math.exp(-params[2] * (math.log10(ra) + params[3]))))
    linear = params[6] + params[7] * ra
    return sigmoid * inc_term + linear


@njit(cache=True)
def beta_weight(ra: float, s: float) -> float:
    """Return the RA-dependent mixing weight used by the translated workflow"""
    return 0.01 + (1.0 - 0.01) / (1.0 + math.exp((ra - 2.5) / s))


@njit(cache=True)
def interp_monotonic_prefix(x: float, xp: Array, fp: Array, count: int) -> float:
    """Interpolate on the strictly increasing prefix of a monotonic grid"""
    if count == 0:
        return math.nan
    if x <= xp[0]:
        return fp[0]

    for i in range(1, count):
        if x <= xp[i]:
            x0 = xp[i - 1]
            x1 = xp[i]
            y0 = fp[i - 1]
            y1 = fp[i]
            if x1 <= x0:
                return y1
            return y0 + ((x - x0) * (y1 - y0) / (x1 - x0))

    return fp[count - 1]


@njit(cache=True)
def posterior_summary_from_loglike(log_l: Array, prior: Array, ra_grid: Array) -> tuple[float, float, float, float, float]:
    """convert a log-likelihood curve into map, mean, and central interval summaries"""
    ngrid = ra_grid.size
    log_post = np.empty(ngrid, dtype=np.float64)
    post = np.empty(ngrid, dtype=np.float64)

    max_log_post = -math.inf
    for i in range(ngrid):
        log_post[i] = log_l[i] + math.log(max(prior[i], REALMIN))
        if log_post[i] > max_log_post:
            max_log_post = log_post[i]

    max_idx = 0
    max_post = -math.inf
    for i in range(ngrid):
        post[i] = math.exp(log_post[i] - max_log_post)

    normalize_density_inplace(post, ra_grid)

    for i in range(ngrid):
        if post[i] > max_post:
            max_post = post[i]
            max_idx = i

    post_mean = 0.0
    for i in range(1, ngrid):
        post_mean += 0.5 * (ra_grid[i - 1] * post[i - 1] + ra_grid[i] * post[i]) * (ra_grid[i] - ra_grid[i - 1])

    # build a monotonic cdf for the posterior quantiles
    cdf = np.empty(ngrid, dtype=np.float64)
    cdf[0] = 0.0
    for i in range(1, ngrid):
        cdf[i] = cdf[i - 1] + 0.5 * (post[i - 1] + post[i]) * (ra_grid[i] - ra_grid[i - 1])

    total = cdf[ngrid - 1]
    if not math.isfinite(total) or total <= 0.0:
        return (ra_grid[max_idx], math.nan, math.nan, math.nan, math.nan)

    for i in range(ngrid):
        cdf[i] /= total

    unique_cdf = np.empty(ngrid, dtype=np.float64)
    unique_ra = np.empty(ngrid, dtype=np.float64)
    count = 0
    for i in range(ngrid):
        if count == 0 or cdf[i] > unique_cdf[count - 1]:
            unique_cdf[count] = cdf[i]
            unique_ra[count] = ra_grid[i]
            count += 1

    ra_low = interp_monotonic_prefix(0.16, unique_cdf, unique_ra, count)
    ra_med = interp_monotonic_prefix(0.50, unique_cdf, unique_ra, count)
    ra_high = interp_monotonic_prefix(0.84, unique_cdf, unique_ra, count)

    return (ra_grid[max_idx], post_mean, ra_med, ra_low, ra_high)


@njit(cache=True)
def firstpass_map_from_loglikes(log_l1: Array, log_l2: Array, prior: Array, ra_grid: Array, s: float) -> tuple[float, float]:
    """blend the independent CPR and m-chi green posteriors with the beta switch"""
    ngrid = ra_grid.size

    log_post1 = np.empty(ngrid, dtype=np.float64)
    log_post2 = np.empty(ngrid, dtype=np.float64)
    post1 = np.empty(ngrid, dtype=np.float64)
    post2 = np.empty(ngrid, dtype=np.float64)
    post = np.empty(ngrid, dtype=np.float64)
    beta1_values = np.empty(ngrid, dtype=np.float64)

    max_log_post1 = -math.inf
    max_log_post2 = -math.inf
    for i in range(ngrid):
        log_post1[i] = log_l1[i] + math.log(max(prior[i], REALMIN))
        log_post2[i] = log_l2[i] + math.log(max(prior[i], REALMIN))
        if log_post1[i] > max_log_post1:
            max_log_post1 = log_post1[i]
        if log_post2[i] > max_log_post2:
            max_log_post2 = log_post2[i]

    for i in range(ngrid):
        post1[i] = math.exp(log_post1[i] - max_log_post1)
        post2[i] = math.exp(log_post2[i] - max_log_post2)

    normalize_density_inplace(post1, ra_grid)
    normalize_density_inplace(post2, ra_grid)

    max_idx = 0
    max_post = -math.inf
    for i in range(ngrid):
        beta1 = beta_weight(ra_grid[i], s)
        beta1_values[i] = beta1
        post[i] = beta1 * post1[i] + (1.0 - beta1) * post2[i]

    normalize_density_inplace(post, ra_grid)

    for i in range(ngrid):
        if post[i] > max_post:
            max_post = post[i]
            max_idx = i

    ra_best = ra_grid[max_idx]
    flag = 0.0

    # mirror the matlab low-RA fallback that swaps the radar roles once
    if ra_best < 0.3:
        max_idx = 0
        max_post = -math.inf
        for i in range(ngrid):
            post[i] = beta1_values[i] * post2[i] + (1.0 - beta1_values[i]) * post1[i]
        normalize_density_inplace(post, ra_grid)
        for i in range(ngrid):
            if post[i] > max_post:
                max_post = post[i]
                max_idx = i
        ra_best = ra_grid[max_idx]
        flag = 1.0

    if ra_best < 0.3:
        ra_best = 0.3
        flag = 2.0

    return (ra_best, flag)


@njit(cache=True)
def fit_local_scattering_curve_2rad_wbeta_numba(rad1_vec: Array, rad2_vec: Array, inc_vec: Array, w_vec: Array,
                                                 ra_grid: Array, sigma_floor: float, alpha: float,
                                                 ra0: float, s: float, prior: Array, use_prior: bool,
                                                 mu1_kind: int, mu1_params: Array, sigma1_kind: int, sigma1_params: Array,
                                                 mu2_kind: int, mu2_params: Array, sigma2_kind: int, sigma2_params: Array
                                                 ) -> tuple[float, float]:
    """score one local first-pass patch over the RA grid"""
    nobs = rad1_vec.size
    ngrid = ra_grid.size

    if nobs == 0 or ngrid == 0:
        return (math.nan, math.nan)

    weights = w_vec.copy()
    all_zero = True
    for k in range(nobs):
        weight = weights[k]
        if not math.isfinite(weight) or weight < 0.0:
            weight = 0.0
        weights[k] = weight
        if weight != 0.0:
            all_zero = False

    if all_zero:
        for k in range(nobs):
            weights[k] = 1.0

    mean_weight = 0.0
    for k in range(nobs):
        mean_weight += weights[k]
    mean_weight /= nobs
    if not math.isfinite(mean_weight) or mean_weight <= 0.0:
        mean_weight = 1.0

    sumw = 0.0
    sumw2 = 0.0
    for k in range(nobs):
        weights[k] /= mean_weight
        sumw += weights[k]
        sumw2 += weights[k] * weights[k]

    # inflate the model spread when the local patch has low effective support
    neff_weighted = (sumw * sumw) / sumw2
    neff_corr = math.sqrt(nobs)
    neff = min(neff_weighted, neff_corr)
    neff = max(1.0, min(neff, float(nobs)))
    inflate = math.sqrt(nobs / neff)

    chi1 = np.zeros(ngrid, dtype=np.float64)
    chi2 = np.zeros(ngrid, dtype=np.float64)
    for k in range(nobs):
        inc = inc_vec[k]
        rad1 = rad1_vec[k]
        rad2 = rad2_vec[k]
        weight = weights[k]
        for i in range(ngrid):
            ra = ra_grid[i]

            mu1 = model_value(mu1_kind, mu1_params, inc, ra)
            sigma1 = model_value(sigma1_kind, sigma1_params, inc, ra) / 1.349
            if sigma1 < sigma_floor:
                sigma1 = sigma_floor
            sigma1 *= inflate

            mu2 = model_value(mu2_kind, mu2_params, inc, ra)
            sigma2 = model_value(sigma2_kind, sigma2_params, inc, ra) / 1.349
            if sigma2 < sigma_floor:
                sigma2 = sigma_floor
            sigma2 *= inflate

            diff1 = (rad1 - mu1) / sigma1
            diff2 = (rad2 - mu2) / sigma2
            chi1[i] += weight * diff1 * diff1
            chi2[i] += weight * diff2 * diff2

    log_l1 = np.empty(ngrid, dtype=np.float64)
    log_l2 = np.empty(ngrid, dtype=np.float64)
    max_log_l1 = -math.inf
    max_log_l2 = -math.inf
    for i in range(ngrid):
        log_l1[i] = -0.5 * chi1[i]
        log_l2[i] = -0.5 * chi2[i]
        if log_l1[i] > max_log_l1:
            max_log_l1 = log_l1[i]
        if log_l2[i] > max_log_l2:
            max_log_l2 = log_l2[i]

    for i in range(ngrid):
        log_l1[i] -= max_log_l1
        log_l2[i] -= max_log_l2

    if use_prior:
        prior_work = prior.copy()
    else:
        prior_work = np.ones(ngrid, dtype=np.float64)
        for i in range(ngrid):
            if ra_grid[i] > ra0:
                prior_work[i] = (ra_grid[i] / ra0) ** (-alpha)

    normalize_density_inplace(prior_work, ra_grid)
    return firstpass_map_from_loglikes(log_l1, log_l2, prior_work, ra_grid, s)


@njit(cache=True)
def find_bin(value: float, edges: Array) -> int:
    """return the closed-open bin index for one scalar value"""
    n = edges.size - 1
    if n <= 0:
        return -1
    if value < edges[0] or value > edges[n]:
        return -1
    if value == edges[n]:
        return n - 1
    for i in range(n):
        if value >= edges[i] and value < edges[i + 1]:
            return i
    return -1


@njit(cache=True)
def lookup_rho(ra: float, inc: float, ra_edges: Array, inc_edges: Array, rho_ns: Array) -> float:
    """look up rho from the saved covariance grid"""
    ra_bin = find_bin(ra, ra_edges)
    inc_bin = find_bin(inc, inc_edges)
    if ra_bin < 0 or inc_bin < 0:
        return math.nan
    return rho_ns[ra_bin, inc_bin]


@njit(cache=True)
def smooth_rho(ra: float, inc: float, rho_eta_params: Array, rho_feature_stats: Array) -> float:
    """evaluate the smooth rho = tanh eta fit used in the second pass"""
    log_ra = math.log10(ra)
    x_ra = (log_ra - rho_feature_stats[0]) / rho_feature_stats[1]
    x_inc = (inc - rho_feature_stats[2]) / rho_feature_stats[3]

    eta = rho_eta_params[0]
    eta += rho_eta_params[1] * x_ra
    eta += rho_eta_params[2] * x_ra * x_ra
    eta += rho_eta_params[3] * x_inc
    eta += rho_eta_params[4] * x_inc * x_inc
    eta += rho_eta_params[5] * x_ra * x_inc

    rho = math.tanh(eta)
    if rho > 0.999:
        rho = 0.999
    if rho < -0.999:
        rho = -0.999
    return rho


@njit(cache=True)
def invert_ra_from_rad_second_2rad_joint_numba(rad1_vec: Array, rad2_vec: Array, inc_vec: Array, prior: Array,
                                                ra_grid: Array, has_neff: bool, neff_value: float,
                                                mu1_kind: int, mu1_params: Array, sigma1_kind: int, sigma1_params: Array,
                                                mu2_kind: int, mu2_params: Array, sigma2_kind: int, sigma2_params: Array,
                                                rho_eta_params: Array, rho_feature_stats: Array
                                                ) -> tuple[float, float, float, float, float]:
    """evaluate the joint CPR-G likelihood across one posterior RA grid"""
    nobs = rad1_vec.size
    ngrid = ra_grid.size
    if nobs == 0 or ngrid == 0:
        return (math.nan, math.nan, math.nan, math.nan, math.nan)

    if has_neff:
        neff = neff_value
    else:
        neff = math.sqrt(nobs)
    if not math.isfinite(neff) or neff <= 0.0:
        neff = 1.0
    neff = min(neff, float(nobs))
    inflate = math.sqrt(nobs / neff)

    log_l = np.zeros(ngrid, dtype=np.float64)
    for k in range(nobs):
        inc = inc_vec[k]
        rad1 = rad1_vec[k]
        rad2 = rad2_vec[k]
        for i in range(ngrid):
            ra = ra_grid[i]

            mu1 = model_value(mu1_kind, mu1_params, inc, ra)
            sigma1 = model_value(sigma1_kind, sigma1_params, inc, ra) / 1.349
            if sigma1 < 0.02:
                sigma1 = 0.02
            sigma1 *= inflate

            mu2 = model_value(mu2_kind, mu2_params, inc, ra)
            sigma2 = model_value(sigma2_kind, sigma2_params, inc, ra) / 1.349
            if sigma2 < 0.02:
                sigma2 = 0.02
            sigma2 *= inflate

            # rho = lookup_rho(ra, inc, ra_edges, inc_edges, rho_ns)
            rho = smooth_rho(ra, inc, rho_eta_params, rho_feature_stats)
            if not math.isfinite(rho):
                rho = 0.0

            dx1 = rad1 - mu1
            dx2 = rad2 - mu2

            one_minus_rho2 = 1.0 - rho * rho
            if one_minus_rho2 < 1e-12:
                one_minus_rho2 = 1e-12

            # quadratic form for the 2d gaussian residual model
            quad = ((dx1 * dx1) / (sigma1 * sigma1))
            quad -= 2.0 * rho * dx1 * dx2 / (sigma1 * sigma2)
            quad += (dx2 * dx2) / (sigma2 * sigma2)
            quad /= one_minus_rho2

            log_det_sigma = 2.0 * math.log(sigma1) + 2.0 * math.log(sigma2) + math.log(one_minus_rho2)
            log_l[i] += -0.5 * (log_det_sigma + quad)

    prior_work = prior.copy()
    normalize_density_inplace(prior_work, ra_grid)
    return posterior_summary_from_loglike(log_l, prior_work, ra_grid)


@njit(cache=True)
def invert_ra_from_rad_second_2rad_joint_lookup_numba(rad1_vec: Array, rad2_vec: Array, inc_vec: Array, prior: Array,
                                                       ra_grid: Array, has_neff: bool, neff_value: float,
                                                       mu1_kind: int, mu1_params: Array, sigma1_kind: int, sigma1_params: Array,
                                                       mu2_kind: int, mu2_params: Array, sigma2_kind: int, sigma2_params: Array,
                                                       ra_edges: Array, inc_edges: Array, rho_ns: Array
                                                       ) -> tuple[float, float, float, float, float]:
    """evaluate the joint CPR-G likelihood using the saved lookup table rho"""
    nobs = rad1_vec.size
    ngrid = ra_grid.size
    if nobs == 0 or ngrid == 0:
        return (math.nan, math.nan, math.nan, math.nan, math.nan)

    if has_neff:
        neff = neff_value
    else:
        neff = math.sqrt(nobs)
    if not math.isfinite(neff) or neff <= 0.0:
        neff = 1.0
    neff = min(neff, float(nobs))
    inflate = math.sqrt(nobs / neff)

    log_l = np.zeros(ngrid, dtype=np.float64)
    for k in range(nobs):
        inc = inc_vec[k]
        rad1 = rad1_vec[k]
        rad2 = rad2_vec[k]
        for i in range(ngrid):
            ra = ra_grid[i]

            mu1 = model_value(mu1_kind, mu1_params, inc, ra)
            sigma1 = model_value(sigma1_kind, sigma1_params, inc, ra) / 1.349
            if sigma1 < 0.02:
                sigma1 = 0.02
            sigma1 *= inflate

            mu2 = model_value(mu2_kind, mu2_params, inc, ra)
            sigma2 = model_value(sigma2_kind, sigma2_params, inc, ra) / 1.349
            if sigma2 < 0.02:
                sigma2 = 0.02
            sigma2 *= inflate

            rho = lookup_rho(ra, inc, ra_edges, inc_edges, rho_ns)
            if not math.isfinite(rho):
                rho = 0.0

            dx1 = rad1 - mu1
            dx2 = rad2 - mu2

            one_minus_rho2 = 1.0 - rho * rho
            if one_minus_rho2 < 1e-12:
                one_minus_rho2 = 1e-12

            quad = ((dx1 * dx1) / (sigma1 * sigma1))
            quad -= 2.0 * rho * dx1 * dx2 / (sigma1 * sigma2)
            quad += (dx2 * dx2) / (sigma2 * sigma2)
            quad /= one_minus_rho2

            log_det_sigma = 2.0 * math.log(sigma1) + 2.0 * math.log(sigma2) + math.log(one_minus_rho2)
            log_l[i] += -0.5 * (log_det_sigma + quad)

    prior_work = prior.copy()
    normalize_density_inplace(prior_work, ra_grid)
    return posterior_summary_from_loglike(log_l, prior_work, ra_grid)


def load_parallel_scattering_models() -> dict[str, ModelFit]:
    """return the v2 scattering-fit coefficients"""
    return {
        "Rad1muFit": ModelFit(
            model_fun=_sigmoid_cosd_exp_ra,
            params=np.array(
                [-0.837883, 0.585569, 11.1532, 0.147126, 1.16211, 0.286385, 0.967473, -0.0141565],
                dtype=float,
            ),
            model_kind=MODEL_SIGMOID_COSD_EXP_RA,
        ),
        "Rad1iqrFit": ModelFit(
            model_fun=_sigmoid_cosd_linear_ra,
            params=np.array(
                [0.0870258, 0.137405, 28.9929, 0.293797, 0.951332, 0.4565, 0.0716333, 0.00852198],
                dtype=float,
            ),
            model_kind=MODEL_SIGMOID_COSD_LINEAR_RA,
        ),
        "Rad2muFit": ModelFit(
            model_fun=_sigmoid_cosd_exp_ra,
            params=np.array(
                [0.104786, 0.472109, 7.53962, 0.145583, 1.6987, -0.00172229, 0.186198, 0.00914963],
                dtype=float,
            ),
            model_kind=MODEL_SIGMOID_COSD_EXP_RA,
        ),
        "Rad2iqrFit": ModelFit(
            model_fun=_sigmoid_cosd_exp_ra,
            params=np.array(
                [0.0958137, 0.259142, 17.003, 0.271329, 3.08496, -0.058395, 0.0377035, 0.00164396],
                dtype=float,
            ),
            model_kind=MODEL_SIGMOID_COSD_EXP_RA,
        ),
    }


def load_covariance_model(npz_path: str | Path | None = None) -> dict[str, Any]:
    """load the saved covariance grid and the smooth rho fit parameters"""
    if npz_path is None:
        npz_path = covmodel_path
    loaded = np.load(Path(npz_path), allow_pickle=False)
    return {
        "RA_centers": np.asarray(loaded["ra_centers"], dtype=float),
        "RA_edges": np.asarray(loaded["ra_edges"], dtype=float),
        "Inc_centers": np.asarray(loaded["inc_centers"], dtype=float),
        "Inc_edges": np.asarray(loaded["inc_edges"], dtype=float),
        "rho": np.asarray(loaded["rho"], dtype=float),
        "rho_ns": np.asarray(loaded["rho_ns"], dtype=float),
        "useIncBins": bool(loaded["use_inc_bins"]),
        "minBinCount": float(loaded["min_bin_count"]),
        "rhoEtaParams": np.array(
            [0.598961212386507, 0.03848833871715411, -0.011973352070269243, -0.12846569439876904, -0.005791456829483283, 0.009761158800647976],
            dtype=float,
        ),
        "rhoFeatureStats": np.array(
            [-0.4276433510339205, 0.2652281082339094, 51.424386143824044, 7.525921078673727],
            dtype=float,
        ),
    }


def load_analysis_mat(mat_path: str | Path, variable_names: list[str] | None = None) -> dict[str, Array]:
    """load the matlab analysis export"""
    mat_path = Path(mat_path).resolve()
    if variable_names is None:
        variable_names = list(ANALYSIS_VARIABLES)

    raw = loadmat(mat_path, variable_names=variable_names)
    data: dict[str, Array] = {}
    for name in variable_names:
        if name not in raw:
            raise ValueError(f"MAT file is missing required variable {name!r}.")
        data[name] = np.asarray(raw[name], dtype=float)
    return data


def load_analysis_dataset(data_path: str | Path, variable_names: list[str] | None = None) -> dict[str, Array]:
    """load the matlab analysis dataset"""
    data_path = Path(data_path)
    if variable_names is None:
        variable_names = list(ANALYSIS_VARIABLES)

    suffix = data_path.suffix.lower()
    if suffix == ".mat":
        return load_analysis_mat(data_path, variable_names)
    raise ValueError(f"Unsupported data file type {data_path.suffix!r}. Expected .mat file.")


def preprocess_minirf_data(data: dict[str, Array]) -> dict[str, Array]:
    """apply quality masking and derive CPR and m-chi green observables"""
    s = {key: np.array(value, dtype=float, copy=True) for key, value in data.items()}

    pwrchk = s["miniRF_1"] < np.sqrt(s["miniRF_2"] ** 2 + s["miniRF_3"] ** 2 + s["miniRF_4"] ** 2)
    for key in ("miniRF_1", "miniRF_2", "miniRF_3", "miniRF_4"):
        s[key][pwrchk] = np.nan

    with np.errstate(invalid="ignore", divide="ignore"):
        s["m"] = np.sqrt(s["miniRF_2"] ** 2 + s["miniRF_3"] ** 2 + s["miniRF_4"] ** 2) / s["miniRF_1"]
        s["g"] = np.sqrt(s["miniRF_1"] * (1.0 - s["m"]))
        s["cpr"] = (s["miniRF_1"] - s["miniRF_4"]) / (s["miniRF_1"] + s["miniRF_4"])

    s["inc"] = s["miniRF_5"]
    s["num"] = s["miniRF_6"]
    s["ra"] = s["ra"] * 100.0

    mask = (s["num"] == 1) & (s["ra"] > 0.01) & ~np.isnan(s["cpr"]) & ~np.isnan(s["inc"])

    for key in ("cpr", "g", "inc", "ra"):
        s[key][~mask] = np.nan

    s["Rad1"] = s["cpr"]
    s["Rad2"] = s["g"]
    s["IncidenceMap"] = s["inc"]
    s["RAMap"] = s["ra"]
    return s


def make_sigma_map(ra_pred_med: Array, ra_pred_low: Array, ra_pred_high: Array, ra_truth: Array) -> Array:
    """compute the per-pixel sigma offset between the prediction and diviner RA"""
    ra_pred_med = np.asarray(ra_pred_med, dtype=float)
    ra_pred_low = np.asarray(ra_pred_low, dtype=float)
    ra_pred_high = np.asarray(ra_pred_high, dtype=float)
    ra_truth = np.asarray(ra_truth, dtype=float)

    sigma_map = np.full(ra_pred_med.shape, np.nan, dtype=float)
    valid = np.isfinite(ra_pred_med) & np.isfinite(ra_pred_low) & np.isfinite(ra_pred_high) & np.isfinite(ra_truth)
    if not np.any(valid):
        return sigma_map

    lower_width = ra_pred_med[valid] - ra_pred_low[valid]
    upper_width = ra_pred_high[valid] - ra_pred_med[valid]
    sigma_width = np.where(ra_truth[valid] <= ra_pred_med[valid], lower_width, upper_width)
    sigma_width = np.where(sigma_width < 1e-12, 1e-12, sigma_width)

    delta = np.abs(ra_truth[valid] - ra_pred_med[valid])
    sigma_map[valid] = delta / sigma_width
    return sigma_map


def compute_sigma_envelope(ra_pred_med: Array, ra_pred_low: Array, ra_pred_high: Array,
                           ra_truth: Array, max_sigma: int = 5) -> Array:
    """map each pixel to the integer sigma envelope containing the truth"""
    sigma_map = make_sigma_map(ra_pred_med, ra_pred_low, ra_pred_high, ra_truth)
    sigma_envelope_map = np.floor(sigma_map)
    sigma_envelope_map = np.where(sigma_envelope_map > max_sigma, float(max_sigma), sigma_envelope_map)
    return sigma_envelope_map


def make_ra_map_sigma_plot(ra_pred: Array, sigma_map: Array | None = None,
                           output_path: str | Path | None = None) -> Path:
    """quicklook plot with the predicted RA map and diviner sigma map"""
    if output_path is None:
        output_path = map_plot_output_path
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    ra_pred = np.asarray(ra_pred, dtype=float)
    sigma_map_array = None if sigma_map is None else np.asarray(sigma_map, dtype=float)

    fig, (ax_map, ax_sigma) = plt.subplots(1, 2, figsize=(12, 5.5), constrained_layout=True)

    cmap = plt.get_cmap("viridis").copy()
    cmap.set_bad(color=nan_plot_color)
    image = ax_map.imshow(np.ma.masked_invalid(ra_pred), cmap=cmap, origin="upper")
    fig.colorbar(image, ax=ax_map, label="predicted RA (%)")
    ax_map.set_title("predicted RA map")
    ax_map.set_xticks([])
    ax_map.set_yticks([])

    sigma_cmap = plt.get_cmap("magma").copy()
    sigma_cmap.set_bad(color=nan_plot_color)
    if sigma_map_array is not None and np.any(np.isfinite(sigma_map_array)):
        sigma_max = float(np.nanpercentile(sigma_map_array[np.isfinite(sigma_map_array)], 99.0))
        sigma_max = max(sigma_max, 1.0)
        sigma_image = ax_sigma.imshow(np.ma.masked_invalid(sigma_map_array), cmap=sigma_cmap, origin="upper", vmin=0.0, vmax=sigma_max)
        fig.colorbar(sigma_image, ax=ax_sigma, label="sigma offset")
    else:
        ax_sigma.imshow(np.ma.masked_all(ra_pred.shape), cmap=sigma_cmap, origin="upper")
    ax_sigma.set_title("Diviner RA sigma map")
    ax_sigma.set_xticks([])
    ax_sigma.set_yticks([])

    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    return output_path


def make_ra_histogram_plot(ra_true: Array, ra_pred_median: Array, ra_pred_mean: Array,
                           output_path: str | Path | None = None) -> Path:
    """save full and tail RA histograms for true distribution, posterior median, and posterior mean"""
    if output_path is None:
        output_path = hist_plot_output_path
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    ra_true = np.asarray(ra_true, dtype=float)
    ra_pred_median = np.asarray(ra_pred_median, dtype=float)
    ra_pred_mean = np.asarray(ra_pred_mean, dtype=float)

    true_valid = ra_true[np.isfinite(ra_true)]
    median_valid = ra_pred_median[np.isfinite(ra_pred_median)]
    mean_valid = ra_pred_mean[np.isfinite(ra_pred_mean)]

    if true_valid.size == 0:
        raise ValueError("Cannot plot an empty reference RA distribution.")

    hist_max = float(np.nanmax(true_valid))
    if median_valid.size > 0:
        hist_max = max(hist_max, float(np.nanmax(median_valid)))
    if mean_valid.size > 0:
        hist_max = max(hist_max, float(np.nanmax(mean_valid)))
    hist_max = max(hist_max, 10.0)

    bins_full = np.linspace(0.0, hist_max, 120)
    bins_tail = np.linspace(0.35, min(hist_max, 10.0), 120)

    fig, (ax_full, ax_tail) = plt.subplots(1, 2, figsize=(14, 5.5), constrained_layout=True)

    ax_full.hist(true_valid, bins=bins_full, color="#264653", alpha=0.35, label="reference RA")
    ax_full.hist(median_valid, bins=bins_full, color="#e76f51", alpha=0.5, label="posterior median")
    ax_full.hist(mean_valid, bins=bins_full, color="#2a9d8f", alpha=0.5, label="posterior mean")
    ax_full.set_xlabel("RA (%)")
    ax_full.set_ylabel("log10(count)")
    ax_full.set_yscale("log")
    ax_full.set_title("smooth rho full distribution")
    ax_full.grid(True, alpha=0.25)
    ax_full.legend(loc="best", frameon=True)

    ax_tail.hist(true_valid[true_valid >= 0.35], bins=bins_tail, color="#264653", alpha=0.35, label="reference RA")
    ax_tail.hist(median_valid[median_valid >= 0.35], bins=bins_tail, color="#e76f51", alpha=0.5, label="posterior median")
    ax_tail.hist(mean_valid[mean_valid >= 0.35], bins=bins_tail, color="#2a9d8f", alpha=0.5, label="posterior mean")
    ax_tail.set_xlabel("RA (%)")
    ax_tail.set_ylabel("log10(count)")
    ax_tail.set_yscale("log")
    ax_tail.set_title("smooth rho tail for RA >= 0.35%")
    ax_tail.grid(True, alpha=0.25)
    ax_tail.legend(loc="best", frameon=True)

    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    return output_path


def fit_local_scattering_curve_2rad_wbeta(rad1_vec: Array, rad2_vec: Array, inc_vec: Array,
                                          mu1_fit: ModelFit, sigma1_fit: ModelFit, mu2_fit: ModelFit, sigma2_fit: ModelFit,
                                          w_vec: Array, settings: dict[str, Any]) -> dict[str, Any]:
    """wrapper around the numba first-pass local inversion kernel"""
    settings = dict(settings)

    rad1_vec = np.asarray(rad1_vec, dtype=float).reshape(-1)
    rad2_vec = np.asarray(rad2_vec, dtype=float).reshape(-1)
    inc_vec = np.asarray(inc_vec, dtype=float).reshape(-1)
    w_vec = np.asarray(w_vec, dtype=float).reshape(-1)

    if not (rad1_vec.size == rad2_vec.size == inc_vec.size == w_vec.size):
        raise ValueError("rad1_vec, rad2_vec, inc_vec, and w_vec must have the same number of elements.")

    valid = np.isfinite(rad1_vec) & np.isfinite(rad2_vec) & np.isfinite(inc_vec) & np.isfinite(w_vec)
    rad1_vec = rad1_vec[valid]
    rad2_vec = rad2_vec[valid]
    inc_vec = inc_vec[valid]
    w_vec = w_vec[valid]

    ra_grid = np.asarray(settings["RA_grid"], dtype=float).reshape(-1)

    if rad1_vec.size == 0 or rad2_vec.size == 0:
        return {"MAP": np.nan, "flag": np.nan}

    prior_opt = settings.get("prior")
    if prior_opt is not None:
        prior = np.asarray(prior_opt, dtype=float).reshape(-1)
        if prior.size != ra_grid.size:
            raise ValueError("settings['prior'] must have the same length as RA_grid.")
        use_prior = True
    else:
        prior = np.empty(0, dtype=float)
        use_prior = False

    ra_map, flag = fit_local_scattering_curve_2rad_wbeta_numba(
        rad1_vec,
        rad2_vec,
        inc_vec,
        w_vec,
        ra_grid,
        float(settings["sigmaFloor"]),
        float(settings["alpha"]),
        float(settings["RA0"]),
        float(settings["s"]),
        prior,
        use_prior,
        mu1_fit.model_kind,
        np.asarray(mu1_fit.params, dtype=float),
        sigma1_fit.model_kind,
        np.asarray(sigma1_fit.params, dtype=float),
        mu2_fit.model_kind,
        np.asarray(mu2_fit.params, dtype=float),
        sigma2_fit.model_kind,
        np.asarray(sigma2_fit.params, dtype=float),
    )
    return {"MAP": float(ra_map), "flag": float(flag)}


def invert_ra_2rad_first_pass_spatial(rad1: Array, rad2: Array, ra_map: Array, incidence_map: Array,
                                      firstpass_settings: dict[str, Any], n_workers: int | None = None) -> tuple[Array, Array]:
    """run the first-pass local patch inversion across the crater scene"""
    nrows, ncols = rad1.shape
    half_win = int(firstpass_settings["halfWin"])
    sigma_dist = float(firstpass_settings["sigmaDist"])
    rad1_mu_fit = firstpass_settings["Rad1muFit"]
    rad1_iqr_fit = firstpass_settings["Rad1iqrFit"]
    rad2_mu_fit = firstpass_settings["Rad2muFit"]
    rad2_iqr_fit = firstpass_settings["Rad2iqrFit"]

    # keep fortran order so the pixel indexing matches the matlab translation
    rad1_vec = rad1.reshape(-1, order="F")
    rad2_vec = rad2.reshape(-1, order="F")
    inc_vec = incidence_map.reshape(-1, order="F")
    ra_vec = ra_map.reshape(-1, order="F")

    valid_pixel = np.isfinite(rad1_vec) & np.isfinite(rad2_vec) & np.isfinite(inc_vec) & np.isfinite(ra_vec)
    valid_idx = np.flatnonzero(valid_pixel)

    def work(idx: int) -> tuple[float, float]:
        r, c = np.unravel_index(idx, rad1.shape, order="F")

        r_min = max(0, r - half_win)
        r_max = min(nrows - 1, r + half_win)
        c_min = max(0, c - half_win)
        c_max = min(ncols - 1, c + half_win)

        rr, cc = np.meshgrid(np.arange(r_min, r_max + 1), np.arange(c_min, c_max + 1), indexing="ij")
        dist_pix = np.sqrt((rr - r) ** 2 + (cc - c) ** 2)
        w_spatial = np.exp(-0.5 * (dist_pix / sigma_dist) ** 2)

        rad1_patch = rad1[r_min : r_max + 1, c_min : c_max + 1]
        rad2_patch = rad2[r_min : r_max + 1, c_min : c_max + 1]
        ra_patch = ra_map[r_min : r_max + 1, c_min : c_max + 1]
        inc_patch = incidence_map[r_min : r_max + 1, c_min : c_max + 1]

        rad1_local = rad1_patch.reshape(-1, order="F")
        rad2_local = rad2_patch.reshape(-1, order="F")
        ra_local = ra_patch.reshape(-1, order="F")
        inc_local = inc_patch.reshape(-1, order="F")
        w_local = w_spatial.reshape(-1, order="F")

        valid = np.isfinite(rad1_local) & np.isfinite(rad2_local) & np.isfinite(ra_local) & np.isfinite(inc_local)
        if not np.any(valid):
            return (np.nan, np.nan)

        rad1_local = rad1_local[valid]
        rad2_local = rad2_local[valid]
        inc_local = inc_local[valid]
        w_local = w_local[valid]

        nobs = rad1_local.size

        local_settings = dict(firstpass_settings)
        # empirical alpha relation
        local_settings["alpha"] = 0.034 * nobs + 0.106

        ra_pred = fit_local_scattering_curve_2rad_wbeta(
            rad1_local,
            rad2_local,
            inc_local,
            rad1_mu_fit,
            rad1_iqr_fit,
            rad2_mu_fit,
            rad2_iqr_fit,
            w_local,
            local_settings,
        )
        return (float(ra_pred["MAP"]), float(ra_pred["flag"]))

    if n_workers == 1:
        packed = [work(idx) for idx in valid_idx]
    else:
        with ThreadPoolExecutor(max_workers=n_workers) as pool:
            packed = list(pool.map(work, valid_idx))

    ra_pred_temp = np.full(valid_idx.shape, np.nan, dtype=float)
    flag_temp = np.full(valid_idx.shape, np.nan, dtype=float)
    for i, (ra_pred, flag) in enumerate(packed):
        ra_pred_temp[i] = ra_pred
        flag_temp[i] = flag

    ra_pred_flat = np.full(rad1.size, np.nan, dtype=float)
    flag_flat = np.full(rad1.size, np.nan, dtype=float)
    ra_pred_flat[valid_idx] = ra_pred_temp
    flag_flat[valid_idx] = flag_temp

    return (
        ra_pred_flat.reshape(rad1.shape, order="F"),
        flag_flat.reshape(rad1.shape, order="F"),
    )


def invert_ra_from_rad_second_2rad(rad1: Array, rad2: Array, inc: Array, mu1_fit: ModelFit, sigma1_fit: ModelFit,
                                   mu2_fit: ModelFit, sigma2_fit: ModelFit, secondpass_settings: dict[str, Any]) -> dict[str, Any]:
    """invert one second-pass posterior with the hardcoded joint gaussian covariance model"""
    secondpass_settings = dict(secondpass_settings)

    if "covModel" not in secondpass_settings or secondpass_settings["covModel"] is None:
        raise ValueError("secondpass_settings['covModel'] must be provided for the joint gaussian mode.")

    prior = np.asarray(secondpass_settings["prior"], dtype=float).reshape(-1)

    rad1 = np.asarray(rad1, dtype=float).reshape(-1)
    rad2 = np.asarray(rad2, dtype=float).reshape(-1)
    inc = np.asarray(inc, dtype=float).reshape(-1)

    if not (rad1.size == rad2.size == inc.size):
        raise ValueError("rad1, rad2, and inc must have the same number of elements.")

    valid_obs = np.isfinite(rad1) & np.isfinite(rad2) & np.isfinite(inc)
    rad1 = rad1[valid_obs]
    rad2 = rad2[valid_obs]
    inc = inc[valid_obs]

    if rad1.size == 0:
        return {"MAP": np.nan, "mean": np.nan, "med": np.nan, "low": np.nan, "high": np.nan}

    ra_grid = np.asarray(secondpass_settings["RA_grid"], dtype=float).reshape(-1)
    if prior.size != ra_grid.size:
        raise ValueError("secondpass_settings['prior'] must have the same length as RA_grid.")

    cov_model = secondpass_settings["covModel"]
    covariance_mode = str(secondpass_settings["covarianceMode"]).lower()
    rho_eta_params = np.asarray(cov_model["rhoEtaParams"], dtype=float).reshape(-1)
    rho_feature_stats = np.asarray(cov_model["rhoFeatureStats"], dtype=float).reshape(-1)
    ra_edges = np.asarray(cov_model["RA_edges"], dtype=float).reshape(-1)
    inc_edges = np.asarray(cov_model["Inc_edges"], dtype=float).reshape(-1)
    rho_ns = np.asarray(cov_model["rho_ns"], dtype=float)

    has_neff = "Neff" in secondpass_settings and secondpass_settings["Neff"] is not None
    neff_value = float(secondpass_settings["Neff"]) if has_neff else 0.0

    if covariance_mode == "smooth":
        ra_map, ra_mean, ra_med, ra_low, ra_high = invert_ra_from_rad_second_2rad_joint_numba(
            rad1,
            rad2,
            inc,
            prior,
            ra_grid,
            has_neff,
            neff_value,
            mu1_fit.model_kind,
            np.asarray(mu1_fit.params, dtype=float),
            sigma1_fit.model_kind,
            np.asarray(sigma1_fit.params, dtype=float),
            mu2_fit.model_kind,
            np.asarray(mu2_fit.params, dtype=float),
            sigma2_fit.model_kind,
            np.asarray(sigma2_fit.params, dtype=float),
            rho_eta_params,
            rho_feature_stats,
        )
    elif covariance_mode == "lookup":
        ra_map, ra_mean, ra_med, ra_low, ra_high = invert_ra_from_rad_second_2rad_joint_lookup_numba(
            rad1,
            rad2,
            inc,
            prior,
            ra_grid,
            has_neff,
            neff_value,
            mu1_fit.model_kind,
            np.asarray(mu1_fit.params, dtype=float),
            sigma1_fit.model_kind,
            np.asarray(sigma1_fit.params, dtype=float),
            mu2_fit.model_kind,
            np.asarray(mu2_fit.params, dtype=float),
            sigma2_fit.model_kind,
            np.asarray(sigma2_fit.params, dtype=float),
            ra_edges,
            inc_edges,
            rho_ns,
        )
    else:
        raise ValueError(f"Unknown covarianceMode {covariance_mode!r}. Use 'smooth' or 'lookup'.")

    return {
        "MAP": float(ra_map),
        "mean": float(ra_mean),
        "med": float(ra_med),
        "low": float(ra_low),
        "high": float(ra_high),
        "RA_grid": ra_grid,
        "post1": None,
        "post2": None,
        "post": None,
    }


def invert_ra_2rad_second_pass_spatial(rad1: Array, rad2: Array, inc_obs: Array, ra_first: Array,
                                       secondpass_settings: dict[str, Any], n_workers: int | None = None) -> dict[str, Array]:
    """run the per-pixel second-pass posterior update over the full map"""
    secondpass_settings = dict(secondpass_settings)

    mu1_fit = secondpass_settings["Rad1muFit"]
    sigma1_fit = secondpass_settings["Rad1iqrFit"]
    mu2_fit = secondpass_settings["Rad2muFit"]
    sigma2_fit = secondpass_settings["Rad2iqrFit"]

    window_size = int(secondpass_settings["windowSize"])
    mode = str(secondpass_settings["spatialPriorMode"]).lower()
    if mode == "direct":
        ra_local = ra_first
    elif mode == "boxmedian":
        ra_local = boxcar_median(ra_first, window_size)
    elif mode == "movmedian2":
        ra_local = moving_median_1d(ra_first, window_size, axis=0)
        ra_local = moving_median_1d(ra_local, window_size, axis=1)
    else:
        raise ValueError("Unknown secondpass_settings['spatialPriorMode']. Use 'direct', 'boxmedian', or 'movmedian2'.")

    ra_grid = np.asarray(secondpass_settings["RA_grid"], dtype=float).reshape(-1)
    pixel_settings_template = dict(secondpass_settings)
    pixel_settings_template["RA_grid"] = ra_grid
    frac_spatial = float(secondpass_settings["sigmaSpatial"])
    min_sigma_spatial = float(secondpass_settings["minSigmaSpatial"])
    max_sigma_spatial = float(secondpass_settings["maxSigmaSpatial"])

    rad1_vec = rad1.reshape(-1, order="F")
    rad2_vec = rad2.reshape(-1, order="F")
    inc_vec = inc_obs.reshape(-1, order="F")
    ra0_vec = ra_local.reshape(-1, order="F")

    valid_pixel = np.isfinite(rad1_vec) & np.isfinite(rad2_vec) & np.isfinite(inc_vec) & np.isfinite(ra0_vec)
    valid_idx = np.flatnonzero(valid_pixel)

    def work(idx: int) -> tuple[float, float, float, float, float]:
        rad1_value = rad1_vec[idx]
        rad2_value = rad2_vec[idx]
        inc_value = inc_vec[idx]
        ra0 = ra0_vec[idx]

        # widen or narrow the prior in proportion to the first-pass estimate
        sigma_spatial = frac_spatial * ra0
        if not np.isfinite(sigma_spatial) or sigma_spatial <= 0.0:
            return (np.nan, np.nan, np.nan, np.nan, np.nan)

        sigma_spatial = max(sigma_spatial, min_sigma_spatial)
        sigma_spatial = min(sigma_spatial, max_sigma_spatial)

        prior = normpdf(ra_grid, ra0, sigma_spatial)
        prior = np.where(np.isfinite(prior) & (prior >= 0.0), prior, 0.0)
        if np.all(prior == 0.0):
            return (np.nan, np.nan, np.nan, np.nan, np.nan)
        prior = prior / np.trapezoid(prior, ra_grid)

        pixel_settings = dict(pixel_settings_template)
        pixel_settings["prior"] = prior

        ra_pred = invert_ra_from_rad_second_2rad(
            rad1_value,
            rad2_value,
            inc_value,
            mu1_fit,
            sigma1_fit,
            mu2_fit,
            sigma2_fit,
            pixel_settings,
        )
        return (
            float(ra_pred["MAP"]),
            float(ra_pred["mean"]),
            float(ra_pred["med"]),
            float(ra_pred["low"]),
            float(ra_pred["high"]),
        )

    if n_workers == 1:
        packed = [work(idx) for idx in valid_idx]
    else:
        with ThreadPoolExecutor(max_workers=n_workers) as pool:
            packed = list(pool.map(work, valid_idx))

    ra_second_vec = np.full(valid_idx.shape, np.nan, dtype=float)
    ra_mean_vec = np.full(valid_idx.shape, np.nan, dtype=float)
    ra_med_vec = np.full(valid_idx.shape, np.nan, dtype=float)
    ra_low_vec = np.full(valid_idx.shape, np.nan, dtype=float)
    ra_high_vec = np.full(valid_idx.shape, np.nan, dtype=float)

    for i, (ra_second, ra_mean, ra_med, ra_low, ra_high) in enumerate(packed):
        ra_second_vec[i] = ra_second
        ra_mean_vec[i] = ra_mean
        ra_med_vec[i] = ra_med
        ra_low_vec[i] = ra_low
        ra_high_vec[i] = ra_high

    def scatter(values: Array) -> Array:
        flat = np.full(rad1.size, np.nan, dtype=float)
        flat[valid_idx] = values
        return flat.reshape(rad1.shape, order="F")

    return {
        "second": scatter(ra_second_vec),
        "mean": scatter(ra_mean_vec),
        "med": scatter(ra_med_vec),
        "low": scatter(ra_low_vec),
        "high": scatter(ra_high_vec),
    }


def build_default_pipeline_options(models: dict[str, ModelFit] | None = None,
                                   cov_model: dict[str, Any] | None = None) -> dict[str, dict[str, Any]]:
    """build the pipeline settings from the hardcoded variables"""
    if models is None:
        models = load_parallel_scattering_models()
    if cov_model is None:
        cov_model = load_covariance_model()

    ra_grid = default_ra_grid(ra_min=ra_grid_min, ra_max=ra_grid_max, n_grid_points=ra_grid_points)
    firstpass_half_win = firstpass_window_size // 2
    firstpass_sigma_dist = firstpass_window_size / 3.0

    firstpass_settings: dict[str, Any] = {
        "s": firstpass_s,
        "RA_grid": ra_grid,
        "sigmaFloor": firstpass_sigma_floor,
        "RA0": firstpass_ra0,
        "halfWin": firstpass_half_win,
        "sigmaDist": firstpass_sigma_dist,
        "Rad1muFit": models["Rad1muFit"],
        "Rad1iqrFit": models["Rad1iqrFit"],
        "Rad2muFit": models["Rad2muFit"],
        "Rad2iqrFit": models["Rad2iqrFit"],
    }

    secondpass_settings = dict(firstpass_settings)
    secondpass_settings["windowSize"] = secondpass_window_size
    secondpass_settings["spatialPriorMode"] = secondpass_spatial_prior_mode
    secondpass_settings["covarianceMode"] = secondpass_covariance_mode
    secondpass_settings["sigmaSpatial"] = secondpass_sigma_spatial
    secondpass_settings["minSigmaSpatial"] = secondpass_min_sigma_spatial
    secondpass_settings["maxSigmaSpatial"] = secondpass_max_sigma_spatial
    secondpass_settings["covModel"] = cov_model

    eval_settings: dict[str, Any] = {
        "regimeEdges": list(eval_regime_edges),
        "regimeNames": list(eval_regime_names),
        "smoothWindow": eval_smooth_window,
        "mapMode": eval_map_mode,
        "report": eval_report,
    }

    return {"firstpass": firstpass_settings, "secondpass": secondpass_settings, "eval": eval_settings}


def predict_ra_from_preprocessed_with_firstpass(s: dict[str, Array], ra_pred_map: Array, flag_map: Array,
        secondpass_options: dict[str, Any], n_workers: int | None = None) -> dict[str, Any]:
    """reuse a cached first pass and run only the second pass prediction"""
    ra_pred_maps = invert_ra_2rad_second_pass_spatial(
        s["Rad1"],
        s["Rad2"],
        s["IncidenceMap"],
        ra_pred_map,
        secondpass_options,
        n_workers=n_workers,
    )
    ra_pred = ra_pred_maps["med"]
    ra_truth = s.get("ra")

    result = {
        "S": s,
        "RA_pred_MAP": ra_pred_map,
        "FlagMap": flag_map,
        "RApredMaps": ra_pred_maps,
        "RApred": ra_pred,
    }

    if ra_truth is not None:
        # always attach the continuous sigma offset when diviner RA is available
        result["sigma_map"] = make_sigma_map(
            ra_pred_maps["med"],
            ra_pred_maps["low"],
            ra_pred_maps["high"],
            ra_truth,
        )
        result["sigma_envelope_map"] = compute_sigma_envelope(
            ra_pred_maps["med"],
            ra_pred_maps["low"],
            ra_pred_maps["high"],
            ra_truth,
            max_sigma=5,
        )

    return result


def predict_ra_from_preprocessed(s: dict[str, Array], firstpass_options: dict[str, Any],
                                 secondpass_options: dict[str, Any], n_workers: int | None = None) -> dict[str, Any]:
    """run the full two-pass v2 prediction from preprocessed inputs"""
    ra_pred_map, flag_map = invert_ra_2rad_first_pass_spatial(
        s["Rad1"],
        s["Rad2"],
        s["RAMap"],
        s["IncidenceMap"],
        firstpass_options,
        n_workers=n_workers,
    )
    return predict_ra_from_preprocessed_with_firstpass(
        s,
        ra_pred_map,
        flag_map,
        secondpass_options,
        n_workers=n_workers,
    )


def driver_pred_ra_from_radar(data_path: str | Path, n_workers: int | None = None,
                              compare_reference: bool | None = None) -> dict[str, Any]:
    """top-level entry point for the translated v2 crater test run"""
    models = load_parallel_scattering_models()
    cov_model = load_covariance_model()
    data = load_analysis_dataset(data_path, list(ANALYSIS_VARIABLES))
    s = preprocess_minirf_data(data)
    pipeline_options = build_default_pipeline_options(models, cov_model)
    if compare_reference is None:
        compare_reference = compare_saved_reference
    result = predict_ra_from_preprocessed(
        s,
        pipeline_options["firstpass"],
        pipeline_options["secondpass"],
        n_workers=n_workers,
    )

    eval_options = dict(pipeline_options["eval"])
    eval_options["RA_low"] = result["RApredMaps"]["low"]
    eval_options["RA_high"] = result["RApredMaps"]["high"]
    metrics_cpr = evaluate_ra_prediction(s["ra"], result["RApred"], eval_options)
    result["metrics_cpr"] = metrics_cpr

    reference = None
    if compare_reference and reference_path.exists():
        reference = np.load(reference_path)
        mean_diff = float(np.nanmean(reference - result["RApred"]))
        result["reference_mean_diff"] = mean_diff
        print(f"\nmean difference to saved v2 reference: {mean_diff:g}")

    map_plot_path = make_ra_map_sigma_plot(result["RApred"], result.get("sigma_map"), map_plot_output_path)
    hist_plot_path = make_ra_histogram_plot(s["ra"], result["RApredMaps"]["med"], result["RApredMaps"]["mean"], hist_plot_output_path)
    result["map_plot_path"] = map_plot_path
    result["hist_plot_path"] = hist_plot_path
    print(f"Saved RA map and sigma map plot to {map_plot_path}")
    print(f"Saved RA histogram plot to {hist_plot_path}")

    return result


if __name__ == "__main__":
    driver_pred_ra_from_radar(data_path, n_workers=workers)
