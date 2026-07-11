#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Diagnostics for CPR-G residual covariance in the v2 workflow
"""

from pathlib import Path
from typing import Any
import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse


Array = np.ndarray
ANALYSIS_VARIABLES = ["miniRF_1", "miniRF_2", "miniRF_3", "miniRF_4", "miniRF_5", "miniRF_6", "ra"]

# path to the input dataset
data_path = Path(__file__).resolve().parents[1] / "data/GiordanoBruno_analysis.csv"

# output path for the diagnostic figure
output_plot_path = Path(__file__).resolve().parents[1] / "output/cpr_green_covariance_diagnostics.png"

# minimum scale floor used to standardize residuals
sigma_floor = 0.02

# optional incidence bounds applied during preprocessing
min_incidence = None
max_incidence = None

# number of RA bins for the slope diagnostic
n_ra_bins = 16

# minimum valid points required per RA bin
min_points_per_bin = 500

# hexbin resolution for the residual cloud
hexbin_gridsize = 90

# maximum absolute z value shown on the residual plot axes
residual_axis_limit = 6.0

# maximum number of residual points used for the smooth rho fit
fit_sample_size = 200000

# holdout fraction for validation during the smooth rho fit
validation_fraction = 0.2

# deterministic seed for the fit subset and split
fit_random_seed = 0

# ridge penalty used to regularize the eta coefficients
fit_ridge_lambda = 0.1

# heuristic threshold for calling the incidence effect weak
incidence_range_threshold = 0.05

# heuristic threshold for calling the incidence effect weak
validation_nll_threshold = 1e-3


def cosd(x: float | Array) -> Array:
    """Return the cosine of angles expressed in degrees."""
    return np.cos(np.deg2rad(x))


def sigmoid_cosd_exp_ra(params: Array, inc: float | Array, ra: float | Array) -> Array:
    """Evaluate the saved sigmoid-plus-cosine model with exponential RA incidence scaling."""
    p = np.asarray(params, dtype=float)
    inc = np.asarray(inc, dtype=float)
    ra = np.asarray(ra, dtype=float)
    inc_term = np.power(cosd(inc), p[4] * np.exp(p[5] * ra))
    sigmoid = p[0] + (p[1] / (1.0 + np.exp(-p[2] * (np.log10(ra) + p[3]))))
    linear = p[6] + p[7] * ra
    return sigmoid * inc_term + linear


def sigmoid_cosd_linear_ra(params: Array, inc: float | Array, ra: float | Array) -> Array:
    """Evaluate the saved sigmoid-plus-cosine model with linear RA incidence scaling."""
    p = np.asarray(params, dtype=float)
    inc = np.asarray(inc, dtype=float)
    ra = np.asarray(ra, dtype=float)
    inc_term = np.power(cosd(inc), p[4] + p[5] * ra)
    sigmoid = p[0] + (p[1] / (1.0 + np.exp(-p[2] * (np.log10(ra) + p[3]))))
    linear = p[6] + p[7] * ra
    return sigmoid * inc_term + linear


def load_analysis_csv(csv_path: str | Path, variable_names: list[str] | None = None) -> dict[str, Array]:
    """Rebuild 2D analysis arrays from the flattened CSV export format."""
    csv_path = Path(csv_path).resolve()
    if variable_names is None:
        variable_names = ANALYSIS_VARIABLES

    required_columns = ["row", "col", *variable_names]
    with csv_path.open("r", encoding="utf-8") as handle:
        header = handle.readline().strip().split(",")

    missing_columns = [name for name in required_columns if name not in header]
    if missing_columns:
        raise ValueError(f"CSV file is missing required columns: {missing_columns}")

    usecols = [header.index(name) for name in required_columns]
    raw = np.loadtxt(csv_path, delimiter=",", skiprows=1, usecols=usecols)
    if raw.ndim == 1:
        raw = raw[np.newaxis, :]

    row_idx = raw[:, 0].astype(int)
    col_idx = raw[:, 1].astype(int)
    if row_idx.size == 0 or col_idx.size == 0:
        raise ValueError("CSV file does not contain any data rows.")

    nrows = int(np.max(row_idx))
    ncols = int(np.max(col_idx))
    if raw.shape[0] != nrows * ncols:
        raise ValueError("CSV data do not form a complete rectangular grid.")

    data: dict[str, Array] = {}
    for offset, name in enumerate(variable_names, start=2):
        data[name] = raw[:, offset].reshape((nrows, ncols), order="F")

    return data


def load_analysis_dataset(data_path: str | Path, variable_names: list[str] | None = None) -> dict[str, Array]:
    """Load the analysis dataset from CSV."""
    data_path = Path(data_path)
    if data_path.suffix.lower() != ".csv":
        raise ValueError(f"Unsupported data file type {data_path.suffix!r}. Expected .csv.")
    return load_analysis_csv(data_path, variable_names)


def preprocess_minirf_data(data: dict[str, Array]) -> dict[str, Array]:
    """Match the Mini-RF preprocessing used by the translated workflow."""
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

    mask = (s["num"] == 1) & (s["ra"] > 0.01) & np.isfinite(s["cpr"]) & np.isfinite(s["inc"])

    if min_incidence is not None:
        mask &= s["inc"] > float(min_incidence)

    if max_incidence is not None:
        mask &= s["inc"] < float(max_incidence)

    for key in ("cpr", "g", "inc", "ra"):
        s[key][~mask] = np.nan

    s["Rad1"] = s["cpr"]
    s["Rad2"] = s["g"]
    s["IncidenceMap"] = s["inc"]
    s["RAMap"] = s["ra"]
    return s


def load_parallel_scattering_models() -> dict[str, dict[str, Any]]:
    """Return the saved scattering-model fits used by the translated workflow."""
    return {
        "Rad1muFit": {
            "model_fun": sigmoid_cosd_exp_ra,
            "params": np.array(
                [-0.837883, 0.585569, 11.1532, 0.147126, 1.16211, 0.286385, 0.967473, -0.0141565],
                dtype=float,
            ),
        },
        "Rad1iqrFit": {
            "model_fun": sigmoid_cosd_linear_ra,
            "params": np.array(
                [0.0870258, 0.137405, 28.9929, 0.293797, 0.951332, 0.4565, 0.0716333, 0.00852198],
                dtype=float,
            ),
        },
        "Rad2muFit": {
            "model_fun": sigmoid_cosd_exp_ra,
            "params": np.array(
                [0.104786, 0.472109, 7.53962, 0.145583, 1.6987, -0.00172229, 0.186198, 0.00914963],
                dtype=float,
            ),
        },
        "Rad2iqrFit": {
            "model_fun": sigmoid_cosd_exp_ra,
            "params": np.array(
                [0.0958137, 0.259142, 17.003, 0.271329, 3.08496, -0.058395, 0.0377035, 0.00164396],
                dtype=float,
            ),
        },
    }


def _safe_linear_slope(x: Array, y: Array) -> float:
    """Return the least-squares slope of y on x for finite vectors."""
    x = np.asarray(x, dtype=float).reshape(-1)
    y = np.asarray(y, dtype=float).reshape(-1)
    valid = np.isfinite(x) & np.isfinite(y)
    x = x[valid]
    y = y[valid]
    if x.size < 2:
        return float("nan")

    x_centered = x - np.mean(x)
    denom = float(np.dot(x_centered, x_centered))
    if denom <= 0.0 or not np.isfinite(denom):
        return float("nan")

    y_centered = y - np.mean(y)
    return float(np.dot(x_centered, y_centered) / denom)


def _safe_corr(x: Array, y: Array) -> float:
    """Return the Pearson correlation for finite vectors."""
    x = np.asarray(x, dtype=float).reshape(-1)
    y = np.asarray(y, dtype=float).reshape(-1)
    valid = np.isfinite(x) & np.isfinite(y)
    x = x[valid]
    y = y[valid]
    if x.size < 2:
        return float("nan")
    if np.allclose(x, x[0]) or np.allclose(y, y[0]):
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def _add_covariance_ellipse(ax: plt.Axes, x: Array, y: Array, n_std: float, color: str, label: str) -> None:
    """Overlay one covariance ellipse on an axes."""
    x = np.asarray(x, dtype=float).reshape(-1)
    y = np.asarray(y, dtype=float).reshape(-1)
    valid = np.isfinite(x) & np.isfinite(y)
    x = x[valid]
    y = y[valid]
    if x.size < 2:
        return

    cov = np.cov(x, y)
    if cov.shape != (2, 2) or not np.all(np.isfinite(cov)):
        return

    evals, evecs = np.linalg.eigh(cov)
    order = np.argsort(evals)[::-1]
    evals = evals[order]
    evecs = evecs[:, order]
    if np.any(evals < 0.0):
        return

    angle = float(np.degrees(np.arctan2(evecs[1, 0], evecs[0, 0])))
    width = float(2.0 * n_std * np.sqrt(evals[0]))
    height = float(2.0 * n_std * np.sqrt(evals[1]))
    ellipse = Ellipse(
        xy=(float(np.mean(x)), float(np.mean(y))),
        width=width,
        height=height,
        angle=angle,
        facecolor="none",
        edgecolor=color,
        linewidth=1.8,
        alpha=0.95,
        label=label,
    )
    ax.add_patch(ellipse)


def _compute_standardized_residuals(s: dict[str, Array], sigma_floor: float) -> dict[str, Array]:
    """Compute raw and standardized CPR/Green residual diagnostics."""
    models = load_parallel_scattering_models()

    rad1 = np.asarray(s["Rad1"], dtype=float)
    rad2 = np.asarray(s["Rad2"], dtype=float)
    inc = np.asarray(s["IncidenceMap"], dtype=float)
    ra = np.asarray(s["RAMap"], dtype=float)

    valid = np.isfinite(rad1) & np.isfinite(rad2) & np.isfinite(inc) & np.isfinite(ra)
    rad1_v = rad1[valid]
    rad2_v = rad2[valid]
    inc_v = inc[valid]
    ra_v = ra[valid]

    mu1 = models["Rad1muFit"]["model_fun"](models["Rad1muFit"]["params"], inc_v, ra_v)
    sigma1 = models["Rad1iqrFit"]["model_fun"](models["Rad1iqrFit"]["params"], inc_v, ra_v) / 1.349
    sigma1 = np.maximum(np.asarray(sigma1, dtype=float), sigma_floor)

    mu2 = models["Rad2muFit"]["model_fun"](models["Rad2muFit"]["params"], inc_v, ra_v)
    sigma2 = models["Rad2iqrFit"]["model_fun"](models["Rad2iqrFit"]["params"], inc_v, ra_v) / 1.349
    sigma2 = np.maximum(np.asarray(sigma2, dtype=float), sigma_floor)

    # these z scores are the inputs to the joint residual covariance model
    resid1 = rad1_v - np.asarray(mu1, dtype=float)
    resid2 = rad2_v - np.asarray(mu2, dtype=float)
    z1 = resid1 / sigma1
    z2 = resid2 / sigma2

    return {
        "ra": ra_v,
        "inc": inc_v,
        "rad1": rad1_v,
        "rad2": rad2_v,
        "mu1": np.asarray(mu1, dtype=float),
        "mu2": np.asarray(mu2, dtype=float),
        "sigma1": sigma1,
        "sigma2": sigma2,
        "resid1": resid1,
        "resid2": resid2,
        "z1": z1,
        "z2": z2,
    }


def _build_rho_design(ra: Array, inc: Array, mode: str, feature_stats: dict[str, float] | None = None) -> tuple[Array, dict[str, float]]:
    """Build the low-order eta design matrix for one smooth rho model."""
    ra = np.asarray(ra, dtype=float).reshape(-1)
    inc = np.asarray(inc, dtype=float).reshape(-1)
    log_ra = np.log10(ra)

    if feature_stats is None:
        ra_center = float(np.mean(log_ra))
        ra_scale = float(np.std(log_ra))
        inc_center = float(np.mean(inc))
        inc_scale = float(np.std(inc))
        if not np.isfinite(ra_scale) or ra_scale <= 0.0:
            ra_scale = 1.0
        if not np.isfinite(inc_scale) or inc_scale <= 0.0:
            inc_scale = 1.0
        feature_stats = {
            "ra_center": ra_center,
            "ra_scale": ra_scale,
            "inc_center": inc_center,
            "inc_scale": inc_scale,
        }

    x_ra = (log_ra - feature_stats["ra_center"]) / feature_stats["ra_scale"]
    x_inc = (inc - feature_stats["inc_center"]) / feature_stats["inc_scale"]

    if mode == "constant":
        design = np.ones((ra.size, 1), dtype=float)
    elif mode == "ra":
        design = np.column_stack([np.ones_like(x_ra), x_ra, x_ra**2])
    elif mode == "ra_inc":
        design = np.column_stack([np.ones_like(x_ra), x_ra, x_ra**2, x_inc, x_inc**2, x_ra * x_inc])
    else:
        raise ValueError(f"Unknown smooth rho mode {mode!r}.")

    return np.asarray(design, dtype=float), feature_stats


def _predict_smooth_rho(fit_result: dict[str, Any], ra: Array, inc: Array) -> Array:
    """Evaluate one fitted smooth rho model."""
    design, _ = _build_rho_design(ra, inc, fit_result["mode"], fit_result["feature_stats"])
    rho = np.tanh(design @ np.asarray(fit_result["params"], dtype=float))
    return np.clip(np.asarray(rho, dtype=float), -0.999, 0.999)


def _rho_negloglik(params: Array, design: Array, z1: Array, z2: Array, ridge_lambda: float) -> float:
    """Return the mean negative log-likelihood for a tanh-linked rho model."""
    params = np.asarray(params, dtype=float)
    rho = np.tanh(np.asarray(design, dtype=float) @ params)
    rho = np.clip(rho, -0.999, 0.999)
    one_minus_rho2 = np.maximum(1.0 - rho**2, 1e-12)
    quad = (z1**2 - 2.0 * rho * z1 * z2 + z2**2) / one_minus_rho2
    nll = 0.5 * np.log(one_minus_rho2) + 0.5 * quad
    penalty = float(ridge_lambda) * float(np.sum(params[1:] ** 2))
    return float(np.mean(nll) + penalty)


def _fit_smooth_rho_model(ra_train: Array, inc_train: Array,
                          z1_train: Array, z2_train: Array,
                          ra_val: Array, inc_val: Array,
                          z1_val: Array, z2_val: Array,
                          mode: str, ridge_lambda: float) -> dict[str, Any]:
    """Fit one smooth rho = tanh(eta) model and score it on a holdout split."""
    design_train, feature_stats = _build_rho_design(ra_train, inc_train, mode)
    design_val, _ = _build_rho_design(ra_val, inc_val, mode, feature_stats)

    init_rho = float(np.clip(_safe_corr(z1_train, z2_train), -0.95, 0.95))
    params0 = np.zeros(design_train.shape[1], dtype=float)
    params0[0] = float(np.arctanh(init_rho))

    result = minimize(
        _rho_negloglik,
        params0,
        args=(design_train, z1_train, z2_train, ridge_lambda),
        method="L-BFGS-B",
    )

    params = np.asarray(result.x, dtype=float)
    train_nll = _rho_negloglik(params, design_train, z1_train, z2_train, 0.0)
    val_nll = _rho_negloglik(params, design_val, z1_val, z2_val, 0.0)

    return {
        "mode": mode,
        "params": params,
        "feature_stats": feature_stats,
        "success": bool(result.success),
        "message": str(result.message),
        "n_params": int(params.size),
        "ridge_lambda": float(ridge_lambda),
        "init_rho": init_rho,
        "train_nll": float(train_nll),
        "val_nll": float(val_nll),
    }


def _fit_smooth_rho_models(diag: dict[str, Array]) -> dict[str, Any]:
    """Fit constant, RA-only, and RA+incidence tanh-linked rho models."""
    ra = np.asarray(diag["ra"], dtype=float).reshape(-1)
    inc = np.asarray(diag["inc"], dtype=float).reshape(-1)
    z1 = np.asarray(diag["z1"], dtype=float).reshape(-1)
    z2 = np.asarray(diag["z2"], dtype=float).reshape(-1)

    valid = np.isfinite(ra) & np.isfinite(inc) & np.isfinite(z1) & np.isfinite(z2)
    ra = ra[valid]
    inc = inc[valid]
    z1 = z1[valid]
    z2 = z2[valid]

    rng = np.random.default_rng(fit_random_seed)
    n_total = int(ra.size)
    n_sample = min(n_total, int(fit_sample_size))
    sample_idx = rng.permutation(n_total)[:n_sample]

    ra = ra[sample_idx]
    inc = inc[sample_idx]
    z1 = z1[sample_idx]
    z2 = z2[sample_idx]

    # keep the split deterministic so repeated runs report the same comparison
    n_val = int(round(validation_fraction * n_sample))
    n_val = min(max(n_val, 1), max(n_sample - 1, 1))

    val_idx = np.arange(n_val, dtype=int)
    train_idx = np.arange(n_val, n_sample, dtype=int)
    if train_idx.size == 0:
        train_idx = val_idx

    ra_train = ra[train_idx]
    inc_train = inc[train_idx]
    z1_train = z1[train_idx]
    z2_train = z2[train_idx]
    ra_val = ra[val_idx]
    inc_val = inc[val_idx]
    z1_val = z1[val_idx]
    z2_val = z2[val_idx]

    fit_constant = _fit_smooth_rho_model(
        ra_train,
        inc_train,
        z1_train,
        z2_train,
        ra_val,
        inc_val,
        z1_val,
        z2_val,
        "constant",
        fit_ridge_lambda,
    )
    fit_ra = _fit_smooth_rho_model(
        ra_train,
        inc_train,
        z1_train,
        z2_train,
        ra_val,
        inc_val,
        z1_val,
        z2_val,
        "ra",
        fit_ridge_lambda,
    )
    fit_ra_inc = _fit_smooth_rho_model(
        ra_train,
        inc_train,
        z1_train,
        z2_train,
        ra_val,
        inc_val,
        z1_val,
        z2_val,
        "ra_inc",
        fit_ridge_lambda,
    )

    ra_eval = np.geomspace(float(np.nanpercentile(ra, 1.0)), float(np.nanpercentile(ra, 99.0)), 160)
    inc_levels = np.asarray(np.nanpercentile(inc, [10.0, 50.0, 90.0]), dtype=float)
    inc_grid = np.linspace(float(np.nanpercentile(inc, 1.0)), float(np.nanpercentile(inc, 99.0)), 80)
    inc_mid = float(np.nanmedian(inc))

    constant_curve = _predict_smooth_rho(fit_constant, ra_eval, np.full(ra_eval.shape, inc_mid))
    ra_curve = _predict_smooth_rho(fit_ra, ra_eval, np.full(ra_eval.shape, inc_mid))
    ra_inc_curves = np.vstack(
        [_predict_smooth_rho(fit_ra_inc, ra_eval, np.full(ra_eval.shape, level)) for level in inc_levels]
    )
    rho_surface = np.vstack(
        [_predict_smooth_rho(fit_ra_inc, ra_eval, np.full(ra_eval.shape, inc_value)) for inc_value in inc_grid]
    ).T
    rho_range_by_ra = np.nanmax(rho_surface, axis=1) - np.nanmin(rho_surface, axis=1)

    # call the incidence effect weak only when both the fit gain and rho swing stay small
    delta_nll_ra = float(fit_constant["val_nll"] - fit_ra["val_nll"])
    delta_nll_inc = float(fit_ra["val_nll"] - fit_ra_inc["val_nll"])
    median_range = float(np.nanmedian(rho_range_by_ra))
    max_range = float(np.nanmax(rho_range_by_ra))
    incidence_effect_weak = bool(
        (delta_nll_inc < validation_nll_threshold) and (median_range < incidence_range_threshold)
    )

    return {
        "sample": {
            "n_total": n_total,
            "n_sample": int(n_sample),
            "n_train": int(train_idx.size),
            "n_val": int(val_idx.size),
        },
        "fits": {
            "constant": fit_constant,
            "ra": fit_ra,
            "ra_inc": fit_ra_inc,
        },
        "comparison": {
            "delta_nll_ra": delta_nll_ra,
            "delta_nll_inc": delta_nll_inc,
            "median_incidence_rho_range": median_range,
            "max_incidence_rho_range": max_range,
            "incidence_effect_weak": incidence_effect_weak,
            "selected_mode": "ra" if incidence_effect_weak else "ra_inc",
        },
        "curves": {
            "ra_eval": ra_eval,
            "inc_levels": inc_levels,
            "inc_grid": inc_grid,
            "constant_curve": constant_curve,
            "ra_curve": ra_curve,
            "ra_inc_curves": ra_inc_curves,
            "rho_range_by_ra": rho_range_by_ra,
        },
    }


def _compute_ra_binned_slopes(diag: dict[str, Array], n_bins: int, min_points_per_bin: int) -> dict[str, Array]:
    """Compute RA-binned slope diagnostics for raw and standardized CPR/Green relations."""
    ra = np.asarray(diag["ra"], dtype=float)
    rad1 = np.asarray(diag["rad1"], dtype=float)
    rad2 = np.asarray(diag["rad2"], dtype=float)
    z1 = np.asarray(diag["z1"], dtype=float)
    z2 = np.asarray(diag["z2"], dtype=float)

    finite_ra = ra[np.isfinite(ra)]
    ra_edges = np.quantile(finite_ra, np.linspace(0.0, 1.0, n_bins + 1))
    ra_edges[0] = float(np.min(finite_ra))
    ra_edges[-1] = float(np.max(finite_ra))

    centers: list[float] = []
    raw_slopes: list[float] = []
    residual_slopes: list[float] = []
    residual_rhos: list[float] = []
    counts: list[int] = []

    for i in range(n_bins):
        lower = ra_edges[i]
        upper = ra_edges[i + 1]
        if i == n_bins - 1:
            mask = (ra >= lower) & (ra <= upper)
        else:
            mask = (ra >= lower) & (ra < upper)

        count = int(np.sum(mask))
        if count < min_points_per_bin:
            continue

        centers.append(float(np.nanmedian(ra[mask])))
        raw_slopes.append(_safe_linear_slope(rad1[mask], rad2[mask]))
        residual_slopes.append(_safe_linear_slope(z1[mask], z2[mask]))
        residual_rhos.append(_safe_corr(z1[mask], z2[mask]))
        counts.append(count)

    return {
        "ra_center": np.asarray(centers, dtype=float),
        "raw_slope": np.asarray(raw_slopes, dtype=float),
        "residual_slope": np.asarray(residual_slopes, dtype=float),
        "residual_rho": np.asarray(residual_rhos, dtype=float),
        "count": np.asarray(counts, dtype=int),
    }


def make_covariance_diagnostic_plot(diag: dict[str, Array], binned: dict[str, Array],
                                    smooth_summary: dict[str, Any], output_path: str | Path) -> Path:
    """Render and save the residual, slope, and smooth-rho diagnostic plot."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    z1 = np.asarray(diag["z1"], dtype=float)
    z2 = np.asarray(diag["z2"], dtype=float)
    global_rho = _safe_corr(z1, z2)
    raw_slope_global = _safe_linear_slope(diag["rad1"], diag["rad2"])
    residual_slope_global = _safe_linear_slope(z1, z2)

    fig, axes = plt.subplots(2, 2, figsize=(14, 11), constrained_layout=True)
    ax_cloud = axes[0, 0]
    ax_slope = axes[0, 1]
    ax_rho = axes[1, 0]
    ax_inc = axes[1, 1]

    hb = ax_cloud.hexbin(
        z1,
        z2,
        gridsize=hexbin_gridsize,
        bins="log",
        mincnt=1,
        cmap="viridis",
    )
    fig.colorbar(hb, ax=ax_cloud, label="log10(count)")
    _add_covariance_ellipse(ax_cloud, z1, z2, n_std=1.0, color="white", label="1 sigma ellipse")
    _add_covariance_ellipse(ax_cloud, z1, z2, n_std=2.0, color="#ffb000", label="2 sigma ellipse")
    ax_cloud.axhline(0.0, color="white", linewidth=0.8, alpha=0.5)
    ax_cloud.axvline(0.0, color="white", linewidth=0.8, alpha=0.5)
    ax_cloud.set_xlim(-residual_axis_limit, residual_axis_limit)
    ax_cloud.set_ylim(-residual_axis_limit, residual_axis_limit)
    ax_cloud.set_xlabel("standardized CPR residual z1")
    ax_cloud.set_ylabel("standardized Green residual z2")
    ax_cloud.set_title(f"Standardized residual cloud\nrho={global_rho:.3f}, slope={residual_slope_global:.3f}")
    ax_cloud.legend(loc="upper right", frameon=True)

    ra_center = np.asarray(binned["ra_center"], dtype=float)
    raw_slope = np.asarray(binned["raw_slope"], dtype=float)
    residual_slope = np.asarray(binned["residual_slope"], dtype=float)
    residual_rho = np.asarray(binned["residual_rho"], dtype=float)

    ax_slope.plot(ra_center, raw_slope, marker="o", linewidth=2.0, color="#005f73", label="raw G-on-CPR slope")
    ax_slope.plot(
        ra_center,
        residual_slope,
        marker="s",
        linewidth=1.8,
        linestyle="--",
        color="#bb3e03",
        label="standardized residual slope",
    )
    ax_slope.set_xlabel("RA (%)")
    ax_slope.set_ylabel("slope")
    ax_slope.set_title(
        f"CPR-G relation by RA bin\nraw slope={raw_slope_global:.3f}, global residual slope={residual_slope_global:.3f}"
    )
    ax_slope.grid(True, alpha=0.25)

    ax_slope_rho = ax_slope.twinx()
    ax_slope_rho.plot(
        ra_center,
        residual_rho,
        marker="^",
        linewidth=1.5,
        linestyle=":",
        color="#9b2226",
        label="residual rho",
    )
    ax_slope_rho.set_ylabel("residual rho")
    ax_slope_rho.set_ylim(-1.0, 1.0)

    handles_left, labels_left = ax_slope.get_legend_handles_labels()
    handles_right, labels_right = ax_slope_rho.get_legend_handles_labels()
    ax_slope.legend(handles_left + handles_right, labels_left + labels_right, loc="best", frameon=True)

    curves = smooth_summary["curves"]
    comparison = smooth_summary["comparison"]
    fits = smooth_summary["fits"]

    ra_eval = np.asarray(curves["ra_eval"], dtype=float)
    inc_levels = np.asarray(curves["inc_levels"], dtype=float)
    constant_curve = np.asarray(curves["constant_curve"], dtype=float)
    ra_curve = np.asarray(curves["ra_curve"], dtype=float)
    ra_inc_curves = np.asarray(curves["ra_inc_curves"], dtype=float)
    rho_range_by_ra = np.asarray(curves["rho_range_by_ra"], dtype=float)

    ax_rho.plot(ra_eval, constant_curve, color="#6c757d", linewidth=1.5, linestyle=":", label="constant rho")
    ax_rho.plot(ra_eval, ra_curve, color="#005f73", linewidth=2.0, label="ra-only rho")
    rho_colors = ["#e76f51", "#f4a261", "#bc6c25"]
    for color, inc_level, rho_curve in zip(rho_colors, inc_levels, ra_inc_curves, strict=False):
        ax_rho.plot(
            ra_eval,
            rho_curve,
            color=color,
            linewidth=1.8,
            label=f"ra+inc rho at inc={inc_level:.1f} deg",
        )
    ax_rho.set_xscale("log")
    ax_rho.set_ylim(-1.0, 1.0)
    ax_rho.set_xlabel("RA (%)")
    ax_rho.set_ylabel("predicted rho")
    ax_rho.set_title(
        "smooth rho = tanh(eta) fits\n"
        f"val nll const={fits['constant']['val_nll']:.4f}, "
        f"ra={fits['ra']['val_nll']:.4f}, ra+inc={fits['ra_inc']['val_nll']:.4f}"
    )
    ax_rho.grid(True, alpha=0.25)
    ax_rho.legend(loc="best", frameon=True)

    ax_inc.plot(ra_eval, rho_range_by_ra, color="#9b2226", linewidth=2.0)
    ax_inc.axhline(incidence_range_threshold, color="#6c757d", linestyle="--", linewidth=1.0, alpha=0.8)
    ax_inc.set_xscale("log")
    ax_inc.set_xlabel("RA (%)")
    ax_inc.set_ylabel("rho range across incidence")
    ax_inc.set_title("incidence effect strength across RA")
    ax_inc.grid(True, alpha=0.25)

    fig.suptitle("Mini-RF CPR/Green covariance diagnostics", fontsize=14)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    return output_path


def main() -> None:
    """Load data, compute covariance diagnostics, and save the diagnostic figure."""
    data = load_analysis_dataset(data_path, ANALYSIS_VARIABLES)
    s = preprocess_minirf_data(data)
    diag = _compute_standardized_residuals(s, sigma_floor=sigma_floor)
    binned = _compute_ra_binned_slopes(diag, n_bins=n_ra_bins, min_points_per_bin=min_points_per_bin)
    smooth_summary = _fit_smooth_rho_models(diag)
    out_path = make_covariance_diagnostic_plot(diag, binned, smooth_summary, output_plot_path)

    print(f"Saved covariance diagnostic plot to {out_path}")
    print(f"N valid points: {diag['ra'].size}")
    print(f"Global residual rho: {_safe_corr(diag['z1'], diag['z2']):.4f}")
    print(f"Global raw G-on-CPR slope: {_safe_linear_slope(diag['rad1'], diag['rad2']):.4f}")
    print(f"Global residual slope: {_safe_linear_slope(diag['z1'], diag['z2']):.4f}")
    print(f"Validation nll per point, constant rho: {smooth_summary['fits']['constant']['val_nll']:.6f}")
    print(f"Validation nll per point, rho(ra): {smooth_summary['fits']['ra']['val_nll']:.6f}")
    print(f"Validation nll per point, rho(ra,inc): {smooth_summary['fits']['ra_inc']['val_nll']:.6f}")
    print(f"rho(ra,inc) params: {np.asarray(smooth_summary['fits']['ra_inc']['params'], dtype=float).tolist()}")
    print(f"rho(ra,inc) feature stats: {smooth_summary['fits']['ra_inc']['feature_stats']}")
    print(f"Incidence delta validation nll: {smooth_summary['comparison']['delta_nll_inc']:.6f}")
    print(f"Median rho range across incidence: {smooth_summary['comparison']['median_incidence_rho_range']:.6f}")
    print(f"Incidence effect weak: {smooth_summary['comparison']['incidence_effect_weak']}")
    print(f"Selected smooth rho model: {smooth_summary['comparison']['selected_mode']}")


if __name__ == "__main__":
    main()
