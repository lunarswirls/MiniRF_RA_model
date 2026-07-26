#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
translated v3 ra prediction workflow from the matlab part2 folder
"""

import math
import os
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable
import h5py
import numpy as np
from numba import njit
from scipy.io import loadmat

repo_root = Path(__file__).resolve().parents[1]
mpl_cache_dir = repo_root / "output/.mplcache"
cache_home_dir = repo_root / "output/.cache"
mpl_cache_dir.mkdir(parents=True, exist_ok=True)
cache_home_dir.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", str(mpl_cache_dir))
os.environ.setdefault("XDG_CACHE_HOME", str(cache_home_dir))

import parallel_ra_prediction_v2 as pr2


Array = np.ndarray
REALMIN = np.finfo(float).tiny
ANALYSIS_VARIABLES = ["miniRF_1", "miniRF_2", "miniRF_3", "miniRF_4", "miniRF_5", "miniRF_6", "ra", "wac", "lat", "lon"]
MODEL_SIGMOID_COSD_EXP_RA = 0
MODEL_SIGMOID_COSD_LINEAR_RA = 1

# path to the v3 input dataset
data_path = Path(__file__).resolve().parents[1] / "data/GiordanoBruno.mat"

# path to the matlab part2 scattering fits
scattering_fit_path = Path(__file__).resolve().parent / "Part2/ScatteringFits_2026-07-07_11-57-01.mat"

# path to the saved covariance grid exported from the matlab v2 file
covmodel_path = Path(__file__).resolve().parents[1] / "data/covModel_global_CPR_Green_2026-07-09_085116.npz"

# output path for the predicted RA and sigma maps
map_plot_output_path = Path(__file__).resolve().parents[1] / "output/ra_prediction_v3_map_sigma.png"

# output path for the predicted RA histogram comparison
hist_plot_output_path = Path(__file__).resolve().parents[1] / "output/ra_prediction_v3_histograms.png"

# number of worker threads used by the spatial passes
workers = 20

# lower bound of the posterior RA grid in percent
ra_grid_min = 0.01

# upper bound of the posterior RA grid in percent
ra_grid_max = 10.0

# number of points in the posterior RA grid
ra_grid_points = 500

# minimum model sigma used in the first and second pass
sigma_floor = 0.02

# square window size for the first-pass local neighborhood
firstpass_window_size = 3

# minimum fractional width of the gaussian prior passed to the second pass
firstpass_bwmin = 0.2

# maximum fractional width of the gaussian prior passed to the second pass
firstpass_bwmax = 0.75

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

# chebyshev coefficients fit to src/Part2/Beta.mat on scaled log10(RA)
beta_cheb_coeffs = np.array(
    [
        0.5203371230663933,
        -0.13958830472228942,
        -0.1435073951592003,
        -0.23751035535082798,
        0.005880025087457168,
        0.05139112284753412,
        -0.020600005926807947,
        -0.011283387453001552,
        0.011493653091726664,
        0.0011017963106576814,
        0.00147934118618601,
        0.011128733133124725,
        -0.010093398289530353,
        -0.007501692303393376,
        0.013849708962984457,
        0.005653293323938062,
        -0.015286303687770822,
    ],
    dtype=float,
)


evaluate_ra_prediction = pr2.evaluate_ra_prediction
make_ra_map_sigma_plot = pr2.make_ra_map_sigma_plot
make_ra_histogram_plot = pr2.make_ra_histogram_plot


@dataclass(frozen=True)
class ModelFit:
    model_fun: Callable[[Array, float | Array, Array], Array]
    params: Array
    model_kind: int


def default_ra_grid(ra_min: float = 0.01, ra_max: float = 10.0, n_grid_points: int = 500) -> Array:
    """build default logarithmic RA grid"""
    epsilon = math.log10(ra_max / ra_min) / n_grid_points
    return np.power(10.0, math.log10(ra_min) + np.arange(n_grid_points + 1, dtype=float) * epsilon)


def _sigmoid_cosd_exp_ra(params: Array, inc: float | Array, ra: Array) -> Array:
    """evaluate the saved sigmoid-plus-cosine model with exponential RA incidence scaling"""
    p = np.asarray(params, dtype=float)
    ra = np.asarray(ra, dtype=float)
    inc_term = np.power(np.cos(np.deg2rad(inc)), p[4] * np.exp(p[5] * ra))
    sigmoid = p[0] + (p[1] / (1.0 + np.exp(-p[2] * (np.log10(ra) + p[3]))))
    linear = p[6] + p[7] * ra
    return sigmoid * inc_term + linear


def _sigmoid_cosd_linear_ra(params: Array, inc: float | Array, ra: Array) -> Array:
    """evaluate the saved sigmoid-plus-cosine model with linear RA incidence scaling"""
    p = np.asarray(params, dtype=float)
    ra = np.asarray(ra, dtype=float)
    inc_term = np.power(np.cos(np.deg2rad(inc)), p[4] + p[5] * ra)
    sigmoid = p[0] + (p[1] / (1.0 + np.exp(-p[2] * (np.log10(ra) + p[3]))))
    linear = p[6] + p[7] * ra
    return sigmoid * inc_term + linear


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


def _decode_utf16_chars(values: Array) -> str:
    """decode one matlab uint16 char array"""
    values = np.asarray(values)
    return "".join(chr(int(value)) for value in values.reshape(-1) if int(value) != 0)


def _model_kind_from_text(model_text: str) -> int:
    """map the saved matlab anonymous function text to one model kind"""
    if "p(5)+p(6).*RA" in model_text:
        return MODEL_SIGMOID_COSD_LINEAR_RA
    if "p(5).*exp(p(6).*RA)" in model_text:
        return MODEL_SIGMOID_COSD_EXP_RA
    raise ValueError(f"unsupported matlab model form: {model_text}")


def _model_fun_from_kind(model_kind: int):
    """return the python model function for one saved model kind"""
    if model_kind == MODEL_SIGMOID_COSD_LINEAR_RA:
        return _sigmoid_cosd_linear_ra
    return _sigmoid_cosd_exp_ra


def load_parallel_scattering_models(mat_path: str | Path | None = None) -> dict[str, ModelFit]:
    """load the saved part2 CPR and green scattering fits"""
    if mat_path is None:
        mat_path = scattering_fit_path

    with h5py.File(Path(mat_path), "r") as handle:
        fit_root = handle["BestFitResults"]

        def read_fit(fit_index: int, fit_name: str) -> ModelFit:
            fit_group = handle[fit_root[fit_name][fit_index, 0]]
            params = np.asarray(fit_group["params"], dtype=float).reshape(-1)
            model_text = _decode_utf16_chars(fit_group["modelFun"]["function_handle"]["function"][()])
            model_kind = _model_kind_from_text(model_text)
            return ModelFit(
                model_fun=_model_fun_from_kind(model_kind),
                params=params,
                model_kind=model_kind,
            )

        return {
            "Rad1muFit": read_fit(0, "medianFit"),
            "Rad1iqrFit": read_fit(0, "iqrFit"),
            "Rad2muFit": read_fit(4, "medianFit"),
            "Rad2iqrFit": read_fit(4, "iqrFit"),
        }


def load_analysis_dataset(mat_path: str | Path, variable_names: list[str] | None = None) -> dict[str, Array]:
    """load the crater dataset used by the part2 workflow"""
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


def find_joint_stack(data: dict[str, Array], sel: list[str]) -> tuple[dict[str, Array], Array]:
    """crop to the tight bounding box with finite data in all selected layers"""
    stack = np.stack([np.asarray(data[name], dtype=float) for name in sel], axis=2)
    joint_mask = np.all(np.isfinite(stack), axis=2)
    if not np.any(joint_mask):
        raise ValueError("joint mask is empty for the selected fields")

    rows = np.flatnonzero(np.any(joint_mask, axis=1))
    cols = np.flatnonzero(np.any(joint_mask, axis=0))
    r0 = int(rows[0])
    r1 = int(rows[-1] + 1)
    c0 = int(cols[0])
    c1 = int(cols[-1] + 1)

    cropped = {name: stack[r0:r1, c0:c1, i] for i, name in enumerate(sel)}
    return cropped, joint_mask[r0:r1, c0:c1]


def preprocess_minirf_data(data: dict[str, Array]) -> dict[str, Array]:
    """apply the part2 preprocessing and crop to the common finite footprint"""
    s = {key: np.array(value, dtype=float, copy=True) for key, value in data.items()}

    s["miniRF_1"][s["miniRF_1"] < 0.0] = np.nan
    pwrchk = s["miniRF_1"] < np.sqrt(s["miniRF_2"] ** 2 + s["miniRF_3"] ** 2 + s["miniRF_4"] ** 2)
    for key in ("miniRF_1", "miniRF_2", "miniRF_3", "miniRF_4"):
        s[key][pwrchk] = np.nan

    with np.errstate(invalid="ignore", divide="ignore"):
        s["m"] = np.sqrt(s["miniRF_2"] ** 2 + s["miniRF_3"] ** 2 + s["miniRF_4"] ** 2) / s["miniRF_1"]
        s["cpr"] = (s["miniRF_1"] - s["miniRF_4"]) / (s["miniRF_1"] + s["miniRF_4"])
        s["g"] = np.sqrt(s["miniRF_1"] * (1.0 - s["m"]))

    s["inc"] = s["miniRF_5"]
    s["num"] = s["miniRF_6"]
    s["ra"] = s["ra"] * 100.0

    mask = (s["num"] == 1) & (s["ra"] > 0.01) & np.isfinite(s["cpr"]) & np.isfinite(s["g"]) & np.isfinite(s["inc"])
    for key in ("cpr", "g", "inc", "ra"):
        s[key][~mask] = np.nan

    crop_fields = [name for name in ("cpr", "g", "ra", "inc", "wac", "lat", "lon") if name in s]
    cropped, crop_mask = find_joint_stack(s, crop_fields)

    result = dict(cropped)
    result["crop_mask"] = crop_mask
    result["Rad1"] = result["cpr"]
    result["Rad2"] = result["g"]
    result["RAMap"] = result["ra"]
    result["IncidenceMap"] = result["inc"]
    return result


@njit(cache=True)
def _trapz_1d(y: Array, x: Array) -> float:
    """integrate a sampled 1d curve with the trapezoid rule"""
    total = 0.0
    for i in range(y.size - 1):
        total += 0.5 * (y[i] + y[i + 1]) * (x[i + 1] - x[i])
    return total


@njit(cache=True)
def normalize_density_inplace(values: Array, grid: Array) -> None:
    """clamp invalid density values and normalize them in place"""
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
    """evaluate one hardcoded scattering-model fit at one RA value"""
    inc_cos = math.cos((math.pi / 180.0) * inc_deg)
    if model_kind == MODEL_SIGMOID_COSD_EXP_RA:
        inc_term = inc_cos ** (params[4] * math.exp(params[5] * ra))
    else:
        inc_term = inc_cos ** (params[4] + params[5] * ra)

    sigmoid = params[0] + (params[1] / (1.0 + math.exp(-params[2] * (math.log10(ra) + params[3]))))
    linear = params[6] + params[7] * ra
    return sigmoid * inc_term + linear


@njit(cache=True)
def interp_monotonic_prefix(x: float, xp: Array, fp: Array, count: int) -> float:
    """interpolate on the strictly increasing prefix of a monotonic grid"""
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
def eval_chebyshev(x: float, coeffs: Array) -> float:
    """evaluate a chebyshev series with clenshaw recursion"""
    if coeffs.size == 0:
        return 0.0
    if coeffs.size == 1:
        return coeffs[0]

    b_kplus1 = 0.0
    b_kplus2 = 0.0
    for i in range(coeffs.size - 1, 0, -1):
        b_k = 2.0 * x * b_kplus1 - b_kplus2 + coeffs[i]
        b_kplus2 = b_kplus1
        b_kplus1 = b_k
    return x * b_kplus1 - b_kplus2 + coeffs[0]


@njit(cache=True)
def beta_weight_v3(ra: float, coeffs: Array) -> float:
    """evaluate the fitted beta equation that replaces the matlab lookup table"""
    log_ra = math.log10(ra)
    x_scaled = (2.0 * (log_ra + 2.0) / 3.0) - 1.0
    if x_scaled < -1.0:
        x_scaled = -1.0
    if x_scaled > 1.0:
        x_scaled = 1.0
    beta = eval_chebyshev(x_scaled, coeffs)
    if beta < 1e-3:
        beta = 1e-3
    if beta > 0.999:
        beta = 0.999
    return beta


@njit(cache=True)
def posterior_quantile(post: Array, ra_grid: Array, q: float) -> float:
    """return one posterior quantile from a normalized density"""
    ngrid = post.size
    cdf = np.empty(ngrid, dtype=np.float64)
    cdf[0] = 0.0
    for i in range(1, ngrid):
        cdf[i] = cdf[i - 1] + 0.5 * (post[i - 1] + post[i]) * (ra_grid[i] - ra_grid[i - 1])

    total = cdf[ngrid - 1]
    if not math.isfinite(total) or total <= 0.0:
        return math.nan

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

    return interp_monotonic_prefix(q, unique_cdf, unique_ra, count)


@njit(cache=True)
def build_normalized_gaussian_pdf(ra_grid: Array, mu: float, sigma: float) -> Array:
    """build one normalized gaussian density on the RA grid"""
    post = np.empty(ra_grid.size, dtype=np.float64)
    if not math.isfinite(sigma) or sigma <= 0.0:
        sigma = 1e-12

    norm = sigma * math.sqrt(2.0 * math.pi)
    for i in range(ra_grid.size):
        z = (ra_grid[i] - mu) / sigma
        post[i] = math.exp(-0.5 * z * z) / norm

    normalize_density_inplace(post, ra_grid)
    return post


@njit(cache=True)
def detect_bimodal_prior(ra_grid: Array, post: Array) -> tuple[bool, float, float]:
    """approximate the matlab bimodal prior handler inside numba"""
    max_post = 0.0
    for i in range(post.size):
        if math.isfinite(post[i]) and post[i] > max_post:
            max_post = post[i]

    if max_post <= 0.0:
        return (False, math.nan, math.nan)

    peak_idx = np.empty(post.size, dtype=np.int64)
    peak_val = np.empty(post.size, dtype=np.float64)
    n_peaks = 0
    threshold = 0.10 * max_post
    for i in range(1, post.size - 1):
        if post[i] >= threshold and post[i] >= post[i - 1] and post[i] >= post[i + 1]:
            peak_idx[n_peaks] = i
            peak_val[n_peaks] = post[i]
            n_peaks += 1

    if n_peaks < 2:
        return (False, math.nan, math.nan)

    best_i = -1
    best_j = -1
    best_pair_score = -math.inf
    for i in range(n_peaks - 1):
        for j in range(i + 1, n_peaks):
            dlog = abs(math.log10(ra_grid[peak_idx[i]]) - math.log10(ra_grid[peak_idx[j]]))
            if dlog <= 0.3:
                continue
            pair_score = peak_val[i] + peak_val[j]
            if pair_score > best_pair_score:
                best_pair_score = pair_score
                best_i = peak_idx[i]
                best_j = peak_idx[j]

    if best_i < 0 or best_j < 0:
        return (False, math.nan, math.nan)

    ra_a = ra_grid[best_i]
    ra_b = ra_grid[best_j]
    log_center = 0.5 * (math.log10(ra_a) + math.log10(ra_b))
    ra_center = 10.0 ** log_center
    width_modes = abs(ra_b - ra_a)
    if width_modes <= 0.0:
        width_modes = ra_center

    sigma_prior = 0.5 * width_modes
    sigma_min = 0.1 * ra_center
    sigma_max = 2.0 * ra_center
    if sigma_prior < sigma_min:
        sigma_prior = sigma_min
    if sigma_prior > sigma_max:
        sigma_prior = sigma_max
    return (True, ra_center, sigma_prior)


@njit(cache=True)
def firstpass_summary_from_loglikes(log_l1: Array, log_l2: Array, prior: Array, ra_grid: Array,
                                    beta_coeffs: Array) -> tuple[float, float, float, float, float, float]:
    """turn the local first-pass likelihoods into a reusable prior summary"""
    ngrid = ra_grid.size
    log_post1 = np.empty(ngrid, dtype=np.float64)
    log_post2 = np.empty(ngrid, dtype=np.float64)
    post1 = np.empty(ngrid, dtype=np.float64)
    post2 = np.empty(ngrid, dtype=np.float64)
    post = np.empty(ngrid, dtype=np.float64)
    beta_values = np.empty(ngrid, dtype=np.float64)

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

    max_log_post = -math.inf
    for i in range(ngrid):
        beta_values[i] = beta_weight_v3(ra_grid[i], beta_coeffs)
        post[i] = beta_values[i] * math.log(max(post1[i], REALMIN))
        post[i] += (1.0 - beta_values[i]) * math.log(max(post2[i], REALMIN))
        if post[i] > max_log_post:
            max_log_post = post[i]

    for i in range(ngrid):
        post[i] = math.exp(post[i] - max_log_post)
    normalize_density_inplace(post, ra_grid)

    is_bimodal, prior_center, prior_sigma = detect_bimodal_prior(ra_grid, post)
    if is_bimodal:
        return (prior_center, prior_center, prior_sigma, math.nan, 3.0, 1.0)

    max_idx = 0
    max_post = -math.inf
    for i in range(ngrid):
        if post[i] > max_post:
            max_post = post[i]
            max_idx = i

    ra_best = ra_grid[max_idx]
    flag = 0.0

    if ra_best < 0.3:
        max_idx = 0
        max_post = -math.inf
        max_log_post = -math.inf
        for i in range(ngrid):
            post[i] = (1.0 - beta_values[i]) * math.log(max(post1[i], REALMIN))
            post[i] += beta_values[i] * math.log(max(post2[i], REALMIN))
            if post[i] > max_log_post:
                max_log_post = post[i]
        for i in range(ngrid):
            post[i] = math.exp(post[i] - max_log_post)
        normalize_density_inplace(post, ra_grid)
        for i in range(ngrid):
            if post[i] > max_post:
                max_post = post[i]
                max_idx = i
        ra_best = ra_grid[max_idx]
        flag = 1.0

    if ra_best < 0.3:
        post = build_normalized_gaussian_pdf(ra_grid, 0.3, 0.2)
        ra_best = 0.3
        flag = 2.0

    ra_low = posterior_quantile(post, ra_grid, 0.18)
    ra_high = posterior_quantile(post, ra_grid, 0.86)
    bw_raw = 0.5
    if math.isfinite(ra_low) and math.isfinite(ra_high):
        width = ra_high - ra_low
        if width < 0.0:
            width = 0.0
        bw_raw = width / max(ra_best, 1e-3)

    return (ra_best, ra_best, math.nan, bw_raw, flag, 0.0)


@njit(cache=True)
def fit_local_scattering_curve_2rad_bw_numba(
    rad1_vec: Array,
    rad2_vec: Array,
    inc_vec: Array,
    w_vec: Array,
    center_rad1: float,
    center_rad2: float,
    center_inc: float,
    ra_grid: Array,
    sigma_floor_value: float,
    alpha: float,
    beta_coeffs: Array,
    mu1_kind: int,
    mu1_params: Array,
    sigma1_kind: int,
    sigma1_params: Array,
    mu2_kind: int,
    mu2_params: Array,
    sigma2_kind: int,
    sigma2_params: Array,
) -> tuple[float, float, float, float, float, float]:
    """score one local first-pass patch and summarize the prior handoff"""
    nobs = rad1_vec.size
    ngrid = ra_grid.size
    if nobs == 0 or ngrid == 0:
        return (math.nan, math.nan, math.nan, math.nan, math.nan, math.nan)

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

    neff_weighted = (sumw * sumw) / max(sumw2, 1e-12)
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
            if sigma1 < sigma_floor_value:
                sigma1 = sigma_floor_value
            sigma1 *= inflate

            mu2 = model_value(mu2_kind, mu2_params, inc, ra)
            sigma2 = model_value(sigma2_kind, sigma2_params, inc, ra) / 1.349
            if sigma2 < sigma_floor_value:
                sigma2 = sigma_floor_value
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

    ra_min = ra_grid[0]
    ra_max = ra_grid[ngrid - 1]

    rad1_pred_min = model_value(mu1_kind, mu1_params, center_inc, ra_min)
    rad1_iqr_min = model_value(sigma1_kind, sigma1_params, center_inc, ra_min)
    rad1_low_thresh = rad1_pred_min - rad1_iqr_min

    rad2_pred_min = model_value(mu2_kind, mu2_params, center_inc, ra_min)
    rad2_iqr_min = model_value(sigma2_kind, sigma2_params, center_inc, ra_min)
    rad2_low_thresh = rad2_pred_min - rad2_iqr_min

    rad1_pred_max = model_value(mu1_kind, mu1_params, center_inc, ra_max)
    rad1_iqr_max = model_value(sigma1_kind, sigma1_params, center_inc, ra_max)
    rad1_high_thresh = rad1_pred_max + rad1_iqr_max

    rad2_pred_max = model_value(mu2_kind, mu2_params, center_inc, ra_max)
    rad2_iqr_max = model_value(sigma2_kind, sigma2_params, center_inc, ra_max)
    rad2_high_thresh = rad2_pred_max + rad2_iqr_max

    ra0 = 1.0
    if center_rad1 < rad1_low_thresh or center_rad2 < rad2_low_thresh:
        ra0 = 0.5
    elif center_rad1 > rad1_high_thresh or center_rad2 > rad2_high_thresh:
        ra0 = 5.0

    prior = np.ones(ngrid, dtype=np.float64)
    if ra0 <= 1.0:
        for i in range(ngrid):
            if ra_grid[i] > ra0:
                prior[i] = (ra_grid[i] / ra0) ** (-alpha)
    else:
        for i in range(ngrid):
            if ra_grid[i] < ra0:
                prior[i] = (ra_grid[i] / ra0) ** alpha

    normalize_density_inplace(prior, ra_grid)
    return firstpass_summary_from_loglikes(log_l1, log_l2, prior, ra_grid, beta_coeffs)


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
def posterior_summary_from_loglike(log_l: Array, prior: Array, ra_grid: Array) -> tuple[float, float, float, float, float]:
    """convert a log-likelihood curve into map, mean, and interval summaries"""
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
def invert_ra_from_rad_second_2rad_joint_numba(
    rad1_vec: Array,
    rad2_vec: Array,
    inc_vec: Array,
    prior: Array,
    ra_grid: Array,
    mu1_kind: int,
    mu1_params: Array,
    sigma1_kind: int,
    sigma1_params: Array,
    mu2_kind: int,
    mu2_params: Array,
    sigma2_kind: int,
    sigma2_params: Array,
    rho_eta_params: Array,
    rho_feature_stats: Array,
    sigma_floor_value: float,
) -> tuple[float, float, float, float, float]:
    """evaluate the joint CPR-G likelihood across one posterior RA grid"""
    nobs = rad1_vec.size
    ngrid = ra_grid.size
    if nobs == 0 or ngrid == 0:
        return (math.nan, math.nan, math.nan, math.nan, math.nan)

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
            if sigma1 < sigma_floor_value:
                sigma1 = sigma_floor_value
            sigma1 *= inflate

            mu2 = model_value(mu2_kind, mu2_params, inc, ra)
            sigma2 = model_value(sigma2_kind, sigma2_params, inc, ra) / 1.349
            if sigma2 < sigma_floor_value:
                sigma2 = sigma_floor_value
            sigma2 *= inflate

            rho = smooth_rho(ra, inc, rho_eta_params, rho_feature_stats)
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


def build_default_pipeline_options(models: dict[str, ModelFit] | None = None,
                                   cov_model: dict[str, Any] | None = None) -> dict[str, dict[str, Any]]:
    """build the v3 pipeline settings from the hardcoded variables"""
    if models is None:
        models = load_parallel_scattering_models()
    if cov_model is None:
        cov_model = load_covariance_model()

    ra_grid = default_ra_grid(ra_min=ra_grid_min, ra_max=ra_grid_max, n_grid_points=ra_grid_points)
    firstpass_half_win = firstpass_window_size // 2
    firstpass_sigma_dist = firstpass_window_size / 3.0

    firstpass_settings: dict[str, Any] = {
        "RA_grid": ra_grid,
        "sigmaFloor": sigma_floor,
        "halfWin": firstpass_half_win,
        "sigmaDist": firstpass_sigma_dist,
        "bwmin": firstpass_bwmin,
        "bwmax": firstpass_bwmax,
        "betaChebCoeffs": np.array(beta_cheb_coeffs, dtype=float),
        "Rad1muFit": models["Rad1muFit"],
        "Rad1iqrFit": models["Rad1iqrFit"],
        "Rad2muFit": models["Rad2muFit"],
        "Rad2iqrFit": models["Rad2iqrFit"],
    }

    secondpass_settings: dict[str, Any] = {
        "RA_grid": ra_grid,
        "sigmaFloor": sigma_floor,
        "covModel": cov_model,
    }

    eval_settings: dict[str, Any] = {
        "regimeEdges": list(eval_regime_edges),
        "regimeNames": list(eval_regime_names),
        "smoothWindow": eval_smooth_window,
        "mapMode": eval_map_mode,
        "report": eval_report,
    }

    return {"firstpass": firstpass_settings, "secondpass": secondpass_settings, "eval": eval_settings}


def invert_ra_2rad_first_pass_spatial_summary(
    rad1: Array,
    rad2: Array,
    ra_map: Array,
    incidence_map: Array,
    firstpass_settings: dict[str, Any],
    n_workers: int | None = None,
) -> dict[str, Array]:
    """run the first-pass local inversion and cache the reusable prior summaries"""
    nrows, ncols = rad1.shape
    half_win = int(firstpass_settings["halfWin"])
    sigma_dist = float(firstpass_settings["sigmaDist"])
    rad1_mu_fit = firstpass_settings["Rad1muFit"]
    rad1_iqr_fit = firstpass_settings["Rad1iqrFit"]
    rad2_mu_fit = firstpass_settings["Rad2muFit"]
    rad2_iqr_fit = firstpass_settings["Rad2iqrFit"]
    beta_coeffs = np.asarray(firstpass_settings["betaChebCoeffs"], dtype=float)
    ra_grid = np.asarray(firstpass_settings["RA_grid"], dtype=float).reshape(-1)

    rad1_vec = rad1.reshape(-1, order="F")
    rad2_vec = rad2.reshape(-1, order="F")
    inc_vec = incidence_map.reshape(-1, order="F")
    valid_pixel = np.isfinite(rad1_vec) & np.isfinite(rad2_vec) & np.isfinite(inc_vec)
    valid_idx = np.flatnonzero(valid_pixel)

    def work(idx: int) -> tuple[float, float, float, float, float, float]:
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
            return (np.nan, np.nan, np.nan, np.nan, np.nan, np.nan)

        rad1_local = rad1_local[valid]
        rad2_local = rad2_local[valid]
        inc_local = inc_local[valid]
        w_local = w_local[valid]

        alpha = 0.034 * rad1_local.size + 0.106
        return fit_local_scattering_curve_2rad_bw_numba(
            rad1_local,
            rad2_local,
            inc_local,
            w_local,
            float(rad1[r, c]),
            float(rad2[r, c]),
            float(incidence_map[r, c]),
            ra_grid,
            float(firstpass_settings["sigmaFloor"]),
            float(alpha),
            beta_coeffs,
            rad1_mu_fit.model_kind,
            np.asarray(rad1_mu_fit.params, dtype=float),
            rad1_iqr_fit.model_kind,
            np.asarray(rad1_iqr_fit.params, dtype=float),
            rad2_mu_fit.model_kind,
            np.asarray(rad2_mu_fit.params, dtype=float),
            rad2_iqr_fit.model_kind,
            np.asarray(rad2_iqr_fit.params, dtype=float),
        )

    if n_workers == 1:
        packed = [work(idx) for idx in valid_idx]
    else:
        with ThreadPoolExecutor(max_workers=n_workers) as pool:
            packed = list(pool.map(work, valid_idx))

    map_temp = np.full(valid_idx.shape, np.nan, dtype=float)
    center_temp = np.full(valid_idx.shape, np.nan, dtype=float)
    sigma_temp = np.full(valid_idx.shape, np.nan, dtype=float)
    bw_temp = np.full(valid_idx.shape, np.nan, dtype=float)
    flag_temp = np.full(valid_idx.shape, np.nan, dtype=float)
    bimodal_temp = np.full(valid_idx.shape, np.nan, dtype=float)

    for i, values in enumerate(packed):
        map_temp[i], center_temp[i], sigma_temp[i], bw_temp[i], flag_temp[i], bimodal_temp[i] = values

    def scatter(values: Array) -> Array:
        flat = np.full(rad1.size, np.nan, dtype=float)
        flat[valid_idx] = values
        return flat.reshape(rad1.shape, order="F")

    return {
        "RA_first": scatter(map_temp),
        "PriorCenter": scatter(center_temp),
        "PriorSigma": scatter(sigma_temp),
        "BWraw": scatter(bw_temp),
        "FlagMap": scatter(flag_temp),
        "IsBimodal": scatter(bimodal_temp),
    }


def _build_secondpass_prior(
    ra_grid: Array,
    prior_center: float,
    prior_sigma_fixed: float,
    bw_raw: float,
    bwmin: float,
    bwmax: float,
    is_bimodal: bool,
) -> Array:
    """build one second-pass gaussian prior from the cached first-pass summary"""
    if is_bimodal:
        sigma_prior = prior_sigma_fixed
    else:
        bw = float(np.clip(bw_raw, bwmin, bwmax))
        sigma_prior = prior_center * bw

    sigma_prior = max(float(sigma_prior), 1e-12)
    prior = np.exp(-0.5 * ((ra_grid - prior_center) / sigma_prior) ** 2) / (sigma_prior * math.sqrt(2.0 * math.pi))
    prior = np.asarray(prior, dtype=float)
    prior[~np.isfinite(prior)] = 0.0
    area = float(np.trapezoid(prior, ra_grid))
    if not np.isfinite(area) or area <= 0.0:
        prior = np.ones_like(ra_grid, dtype=float)
        area = float(np.trapezoid(prior, ra_grid))
    prior /= area
    return prior


def invert_ra_2rad_second_pass_spatial(
    rad1: Array,
    rad2: Array,
    inc_obs: Array,
    firstpass_state: dict[str, Array],
    firstpass_settings: dict[str, Any],
    secondpass_settings: dict[str, Any],
    n_workers: int | None = None,
) -> dict[str, Array]:
    """run the part2 joint-gaussian second pass over the full map"""
    rad1_mu_fit = firstpass_settings["Rad1muFit"]
    rad1_iqr_fit = firstpass_settings["Rad1iqrFit"]
    rad2_mu_fit = firstpass_settings["Rad2muFit"]
    rad2_iqr_fit = firstpass_settings["Rad2iqrFit"]
    ra_grid = np.asarray(secondpass_settings["RA_grid"], dtype=float).reshape(-1)
    rho_eta_params = np.asarray(secondpass_settings["covModel"]["rhoEtaParams"], dtype=float).reshape(-1)
    rho_feature_stats = np.asarray(secondpass_settings["covModel"]["rhoFeatureStats"], dtype=float).reshape(-1)
    bwmin = float(firstpass_settings["bwmin"])
    bwmax = float(firstpass_settings["bwmax"])
    sigma_floor_value = float(secondpass_settings["sigmaFloor"])

    valid_pixel = np.isfinite(rad1) & np.isfinite(rad2) & np.isfinite(inc_obs) & np.isfinite(firstpass_state["PriorCenter"])
    valid_idx = np.flatnonzero(valid_pixel.reshape(-1, order="F"))

    def work(idx: int) -> tuple[float, float, float, float, float]:
        r, c = np.unravel_index(idx, rad1.shape, order="F")
        prior_center = float(firstpass_state["PriorCenter"][r, c])
        prior_sigma_fixed = float(firstpass_state["PriorSigma"][r, c])
        bw_raw = float(firstpass_state["BWraw"][r, c])
        is_bimodal = bool(firstpass_state["IsBimodal"][r, c] > 0.5)
        prior = _build_secondpass_prior(ra_grid, prior_center, prior_sigma_fixed, bw_raw, bwmin, bwmax, is_bimodal)

        rad1_vec = np.array([float(rad1[r, c])], dtype=float)
        rad2_vec = np.array([float(rad2[r, c])], dtype=float)
        inc_vec = np.array([float(inc_obs[r, c])], dtype=float)

        return invert_ra_from_rad_second_2rad_joint_numba(
            rad1_vec,
            rad2_vec,
            inc_vec,
            prior,
            ra_grid,
            rad1_mu_fit.model_kind,
            np.asarray(rad1_mu_fit.params, dtype=float),
            rad1_iqr_fit.model_kind,
            np.asarray(rad1_iqr_fit.params, dtype=float),
            rad2_mu_fit.model_kind,
            np.asarray(rad2_mu_fit.params, dtype=float),
            rad2_iqr_fit.model_kind,
            np.asarray(rad2_iqr_fit.params, dtype=float),
            rho_eta_params,
            rho_feature_stats,
            sigma_floor_value,
        )

    if n_workers == 1:
        packed = [work(idx) for idx in valid_idx]
    else:
        with ThreadPoolExecutor(max_workers=n_workers) as pool:
            packed = list(pool.map(work, valid_idx))

    ra_map_temp = np.full(valid_idx.shape, np.nan, dtype=float)
    ra_mean_temp = np.full(valid_idx.shape, np.nan, dtype=float)
    ra_med_temp = np.full(valid_idx.shape, np.nan, dtype=float)
    ra_low_temp = np.full(valid_idx.shape, np.nan, dtype=float)
    ra_high_temp = np.full(valid_idx.shape, np.nan, dtype=float)
    for i, values in enumerate(packed):
        ra_map_temp[i], ra_mean_temp[i], ra_med_temp[i], ra_low_temp[i], ra_high_temp[i] = values

    def scatter(values: Array) -> Array:
        flat = np.full(rad1.size, np.nan, dtype=float)
        flat[valid_idx] = values
        return flat.reshape(rad1.shape, order="F")

    return {
        "MAP": scatter(ra_map_temp),
        "mean": scatter(ra_mean_temp),
        "med": scatter(ra_med_temp),
        "low": scatter(ra_low_temp),
        "high": scatter(ra_high_temp),
    }


def predict_ra_from_preprocessed_with_firstpass_state(
    s: dict[str, Array],
    firstpass_state: dict[str, Array],
    firstpass_options: dict[str, Any],
    secondpass_options: dict[str, Any],
    n_workers: int | None = None,
) -> dict[str, Any]:
    """reuse a cached first pass summary and run the second-pass prediction"""
    ra_pred_maps = invert_ra_2rad_second_pass_spatial(
        s["Rad1"],
        s["Rad2"],
        s["IncidenceMap"],
        firstpass_state,
        firstpass_options,
        secondpass_options,
        n_workers=n_workers,
    )

    ra_pred = ra_pred_maps["med"]
    result = {
        "S": s,
        "FirstPass": firstpass_state,
        "RApredMaps": ra_pred_maps,
        "RApred": ra_pred,
    }

    if "ra" in s:
        result["sigma_map"] = make_sigma_map(
            ra_pred_maps["med"],
            ra_pred_maps["low"],
            ra_pred_maps["high"],
            s["ra"],
        )
        result["sigma_envelope_map"] = compute_sigma_envelope(
            ra_pred_maps["med"],
            ra_pred_maps["low"],
            ra_pred_maps["high"],
            s["ra"],
            max_sigma=5,
        )

    return result


def predict_ra_from_preprocessed(
    s: dict[str, Array],
    firstpass_options: dict[str, Any],
    secondpass_options: dict[str, Any],
    n_workers: int | None = None,
) -> dict[str, Any]:
    """run the full two-pass v3 prediction from preprocessed inputs"""
    firstpass_state = invert_ra_2rad_first_pass_spatial_summary(
        s["Rad1"],
        s["Rad2"],
        s["RAMap"],
        s["IncidenceMap"],
        firstpass_options,
        n_workers=n_workers,
    )
    return predict_ra_from_preprocessed_with_firstpass_state(
        s,
        firstpass_state,
        firstpass_options,
        secondpass_options,
        n_workers=n_workers,
    )


def driver_pred_ra_from_radar(data_path: str | Path, n_workers: int | None = None) -> dict[str, Any]:
    """top-level entry point for the translated v3 crater test run"""
    models = load_parallel_scattering_models()
    cov_model = load_covariance_model()
    data = load_analysis_dataset(data_path, list(ANALYSIS_VARIABLES))
    s = preprocess_minirf_data(data)
    pipeline_options = build_default_pipeline_options(models, cov_model)
    result = predict_ra_from_preprocessed(
        s,
        pipeline_options["firstpass"],
        pipeline_options["secondpass"],
        n_workers=n_workers,
    )

    eval_options = dict(pipeline_options["eval"])
    eval_options["RA_low"] = result["RApredMaps"]["low"]
    eval_options["RA_high"] = result["RApredMaps"]["high"]
    result["metrics_cpr"] = evaluate_ra_prediction(s["ra"], result["RApred"], eval_options)

    map_plot_path = make_ra_map_sigma_plot(result["RApred"], result.get("sigma_map"), map_plot_output_path)
    hist_plot_path = make_ra_histogram_plot(s["ra"], result["RApredMaps"]["med"], result["RApredMaps"]["mean"], hist_plot_output_path)
    result["map_plot_path"] = map_plot_path
    result["hist_plot_path"] = hist_plot_path
    print(f"Saved RA map and sigma map plot to {map_plot_path}", flush=True)
    print(f"Saved RA histogram plot to {hist_plot_path}", flush=True)
    return result


if __name__ == "__main__":
    driver_pred_ra_from_radar(data_path, n_workers=workers)
