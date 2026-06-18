#!/usr/bin/env python
# -*- coding: utf-8 -*-
import math
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable
import numpy as np
from numba import njit


Array = np.ndarray
REALMIN = np.finfo(float).tiny
ANALYSIS_VARIABLES = ["miniRF_1", "miniRF_2", "miniRF_3", "miniRF_4", "miniRF_5", "miniRF_6", "ra"]
MODEL_SIGMOID_COSD_EXP_RA = 0
MODEL_SIGMOID_COSD_LINEAR_RA = 1

# path to the input dataset
data_path = Path(__file__).resolve().parents[1] / "data/GiordanoBruno_analysis.csv"
workers = None


@dataclass(frozen=True)
class ModelFit:
    model_fun: Callable[[Array, float | Array, Array], Array]
    params: Array
    model_kind: int


def cosd(x: float | Array) -> Array:
    """Return the cosine of angles expressed in degrees"""
    return np.cos(np.deg2rad(x))


def normpdf(x: Array, mu: float | Array, sigma: float | Array) -> Array:
    """Evaluate a normal probability density function"""
    sigma = np.asarray(sigma, dtype=float)
    x = np.asarray(x, dtype=float)
    mu = np.asarray(mu, dtype=float)
    return np.exp(-0.5 * ((x - mu) / sigma) ** 2) / (sigma * math.sqrt(2.0 * math.pi))


def student_t_pdf(z: Array, nu: float) -> Array:
    """Evaluate a standard Student-t density for the given z values"""
    coeff = math.exp(math.lgamma((nu + 1.0) / 2.0) - math.lgamma(nu / 2.0))
    coeff /= math.sqrt(nu * math.pi)
    return coeff * np.power(1.0 + (z**2) / nu, -0.5 * (nu + 1.0))


def default_ra_grid(ra_min: float = 0.01, ra_max: float = 10.0, n_grid_points: int = 500) -> Array:
    """Build the default logarithmic RA grid used by the MATLAB workflow"""
    epsilon = math.log10(ra_max / ra_min) / n_grid_points
    return np.power(10.0, math.log10(ra_min) + np.arange(n_grid_points + 1, dtype=float) * epsilon)


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
    """Integrate a sampled 1D curve with the trapezoid rule."""
    total = 0.0
    for i in range(y.size - 1):
        total += 0.5 * (y[i] + y[i + 1]) * (x[i + 1] - x[i])
    return total


@njit(cache=True)
def _normalize_density_inplace(values: Array, grid: Array) -> None:
    """Clamp invalid density values and normalize them in place."""
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
def _model_value(model_kind: int, params: Array, inc_deg: float, ra: float) -> float:
    """Evaluate one hardcoded scattering-model fit at one RA value."""
    inc_cos = math.cos((math.pi / 180.0) * inc_deg)
    if model_kind == MODEL_SIGMOID_COSD_EXP_RA:
        inc_term = inc_cos ** (params[4] * math.exp(params[5] * ra))
    else:
        inc_term = inc_cos ** (params[4] + params[5] * ra)

    sigmoid = params[0] + (params[1] / (1.0 + math.exp(-params[2] * (math.log10(ra) + params[3]))))
    linear = params[6] + params[7] * ra
    return sigmoid * inc_term + linear


@njit(cache=True)
def _beta_weight(ra: float, s: float) -> float:
    """Return the RA-dependent mixing weight used by the translated workflow."""
    return 0.01 + (1.0 - 0.01) / (1.0 + math.exp((ra - 2.5) / s))


@njit(cache=True)
def _interp_monotonic_prefix(x: float, xp: Array, fp: Array, count: int) -> float:
    """Interpolate on the strictly increasing prefix of a monotonic grid."""
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
def _posterior_summary_from_loglikes(log_l1: Array, log_l2: Array, prior: Array, ra_grid: Array,
                                     s: float) -> tuple[float, float, float, float]:
    """Combine radar-channel log-likelihoods with a prior and summarize the posterior."""
    ngrid = ra_grid.size

    log_post1 = np.empty(ngrid, dtype=np.float64)
    log_post2 = np.empty(ngrid, dtype=np.float64)
    post1 = np.empty(ngrid, dtype=np.float64)
    post2 = np.empty(ngrid, dtype=np.float64)
    post = np.empty(ngrid, dtype=np.float64)

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

    _normalize_density_inplace(post1, ra_grid)
    _normalize_density_inplace(post2, ra_grid)

    max_idx = 0
    max_post = -math.inf
    for i in range(ngrid):
        beta1 = _beta_weight(ra_grid[i], s)
        post[i] = beta1 * post1[i] + (1.0 - beta1) * post2[i]

    _normalize_density_inplace(post, ra_grid)

    for i in range(ngrid):
        if post[i] > max_post:
            max_post = post[i]
            max_idx = i

    cdf = np.empty(ngrid, dtype=np.float64)
    cdf[0] = 0.0
    for i in range(1, ngrid):
        cdf[i] = cdf[i - 1] + 0.5 * (post[i - 1] + post[i]) * (ra_grid[i] - ra_grid[i - 1])

    total = cdf[ngrid - 1]
    if not math.isfinite(total) or total <= 0.0:
        return (ra_grid[max_idx], math.nan, math.nan, math.nan)

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

    ra_low = _interp_monotonic_prefix(0.16, unique_cdf, unique_ra, count)
    ra_med = _interp_monotonic_prefix(0.50, unique_cdf, unique_ra, count)
    ra_high = _interp_monotonic_prefix(0.84, unique_cdf, unique_ra, count)

    return (ra_grid[max_idx], ra_med, ra_low, ra_high)


@njit(cache=True)
def _fit_local_scattering_curve_2rad_wbeta_numba(
    rad1_vec: Array,
    rad2_vec: Array,
    inc_vec: Array,
    w_vec: Array,
    ra_grid: Array,
    sigma_floor: float,
    alpha: float,
    ra0: float,
    s: float,
    prior: Array,
    use_prior: bool,
    mu1_kind: int,
    mu1_params: Array,
    sigma1_kind: int,
    sigma1_params: Array,
    mu2_kind: int,
    mu2_params: Array,
    sigma2_kind: int,
    sigma2_params: Array,
) -> float:
    """Return the first-pass MAP estimate with a numba-friendly inner loop."""
    nobs = rad1_vec.size
    ngrid = ra_grid.size
    if nobs == 0 or ngrid == 0:
        return math.nan

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

            mu1 = _model_value(mu1_kind, mu1_params, inc, ra)
            sigma1 = _model_value(sigma1_kind, sigma1_params, inc, ra) / 1.349
            if sigma1 < sigma_floor:
                sigma1 = sigma_floor
            sigma1 *= inflate

            mu2 = _model_value(mu2_kind, mu2_params, inc, ra)
            sigma2 = _model_value(sigma2_kind, sigma2_params, inc, ra) / 1.349
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

    _normalize_density_inplace(prior_work, ra_grid)
    ra_map, _, _, _ = _posterior_summary_from_loglikes(log_l1, log_l2, prior_work, ra_grid, s)
    return ra_map


@njit(cache=True)
def _invert_ra_from_rad_second_2rad_numba(
    rad1_vec: Array,
    rad2_vec: Array,
    inc_vec: Array,
    prior: Array,
    ra_grid: Array,
    nu: float,
    s: float,
    has_neff: bool,
    neff_value: float,
    mu1_kind: int,
    mu1_params: Array,
    sigma1_kind: int,
    sigma1_params: Array,
    mu2_kind: int,
    mu2_params: Array,
    sigma2_kind: int,
    sigma2_params: Array,
) -> tuple[float, float, float, float]:
    """Return second-pass posterior summaries with numba-friendly kernels."""
    nobs = rad1_vec.size
    ngrid = ra_grid.size
    if nobs == 0 or ngrid == 0:
        return (math.nan, math.nan, math.nan, math.nan)

    if has_neff:
        neff = neff_value
    else:
        neff = math.sqrt(nobs)
    if not math.isfinite(neff) or neff <= 0.0:
        neff = 1.0
    neff = min(neff, float(nobs))
    inflate = math.sqrt(nobs / neff)

    coeff = math.exp(math.lgamma((nu + 1.0) / 2.0) - math.lgamma(nu / 2.0))
    coeff /= math.sqrt(nu * math.pi)

    log_l1 = np.zeros(ngrid, dtype=np.float64)
    log_l2 = np.zeros(ngrid, dtype=np.float64)
    for k in range(nobs):
        inc = inc_vec[k]
        rad1 = rad1_vec[k]
        rad2 = rad2_vec[k]
        for i in range(ngrid):
            ra = ra_grid[i]

            mu1 = _model_value(mu1_kind, mu1_params, inc, ra)
            sigma1 = _model_value(sigma1_kind, sigma1_params, inc, ra) / 1.349
            if sigma1 < 0.02:
                sigma1 = 0.02
            sigma1 *= inflate
            z1 = (rad1 - mu1) / sigma1
            lk1 = coeff * ((1.0 + (z1 * z1) / nu) ** (-0.5 * (nu + 1.0))) / sigma1
            log_l1[i] += math.log(max(lk1, REALMIN))

            mu2 = _model_value(mu2_kind, mu2_params, inc, ra)
            sigma2 = _model_value(sigma2_kind, sigma2_params, inc, ra) / 1.349
            if sigma2 < 0.02:
                sigma2 = 0.02
            sigma2 *= inflate
            z2 = (rad2 - mu2) / sigma2
            lk2 = coeff * ((1.0 + (z2 * z2) / nu) ** (-0.5 * (nu + 1.0))) / sigma2
            log_l2[i] += math.log(max(lk2, REALMIN))

    prior_work = prior.copy()
    _normalize_density_inplace(prior_work, ra_grid)
    return _posterior_summary_from_loglikes(log_l1, log_l2, prior_work, ra_grid, s)


def load_parallel_scattering_models() -> dict[str, ModelFit]:
    """Return the hardcoded scattering-model fits used by the translated pipeline"""
    # These are the exact saved formulas and parameter vectors used by the matlab workflow
    return {
        "Rad1muFit": ModelFit(
            model_fun=_sigmoid_cosd_exp_ra,
            params=np.array(
                [-0.923350, 0.685613, 10.194197, 0.134646, 1.392564, 0.229107, 0.960787, -0.012927],
                dtype=float,
            ),
            model_kind=MODEL_SIGMOID_COSD_EXP_RA,
        ),
        "Rad1iqrFit": ModelFit(
            model_fun=_sigmoid_cosd_linear_ra,
            params=np.array(
                [0.0863205, 0.1447666, 21.9325666, 0.2945311, 0.8973019, 0.5230950, 0.0704902, 0.0090042],
                dtype=float,
            ),
            model_kind=MODEL_SIGMOID_COSD_LINEAR_RA,
        ),
        "Rad2muFit": ModelFit(
            model_fun=_sigmoid_cosd_linear_ra,
            params=np.array(
                [0.10502227, 0.50067719, 6.61343753, 0.13577265, 1.70816978, 0.00021586, 0.18486085, 0.00766573],
                dtype=float,
            ),
            model_kind=MODEL_SIGMOID_COSD_LINEAR_RA,
        ),
        "Rad2iqrFit": ModelFit(
            model_fun=_sigmoid_cosd_exp_ra,
            params=np.array(
                [0.0958137, 0.2591425, 17.0029951, 0.2713287, 3.0849616, -0.0583950, 0.0377035, 0.0016440],
                dtype=float,
            ),
            model_kind=MODEL_SIGMOID_COSD_EXP_RA,
        ),
    }


def load_analysis_csv(csv_path: str | Path, variable_names: Iterable[str] | None = None) -> dict[str, Array]:
    """Rebuild 2D analysis arrays from the flattened CSV export format."""
    csv_path = Path(csv_path).resolve()
    if variable_names is None:
        variable_names = ANALYSIS_VARIABLES

    variable_names = list(variable_names)
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


def load_analysis_dataset(data_path: str | Path, variable_names: Iterable[str] | None = None) -> dict[str, Array]:
    """Load an analysis dataset from CSV only."""
    data_path = Path(data_path)
    if variable_names is None:
        variable_names = ANALYSIS_VARIABLES

    if data_path.suffix.lower() != ".csv":
        raise ValueError(f"Unsupported data file type {data_path.suffix!r}. Expected .csv.")

    return load_analysis_csv(data_path, variable_names)


def preprocess_minirf_data(data: dict[str, Array]) -> dict[str, Array]:
    """Match the MATLAB preprocessing step for the Mini-RF analysis inputs."""
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

    mask = (
        (s["num"] == 1)
        & (s["ra"] > 0.01)
        & ~np.isnan(s["cpr"])
        & ~np.isnan(s["inc"])
        & (s["inc"] > 45.0)
        & (s["inc"] < 70.0)
    )

    for key in ("cpr", "g", "inc", "ra"):
        s[key][~mask] = np.nan

    s["Rad1"] = s["cpr"]
    s["Rad2"] = s["g"]
    s["IncidenceMap"] = s["inc"]
    s["RAMap"] = s["ra"]
    return s


def fit_local_scattering_curve_2rad_wbeta(
    rad1_vec: Array,
    rad2_vec: Array,
    inc_vec: Array,
    mu1_fit: ModelFit,
    sigma1_fit: ModelFit,
    mu2_fit: ModelFit,
    sigma2_fit: ModelFit,
    w_vec: Array,
    argopt: dict[str, Any],
) -> dict[str, Any]:
    """Run the first-pass local inversion for one weighted neighborhood."""
    argopt = dict(argopt)
    argopt.setdefault("sigmaFloor", 0.02)
    argopt.setdefault("alpha", 0.0)
    argopt.setdefault("RA0", 1.0)
    argopt.setdefault("NeffMode", "sqrt")
    argopt.setdefault("Neff", None)
    argopt.setdefault("useStudentT", False)
    argopt.setdefault("nu", 5)

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

    ra_grid = np.asarray(argopt.get("RA_grid", default_ra_grid()), dtype=float).reshape(-1)

    if rad1_vec.size == 0 or rad2_vec.size == 0:
        return {
            "MAP": np.nan,
            "med": np.nan,
            "low": np.nan,
            "high": np.nan,
            "post": np.full(ra_grid.shape, np.nan, dtype=float),
            "RA_grid": ra_grid,
        }

    prior_opt = argopt.get("prior")
    if prior_opt is not None:
        prior = np.asarray(prior_opt, dtype=float).reshape(-1)
        if prior.size != ra_grid.size:
            raise ValueError("argopt['prior'] must have the same length as RA_grid.")
        use_prior = True
    else:
        prior = np.empty(0, dtype=float)
        use_prior = False

    ra_map = _fit_local_scattering_curve_2rad_wbeta_numba(
        rad1_vec,
        rad2_vec,
        inc_vec,
        w_vec,
        ra_grid,
        float(argopt["sigmaFloor"]),
        float(argopt["alpha"]),
        float(argopt["RA0"]),
        float(argopt["s"]),
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
    return {"MAP": float(ra_map)}


def invert_ra_2rad_first_pass_spatial(
    rad1: Array,
    rad2: Array,
    ra_map: Array,
    incidence_map: Array,
    argopt_firstpass: dict[str, Any],
    n_workers: int | None = None,
) -> Array:
    """Compute the first-pass MAP RA map from spatial neighborhoods."""
    nrows, ncols = rad1.shape
    half_win = int(argopt_firstpass["halfWin"])
    sigma_dist = float(argopt_firstpass["sigmaDist"])
    rad1_mu_fit = argopt_firstpass["Rad1muFit"]
    rad1_iqr_fit = argopt_firstpass["Rad1iqrFit"]
    rad2_mu_fit = argopt_firstpass["Rad2muFit"]
    rad2_iqr_fit = argopt_firstpass["Rad2iqrFit"]

    rad1_vec = rad1.reshape(-1, order="F")
    rad2_vec = rad2.reshape(-1, order="F")
    inc_vec = incidence_map.reshape(-1, order="F")
    ra_vec = ra_map.reshape(-1, order="F")

    valid_pixel = np.isfinite(rad1_vec) & np.isfinite(rad2_vec) & np.isfinite(inc_vec) & np.isfinite(ra_vec)
    valid_idx = np.flatnonzero(valid_pixel)

    def work(idx: int) -> float:
        """Invert one valid pixel by fitting its local weighted neighborhood."""
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
            return np.nan

        rad1_local = rad1_local[valid]
        rad2_local = rad2_local[valid]
        inc_local = inc_local[valid]
        w_local = w_local[valid]

        nobs = rad1_local.size

        argopt_local = dict(argopt_firstpass)
        argopt_local["alpha"] = 0.034 * nobs + 0.106

        ra_pred = fit_local_scattering_curve_2rad_wbeta(
            rad1_local,
            rad2_local,
            inc_local,
            rad1_mu_fit,
            rad1_iqr_fit,
            rad2_mu_fit,
            rad2_iqr_fit,
            w_local,
            argopt_local,
        )
        return float(ra_pred["MAP"])

    if n_workers == 1:
        ra_pred_temp = np.array([work(idx) for idx in valid_idx], dtype=float)
    else:
        with ThreadPoolExecutor(max_workers=n_workers) as pool:
            ra_pred_temp = np.array(list(pool.map(work, valid_idx)), dtype=float)

    out_flat = np.full(rad1.size, np.nan, dtype=float)
    out_flat[valid_idx] = ra_pred_temp
    return out_flat.reshape(rad1.shape, order="F")


def _nanmedian_window(arr: Array, r0: int, r1: int, c0: int, c1: int) -> float:
    """Return the NaN-aware median of one 2D window."""
    window = arr[r0:r1, c0:c1]
    if np.all(np.isnan(window)):
        return np.nan
    return float(np.nanmedian(window))


def boxcar_median(arr: Array, window_size: int) -> Array:
    """Apply a 2D NaN-aware moving median with square windows."""
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
            out[r, c] = _nanmedian_window(arr, r0, r1, c0, c1)
    return out


def moving_median_1d(arr: Array, window_size: int, axis: int) -> Array:
    """Apply a 1D NaN-aware moving median along one axis."""
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


def invert_ra_from_rad_second_2rad(
    rad1: Array,
    rad2: Array,
    inc: Array,
    mu1_fit: ModelFit,
    sigma1_fit: ModelFit,
    mu2_fit: ModelFit,
    sigma2_fit: ModelFit,
    argopt: dict[str, Any],
) -> dict[str, Any]:
    """Run the second-pass pixelwise Bayesian inversion for one observation set."""
    prior = np.asarray(argopt["prior"], dtype=float).reshape(-1)

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
        return {"MAP": np.nan, "med": np.nan, "low": np.nan, "high": np.nan}

    ra_grid = np.asarray(argopt.get("RA_grid", default_ra_grid()), dtype=float).reshape(-1)
    if prior.size != ra_grid.size:
        raise ValueError("argopt['prior'] must have the same length as RA_grid.")

    has_neff = "Neff" in argopt and argopt["Neff"] is not None
    neff_value = float(argopt["Neff"]) if has_neff else 0.0
    ra_map, ra_med, ra_low, ra_high = _invert_ra_from_rad_second_2rad_numba(
        rad1,
        rad2,
        inc,
        prior,
        ra_grid,
        float(argopt["nu"]),
        float(argopt["s"]),
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
    )

    return {
        "MAP": float(ra_map),
        "med": float(ra_med),
        "low": float(ra_low),
        "high": float(ra_high),
        "RA_grid": ra_grid,
        "post1": None,
        "post2": None,
        "post": None,
    }


def invert_ra_2rad_second_pass_spatial(
    rad1: Array,
    rad2: Array,
    inc_obs: Array,
    ra_first: Array,
    argopt: dict[str, Any],
    n_workers: int | None = None,
) -> dict[str, Array]:
    """Refine the first-pass map with a spatial prior and pixelwise inversion."""
    argopt = dict(argopt)
    argopt.setdefault("windowSize", 5)
    argopt.setdefault("sigmaSpatial", 1.0)
    argopt.setdefault("spatialPriorMode", "direct")
    argopt.setdefault("useStudentT", False)
    argopt.setdefault("nu", 5)

    mu1_fit = argopt["Rad1muFit"]
    sigma1_fit = argopt["Rad1iqrFit"]
    mu2_fit = argopt["Rad2muFit"]
    sigma2_fit = argopt["Rad2iqrFit"]

    window_size = int(argopt["windowSize"])
    mode = str(argopt["spatialPriorMode"]).lower()
    if mode == "direct":
        ra_local = ra_first
    elif mode == "boxmedian":
        ra_local = boxcar_median(ra_first, window_size)
    elif mode == "movmedian2":
        ra_local = moving_median_1d(ra_first, window_size, axis=0)
        ra_local = moving_median_1d(ra_local, window_size, axis=1)
    else:
        raise ValueError("Unknown argopt['spatialPriorMode']. Use 'direct', 'boxmedian', or 'movmedian2'.")

    ra_grid = np.asarray(argopt.get("RA_grid", default_ra_grid()), dtype=float).reshape(-1)
    argopt_pix_template = dict(argopt)
    argopt_pix_template["RA_grid"] = ra_grid
    frac_spatial = float(argopt["sigmaSpatial"])

    rad1_vec = rad1.reshape(-1, order="F")
    rad2_vec = rad2.reshape(-1, order="F")
    inc_vec = inc_obs.reshape(-1, order="F")
    ra0_vec = ra_local.reshape(-1, order="F")

    valid_pixel = np.isfinite(rad1_vec) & np.isfinite(rad2_vec) & np.isfinite(inc_vec) & np.isfinite(ra0_vec)
    valid_idx = np.flatnonzero(valid_pixel)

    def work(idx: int) -> tuple[float, float, float, float]:
        """Refine one valid pixel in the second pass and return summary quantiles."""
        rad1_value = rad1_vec[idx]
        rad2_value = rad2_vec[idx]
        inc_value = inc_vec[idx]
        ra0 = ra0_vec[idx]

        sigma_spatial = frac_spatial * ra0
        if not np.isfinite(sigma_spatial) or sigma_spatial <= 0.0:
            return (np.nan, np.nan, np.nan, np.nan)

        prior = normpdf(ra_grid, ra0, sigma_spatial)
        prior = np.where(np.isfinite(prior) & (prior >= 0.0), prior, 0.0)
        if np.all(prior == 0.0):
            return (np.nan, np.nan, np.nan, np.nan)
        prior = prior / np.trapezoid(prior, ra_grid)

        argopt_pix_local = dict(argopt_pix_template)
        argopt_pix_local["prior"] = prior

        ra_pred = invert_ra_from_rad_second_2rad(
            rad1_value,
            rad2_value,
            inc_value,
            mu1_fit,
            sigma1_fit,
            mu2_fit,
            sigma2_fit,
            argopt_pix_local,
        )
        return (
            float(ra_pred["MAP"]),
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
    ra_med_vec = np.full(valid_idx.shape, np.nan, dtype=float)
    ra_low_vec = np.full(valid_idx.shape, np.nan, dtype=float)
    ra_high_vec = np.full(valid_idx.shape, np.nan, dtype=float)

    for i, (ra_second, ra_med, ra_low, ra_high) in enumerate(packed):
        ra_second_vec[i] = ra_second
        ra_med_vec[i] = ra_med
        ra_low_vec[i] = ra_low
        ra_high_vec[i] = ra_high

    def scatter(values: Array) -> Array:
        """Scatter a flat vector of valid-pixel results back into map shape."""
        flat = np.full(rad1.size, np.nan, dtype=float)
        flat[valid_idx] = values
        return flat.reshape(rad1.shape, order="F")

    return {
        "second": scatter(ra_second_vec),
        "med": scatter(ra_med_vec),
        "low": scatter(ra_low_vec),
        "high": scatter(ra_high_vec),
    }


def _nanpercentile(values: Array, q: float | list[float]) -> float | Array:
    """Compute percentiles after dropping non-finite values."""
    values = np.asarray(values, dtype=float)
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        if np.isscalar(q):
            return float("nan")
        return np.full(len(q), np.nan, dtype=float)
    return np.nanpercentile(finite, q)


def _pearson_corr(x: Array, y: Array) -> float:
    """Compute a finite-only Pearson correlation."""
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


def _rankdata_average(values: Array) -> Array:
    """Assign average ranks, matching the behavior needed for Spearman correlation."""
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


def _spearman_corr(x: Array, y: Array) -> float:
    """Compute a finite-only Spearman correlation."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    valid = np.isfinite(x) & np.isfinite(y)
    x = x[valid]
    y = y[valid]
    if x.size < 2:
        return float("nan")
    return _pearson_corr(_rankdata_average(x), _rankdata_average(y))


def _print_metrics_summary(metrics: dict[str, Any]) -> None:
    """Print the compact evaluation report used by the MATLAB driver."""
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
    """Evaluate a predicted RA map against a reference map."""
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
    ra_true_v = np.asarray(ra_true, dtype=float).reshape(-1, order="F")
    ra_pred_v = np.asarray(ra_pred, dtype=float).reshape(-1, order="F")

    valid = np.isfinite(ra_true_v) & np.isfinite(ra_pred_v)
    ra_true_v = ra_true_v[valid]
    ra_pred_v = ra_pred_v[valid]

    resid = ra_true_v - ra_pred_v
    abs_err = np.abs(resid)

    metrics: dict[str, Any] = {"N": int(ra_true_v.size), "originalSize": orig_size, "global": {}, "highRA": {}, "spatial": {}}

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


def build_default_pipeline_options(models: dict[str, ModelFit] | None = None) -> dict[str, dict[str, Any]]:
    """Build the default first-pass, second-pass, and evaluation option sets."""
    if models is None:
        models = load_parallel_scattering_models()

    ra_grid = default_ra_grid(ra_min=0.01, ra_max=10.0, n_grid_points=500)
    window_size = 3
    half_win = window_size // 2
    sigma_dist = window_size / 3.0

    argopt_firstpass: dict[str, Any] = {
        "s": 1.5,
        "RA_grid": ra_grid,
        "sigmaFloor": 0.02,
        "RA0": 1.0,
        "halfWin": half_win,
        "sigmaDist": sigma_dist,
        "Rad1muFit": models["Rad1muFit"],
        "Rad1iqrFit": models["Rad1iqrFit"],
        "Rad2muFit": models["Rad2muFit"],
        "Rad2iqrFit": models["Rad2iqrFit"],
    }

    arg_second_opt = dict(argopt_firstpass)
    arg_second_opt["spatialPriorMode"] = "direct"
    arg_second_opt["nu"] = 3
    arg_second_opt["sigmaSpatial"] = 0.5
    arg_second_opt["s"] = argopt_firstpass["s"]

    eval_argopt: dict[str, Any] = {
        "regimeEdges": [0.0, 0.35, 1.0, 3.0, 5.0, 12.0],
        "regimeNames": ["Low RA", "Transition", "High RA1", "High RA2", "High RA3"],
        "smoothWindow": 5,
        "mapMode": True,
        "report": True,
    }

    return {"firstpass": argopt_firstpass, "secondpass": arg_second_opt, "eval": eval_argopt}


def predict_ra_from_preprocessed_with_firstpass(
    s: dict[str, Array],
    ra_pred_map: Array,
    secondpass_options: dict[str, Any],
    n_workers: int | None = None,
) -> dict[str, Any]:
    """Run the second pass from a precomputed first-pass RA map."""
    ra_pred_maps = invert_ra_2rad_second_pass_spatial(
        s["Rad1"],
        s["Rad2"],
        s["IncidenceMap"],
        ra_pred_map,
        secondpass_options,
        n_workers=n_workers,
    )
    ra_pred = ra_pred_maps["med"]

    return {
        "S": s,
        "RA_pred_MAP": ra_pred_map,
        "RApredMaps": ra_pred_maps,
        "RApred": ra_pred,
    }


def predict_ra_from_preprocessed(
    s: dict[str, Array],
    firstpass_options: dict[str, Any],
    secondpass_options: dict[str, Any],
    n_workers: int | None = None,
) -> dict[str, Any]:
    """Run the translated two-pass prediction pipeline on preprocessed arrays."""
    ra_pred_map = invert_ra_2rad_first_pass_spatial(
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
        secondpass_options,
        n_workers=n_workers,
    )


def driver_pred_ra_from_radar(
    data_path: str | Path,
    n_workers: int | None = None,
) -> dict[str, Any]:
    """Load data, run the translated pipeline, and return prediction outputs."""
    models = load_parallel_scattering_models()
    data = load_analysis_dataset(data_path, ANALYSIS_VARIABLES)
    s = preprocess_minirf_data(data)
    pipeline_options = build_default_pipeline_options(models)
    result = predict_ra_from_preprocessed(
        s,
        pipeline_options["firstpass"],
        pipeline_options["secondpass"],
        n_workers=n_workers,
    )
    metrics_cpr = evaluate_ra_prediction(s["ra"], result["RApred"], pipeline_options["eval"])
    result["metrics_cpr"] = metrics_cpr

    return result


def main() -> None:
    """Run the standalone translated driver from the hardcoded top-level settings."""
    driver_pred_ra_from_radar(data_path, n_workers=workers)


if __name__ == "__main__":
    main()
