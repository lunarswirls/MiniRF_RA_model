#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Compare lookup-table versus smooth-fit and posterior mean versus median RA distributions
"""

from pathlib import Path
from typing import Any
import numpy as np
from scipy.stats import ks_2samp
import matplotlib.pyplot as plt
import parallel_ra_prediction_v2 as pr


Array = np.ndarray

# path to the v2 input dataset
data_path = Path(__file__).resolve().parents[1] / "data/GiordanoBruno.mat"

# number of worker threads for the prediction pipeline
n_workers = 20

# output path for the lookup versus smooth comparison figure
output_plot_path = Path(__file__).resolve().parents[1] / "output/lookup_vs_smooth_ra_distribution.png"

# output path for the posterior mean versus median comparison figure
mean_median_plot_path = Path(__file__).resolve().parents[1] / "output/posterior_mean_vs_median_ra_distribution.png"

# tail thresholds to compare explicitly
tail_thresholds = [0.35, 1.0, 3.0, 5.0]


def _distribution_summary(values: Array, reference: Array) -> dict[str, Any]:
    """summarize one predicted ra distribution against the diviner reference"""
    values = np.asarray(values, dtype=float).reshape(-1)
    reference = np.asarray(reference, dtype=float).reshape(-1)
    valid = np.isfinite(values) & np.isfinite(reference)
    values = values[valid]
    reference = reference[valid]
    if values.size == 0 or reference.size == 0:
        return {
            "n": 0,
            "median": float("nan"),
            "p90": float("nan"),
            "p99": float("nan"),
            "mean": float("nan"),
            "ks_stat": float("nan"),
            "tail_fractions": {},
            "tail_fraction_deltas": {},
        }

    summary = {
        "n": int(values.size),
        "median": float(np.nanmedian(values)),
        "p90": float(np.nanpercentile(values, 90.0)),
        "p99": float(np.nanpercentile(values, 99.0)),
        "mean": float(np.nanmean(values)),
        "ks_stat": float(ks_2samp(values, reference).statistic),
        "tail_fractions": {},
        "tail_fraction_deltas": {},
    }

    for threshold in tail_thresholds:
        pred_frac = float(np.nanmean(values >= threshold))
        ref_frac = float(np.nanmean(reference >= threshold))
        summary["tail_fractions"][threshold] = pred_frac
        summary["tail_fraction_deltas"][threshold] = pred_frac - ref_frac

    return summary


def _make_distribution_plot(ra_true: Array, ra_pred_lookup: Array, ra_pred_smooth: Array, output_path: str | Path) -> Path:
    """save an overlay of the reference, lookup, and smooth-fit ra distributions"""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    ra_true = np.asarray(ra_true, dtype=float)
    ra_pred_lookup = np.asarray(ra_pred_lookup, dtype=float)
    ra_pred_smooth = np.asarray(ra_pred_smooth, dtype=float)

    valid = np.isfinite(ra_true)
    ra_true_valid = ra_true[valid]
    ra_lookup_valid = ra_pred_lookup[valid & np.isfinite(ra_pred_lookup)]
    ra_smooth_valid = ra_pred_smooth[valid & np.isfinite(ra_pred_smooth)]

    hist_max = float(np.nanmax(ra_true_valid))
    if ra_lookup_valid.size > 0:
        hist_max = max(hist_max, float(np.nanmax(ra_lookup_valid)))
    if ra_smooth_valid.size > 0:
        hist_max = max(hist_max, float(np.nanmax(ra_smooth_valid)))
    hist_max = max(hist_max, 10.0)

    bins_full = np.linspace(0.0, hist_max, 120)
    bins_tail = np.linspace(0.35, min(hist_max, 10.0), 120)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), constrained_layout=True)
    ax_full = axes[0]
    ax_tail = axes[1]

    ax_full.hist(ra_true_valid, bins=bins_full, color="#264653", alpha=0.35, label="reference RA")
    ax_full.hist(ra_pred_lookup[np.isfinite(ra_pred_lookup)], bins=bins_full, color="#e76f51", alpha=0.5, label="lookup rho")
    ax_full.hist(ra_pred_smooth[np.isfinite(ra_pred_smooth)], bins=bins_full, color="#2a9d8f", alpha=0.5, label="smooth rho")
    ax_full.set_xlabel("RA (%)")
    ax_full.set_ylabel("count (log10 scale)")
    ax_full.set_yscale("log")
    ax_full.set_title("full RA distribution")
    ax_full.grid(True, alpha=0.25)
    ax_full.legend(loc="best", frameon=True)

    ax_tail.hist(ra_true_valid[ra_true_valid >= 0.35], bins=bins_tail, color="#264653", alpha=0.35, label="reference RA")
    ax_tail.hist(ra_lookup_valid[ra_lookup_valid >= 0.35], bins=bins_tail, color="#e76f51", alpha=0.5, label="lookup rho")
    ax_tail.hist(ra_smooth_valid[ra_smooth_valid >= 0.35], bins=bins_tail, color="#2a9d8f", alpha=0.5, label="smooth rho")
    ax_tail.set_xlabel("RA (%)")
    ax_tail.set_ylabel("count (log10 scale)")
    ax_tail.set_yscale("log")
    ax_tail.set_title("tail comparison for RA >= 0.35%")
    ax_tail.grid(True, alpha=0.25)
    ax_tail.legend(loc="best", frameon=True)

    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    return output_path


def _make_mean_median_plot(ra_true: Array, smooth_median: Array, smooth_mean: Array, lookup_median: Array,
                           lookup_mean: Array, output_path: str | Path) -> Path:
    """save mean-versus-median distribution overlays for smooth and lookup covariance modes"""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    ra_true = np.asarray(ra_true, dtype=float)
    smooth_median = np.asarray(smooth_median, dtype=float)
    smooth_mean = np.asarray(smooth_mean, dtype=float)
    lookup_median = np.asarray(lookup_median, dtype=float)
    lookup_mean = np.asarray(lookup_mean, dtype=float)

    valid = np.isfinite(ra_true)
    ra_true_valid = ra_true[valid]
    hist_max = float(np.nanmax(ra_true_valid))
    for values in (smooth_median, smooth_mean, lookup_median, lookup_mean):
        finite_values = values[np.isfinite(values)]
        if finite_values.size > 0:
            hist_max = max(hist_max, float(np.nanmax(finite_values)))
    hist_max = max(hist_max, 10.0)

    bins_full = np.linspace(0.0, hist_max, 120)
    bins_tail = np.linspace(0.35, min(hist_max, 10.0), 120)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10), constrained_layout=True)
    panel_specs = [
        (axes[0, 0], smooth_median, smooth_mean, "smooth rho full distribution", bins_full, None),
        (axes[0, 1], smooth_median, smooth_mean, "smooth rho tail for RA >= 0.35%", bins_tail, 0.35),
        (axes[1, 0], lookup_median, lookup_mean, "lookup rho full distribution", bins_full, None),
        (axes[1, 1], lookup_median, lookup_mean, "lookup rho tail for RA >= 0.35%", bins_tail, 0.35),
    ]

    for ax, median_values, mean_values, title, bins, threshold in panel_specs:
        if threshold is None:
            ref_values = ra_true_valid
            median_plot = median_values[np.isfinite(median_values)]
            mean_plot = mean_values[np.isfinite(mean_values)]
        else:
            ref_values = ra_true_valid[ra_true_valid >= threshold]
            median_plot = median_values[np.isfinite(median_values) & (median_values >= threshold)]
            mean_plot = mean_values[np.isfinite(mean_values) & (mean_values >= threshold)]

        ax.hist(ref_values, bins=bins, color="#264653", alpha=0.35, label="reference RA")
        ax.hist(median_plot, bins=bins, color="#e76f51", alpha=0.5, label="posterior median")
        ax.hist(mean_plot, bins=bins, color="#2a9d8f", alpha=0.5, label="posterior mean")
        ax.set_xlabel("RA (%)")
        ax.set_ylabel("count (log10 scale)")
        ax.set_yscale("log")
        ax.set_title(title)
        ax.grid(True, alpha=0.25)
        ax.legend(loc="best", frameon=True)

    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    return output_path


def run_lookup_vs_smooth_distribution_comparison() -> dict[str, Any]:
    """run one shared first pass and compare lookup versus smooth second-pass distributions"""
    models = pr.load_parallel_scattering_models()
    cov_model = pr.load_covariance_model()
    data = pr.load_analysis_dataset(data_path, list(pr.ANALYSIS_VARIABLES))
    s = pr.preprocess_minirf_data(data)
    pipeline_options = pr.build_default_pipeline_options(models, cov_model)

    print("Running shared first pass", flush=True)
    ra_pred_map, flag_map = pr.invert_ra_2rad_first_pass_spatial(
        s["Rad1"],
        s["Rad2"],
        s["RAMap"],
        s["IncidenceMap"],
        pipeline_options["firstpass"],
        n_workers=n_workers,
    )

    smooth_options = dict(pipeline_options["secondpass"])
    smooth_options["covarianceMode"] = "smooth"
    print("Running smooth-fit second pass", flush=True)
    smooth_result = pr.predict_ra_from_preprocessed_with_firstpass(
        s,
        ra_pred_map,
        flag_map,
        smooth_options,
        n_workers=n_workers,
    )

    lookup_options = dict(pipeline_options["secondpass"])
    lookup_options["covarianceMode"] = "lookup"
    print("Running lookup-table second pass", flush=True)
    lookup_result = pr.predict_ra_from_preprocessed_with_firstpass(
        s,
        ra_pred_map,
        flag_map,
        lookup_options,
        n_workers=n_workers,
    )

    smooth_summary = _distribution_summary(smooth_result["RApred"], s["ra"])
    smooth_mean_summary = _distribution_summary(smooth_result["RApredMaps"]["mean"], s["ra"])
    lookup_summary = _distribution_summary(lookup_result["RApred"], s["ra"])
    lookup_mean_summary = _distribution_summary(lookup_result["RApredMaps"]["mean"], s["ra"])
    reference_valid = np.asarray(s["ra"], dtype=float)[np.isfinite(s["ra"])]

    plot_path = _make_distribution_plot(s["ra"], lookup_result["RApred"], smooth_result["RApred"], output_plot_path)
    mean_median_plot = _make_mean_median_plot(
        s["ra"],
        smooth_result["RApredMaps"]["med"],
        smooth_result["RApredMaps"]["mean"],
        lookup_result["RApredMaps"]["med"],
        lookup_result["RApredMaps"]["mean"],
        mean_median_plot_path,
    )

    print(f"Saved lookup-vs-smooth distribution plot to {plot_path}", flush=True)
    print(f"Saved posterior mean-vs-median distribution plot to {mean_median_plot}", flush=True)
    print(f"Reference valid count: {reference_valid.size}", flush=True)
    print(f"Smooth valid count: {smooth_summary['n']}", flush=True)
    print(f"Lookup valid count: {lookup_summary['n']}", flush=True)
    print(f"Smooth median/p90/p99: {smooth_summary['median']:.4f} {smooth_summary['p90']:.4f} {smooth_summary['p99']:.4f}", flush=True)
    print(f"Smooth mean/p90/p99: {smooth_mean_summary['median']:.4f} {smooth_mean_summary['p90']:.4f} {smooth_mean_summary['p99']:.4f}", flush=True)
    print(f"Lookup median/p90/p99: {lookup_summary['median']:.4f} {lookup_summary['p90']:.4f} {lookup_summary['p99']:.4f}", flush=True)
    print(f"Lookup mean/p90/p99: {lookup_mean_summary['median']:.4f} {lookup_mean_summary['p90']:.4f} {lookup_mean_summary['p99']:.4f}", flush=True)
    print(f"Smooth median KS to reference: {smooth_summary['ks_stat']:.6f}", flush=True)
    print(f"Smooth mean KS to reference: {smooth_mean_summary['ks_stat']:.6f}", flush=True)
    print(f"Lookup median KS to reference: {lookup_summary['ks_stat']:.6f}", flush=True)
    print(f"Lookup mean KS to reference: {lookup_mean_summary['ks_stat']:.6f}", flush=True)

    for threshold in tail_thresholds:
        ref_frac = float(np.nanmean(reference_valid >= threshold))
        smooth_frac = smooth_summary["tail_fractions"][threshold]
        smooth_mean_frac = smooth_mean_summary["tail_fractions"][threshold]
        lookup_frac = lookup_summary["tail_fractions"][threshold]
        lookup_mean_frac = lookup_mean_summary["tail_fractions"][threshold]
        print(
            f"Tail fraction >= {threshold:.2f}% RA | reference={ref_frac:.6f} "
            f"smooth_median={smooth_frac:.6f} smooth_mean={smooth_mean_frac:.6f} "
            f"lookup_median={lookup_frac:.6f} lookup_mean={lookup_mean_frac:.6f}",
            flush=True,
        )

    return {
        "smooth": smooth_result,
        "lookup": lookup_result,
        "smooth_summary": smooth_summary,
        "smooth_mean_summary": smooth_mean_summary,
        "lookup_summary": lookup_summary,
        "lookup_mean_summary": lookup_mean_summary,
        "plot_path": plot_path,
        "mean_median_plot": mean_median_plot,
    }


if __name__ == "__main__":
    run_lookup_vs_smooth_distribution_comparison()
