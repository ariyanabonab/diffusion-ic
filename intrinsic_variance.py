"""
Uncertainty quantification for IC reconstruction, matching the formulas at:
https://hackmd.io/@maho3/H1vLErHBMx

Setup
-----
N independent test points i = 1..N, each with:
  - one true value      x_i^true
  - M posterior samples  x_{i,m}^pred  (m = 1..M), drawn from the model's
    posterior for that point.

A "point" can be a whole reconstructed field (N = number of test simulations)
or a single voxel (N = number of voxels pooled across simulations, e.g. all
voxels landing in one present-day-overdensity bin, as in Figure 4). The
functions below work either way — just hand them arrays shaped (N, ...) or
(N, M, ...) and the extra trailing dimensions (voxels, k-bins, etc.) are
carried through elementwise.

Key formulas
------------
True values:
    mu_true    = mean_i( x_i^true )
    sigma_true = std_i( x_i^true ), ddof=1

Predicted values:
    xbar_i = mean_m( x_{i,m}^pred )                      # posterior mean per point
    v_i    = var_m( x_{i,m}^pred ), ddof=1                # posterior variance per point

    mu_pred      = mean_i( xbar_i )
    sigma2_within = mean_i( v_i )                          # avg posterior width
    sigma2_raw    = var_i( xbar_i ), ddof=1                 # raw spread of the means

    sigma2_intrinsic = sigma2_raw - sigma2_within / M       # <-- the correction
    sigma_intrinsic  = sqrt(max(sigma2_intrinsic, 0))

sigma_intrinsic is the number that's comparable to sigma_true. It is NOT
"pooled variance minus between variance" — it corrects the variance of the
posterior MEANS for the extra noise from averaging only M draws.
"""

import numpy as np


# ============================================================
# True-value statistics
# ============================================================

def true_value_stats(true_values, ddof=1):
    """
    true_values : ndarray, shape (N, *extra)

    Returns mu_true, se_mu_true, sigma_true — each of shape (*extra,).
    """
    true_values = np.asarray(true_values)
    n = true_values.shape[0]

    mu_true = true_values.mean(axis=0)
    sigma_true = true_values.std(axis=0, ddof=ddof)
    se_mu_true = sigma_true / np.sqrt(n)

    return {
        "mu_true": mu_true,
        "se_mu_true": se_mu_true,
        "sigma_true": sigma_true,
    }


def bootstrap_true_std(true_values, n_boot=10_000, ddof=1, rng=None):
    """
    Bootstrap SE on sigma_true by resampling the N points with replacement.

    true_values : ndarray, shape (N, *extra)
    Returns se_boot_sigma_true, shape (*extra,).
    """
    true_values = np.asarray(true_values)
    n = true_values.shape[0]
    rng = np.random.default_rng() if rng is None else rng

    boot_stds = np.empty((n_boot,) + true_values.shape[1:])
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        boot_stds[b] = true_values[idx].std(axis=0, ddof=ddof)

    return boot_stds.std(axis=0, ddof=1)


# ============================================================
# Predicted-value statistics (the corrected formula)
# ============================================================

def predicted_value_stats(pred_draws, ddof_within=1, ddof_between=1):
    """
    pred_draws : ndarray, shape (N, M, *extra)
        pred_draws[i, m] = the m-th posterior draw for point i.

    Returns per-point arrays (xbar_i, v_i) and the aggregate statistics:
    mu_pred, se_mu_pred, sigma2_within, sigma2_raw, sigma2_intrinsic,
    sigma_intrinsic.
    """
    pred_draws = np.asarray(pred_draws)
    n, m = pred_draws.shape[0], pred_draws.shape[1]

    xbar_i = pred_draws.mean(axis=1)                       # (N, *extra) - posterior mean per point
    v_i = pred_draws.var(axis=1, ddof=ddof_within)          # (N, *extra) - posterior variance per point

    mu_pred = xbar_i.mean(axis=0)
    sigma2_within = v_i.mean(axis=0)
    sigma2_raw = xbar_i.var(axis=0, ddof=ddof_between)

    se_mu_pred = np.sqrt(sigma2_raw / n)

    sigma2_intrinsic = sigma2_raw - sigma2_within / m
    sigma2_intrinsic = np.clip(sigma2_intrinsic, a_min=0, a_max=None)
    sigma_intrinsic = np.sqrt(sigma2_intrinsic)

    return {
        "xbar_i": xbar_i,
        "v_i": v_i,
        "mu_pred": mu_pred,
        "se_mu_pred": se_mu_pred,
        "sigma2_within": sigma2_within,
        "sigma2_raw": sigma2_raw,
        "sigma2_intrinsic": sigma2_intrinsic,
        "sigma_intrinsic": sigma_intrinsic,
    }


def bootstrap_predicted(pred_draws, n_boot=10_000, ddof_within=1, ddof_between=1, rng=None):
    """
    Hierarchical bootstrap: resample the N points with replacement, keeping
    each resampled point's full set of M draws intact (do NOT resample the
    draws themselves). Matches the doc's prescription.

    pred_draws : ndarray, shape (N, M, *extra)
    Returns se_boot_mu_pred, se_boot_sigma_intrinsic — each shape (*extra,).
    """
    pred_draws = np.asarray(pred_draws)
    n = pred_draws.shape[0]
    rng = np.random.default_rng() if rng is None else rng

    boot_mu = np.empty((n_boot,) + pred_draws.shape[2:])
    boot_sigma_intrinsic = np.empty((n_boot,) + pred_draws.shape[2:])

    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)          # resample POINTS, not draws
        resampled = pred_draws[idx]                # (N, M, *extra), each point's M draws intact
        stats = predicted_value_stats(resampled, ddof_within=ddof_within, ddof_between=ddof_between)
        boot_mu[b] = stats["mu_pred"]
        boot_sigma_intrinsic[b] = stats["sigma_intrinsic"]

    se_boot_mu_pred = boot_mu.std(axis=0, ddof=1)
    se_boot_sigma_intrinsic = boot_sigma_intrinsic.std(axis=0, ddof=1)

    return se_boot_mu_pred, se_boot_sigma_intrinsic


# ============================================================
# Convenience wrapper: full comparison, true vs. predicted
# ============================================================

def compare_true_vs_predicted(true_values, pred_draws, n_boot=10_000, rng=None):
    """
    true_values : ndarray, shape (N, *extra)
    pred_draws  : ndarray, shape (N, M, *extra)

    Returns a dict with mu_true, sigma_true, mu_pred, sigma_intrinsic, their
    error bars, and the residuals (delta_mu, delta_sigma) — i.e. everything
    needed to reproduce the four panels in Figure 4 (mu_x, delta_mu_x,
    sigma_x, delta_sigma_x) for a single bin or a single global point.
    """
    true_stats = true_value_stats(true_values)
    pred_stats = predicted_value_stats(pred_draws)

    se_boot_sigma_true = bootstrap_true_std(true_values, n_boot=n_boot, rng=rng)
    se_boot_mu_pred, se_boot_sigma_intrinsic = bootstrap_predicted(pred_draws, n_boot=n_boot, rng=rng)

    delta_mu = pred_stats["mu_pred"] - true_stats["mu_true"]
    delta_sigma = pred_stats["sigma_intrinsic"] - true_stats["sigma_true"]

    return {
        "mu_true": true_stats["mu_true"],
        "se_mu_true": true_stats["se_mu_true"],
        "sigma_true": true_stats["sigma_true"],
        "se_boot_sigma_true": se_boot_sigma_true,
        "mu_pred": pred_stats["mu_pred"],
        "se_mu_pred": pred_stats["se_mu_pred"],          # analytic; se_boot_mu_pred is the bootstrap version
        "se_boot_mu_pred": se_boot_mu_pred,
        "sigma_intrinsic": pred_stats["sigma_intrinsic"],
        "se_boot_sigma_intrinsic": se_boot_sigma_intrinsic,
        "delta_mu": delta_mu,
        "delta_sigma": delta_sigma,
    }


# ============================================================
# Binning by a conditioning variable (e.g. present-day overdensity y)
# ============================================================

def bin_and_compute(y_values, true_values, pred_draws, n_bins=40,
                     binning="quantile", min_points=10, n_boot=300, rng=None):
    """
    Reproduces the Figure-4 workflow: bins individual points (typically
    voxels, pooled across all your test fields) by a conditioning variable y
    (e.g. the FD/present-day-overdensity value at that voxel), then applies
    compare_true_vs_predicted() independently within each bin.

    y_values    : ndarray, shape (V,)     — binning variable, one per point
    true_values : ndarray, shape (V,)     — true value, one per point
    pred_draws  : ndarray, shape (V, M)   — M posterior draws per point
    n_bins      : target number of bins
    binning     : "quantile" (equal points per bin, robust to skewed y) or
                  "linear" (equal-width bins in y, matches the figure's even
                  x-axis spacing more closely, but can leave sparse bins)
    min_points  : bins with fewer points than this are skipped (too noisy
                  to trust, especially the bootstrap)
    n_boot      : bootstrap replicates PER BIN — kept modest by default
                  since this runs 40x; raise it once you're happy with the
                  binning and want tighter error bars on the final plot.

    Returns a dict of 1D arrays (one entry per surviving bin): bin_center,
    n_points, mu_true, se_mu_true, sigma_true, se_boot_sigma_true, mu_pred,
    se_mu_pred, se_boot_mu_pred, sigma_intrinsic, se_boot_sigma_intrinsic,
    delta_mu, delta_sigma.
    """
    y_values = np.asarray(y_values)
    true_values = np.asarray(true_values)
    pred_draws = np.asarray(pred_draws)
    rng = np.random.default_rng() if rng is None else rng

    if binning == "quantile":
        edges = np.quantile(y_values, np.linspace(0, 1, n_bins + 1))
        edges = np.unique(edges)  # guard against repeated values collapsing bins
    elif binning == "linear":
        edges = np.linspace(y_values.min(), y_values.max(), n_bins + 1)
    else:
        raise ValueError("binning must be 'quantile' or 'linear'")

    bin_idx = np.digitize(y_values, edges[1:-1], right=False)

    keys = ["bin_center", "n_points", "mu_true", "se_mu_true", "sigma_true",
            "se_boot_sigma_true", "mu_pred", "se_mu_pred", "se_boot_mu_pred",
            "sigma_intrinsic", "se_boot_sigma_intrinsic", "delta_mu", "delta_sigma"]
    results = {k: [] for k in keys}

    n_bins_actual = len(edges) - 1
    for b in range(n_bins_actual):
        mask = bin_idx == b
        n_pts = int(mask.sum())
        if n_pts < min_points:
            continue

        tb = true_values[mask]
        pb = pred_draws[mask]  # (n_pts, M)
        stats = compare_true_vs_predicted(tb, pb, n_boot=n_boot, rng=rng)

        results["bin_center"].append(0.5 * (edges[b] + edges[b + 1]))
        results["n_points"].append(n_pts)
        for k in keys[2:]:
            results[k].append(float(stats[k]))

    return {k: np.array(v) for k, v in results.items()}


def plot_binned_comparison(results, xlabel="Present-day overdensity y", save_path=None):
    """
    Four-panel plot matching Figure 4: mean (top), delta-mean, stdev,
    delta-stdev — truth in black diamonds, inferred in blue circles.

    results : the dict returned by bin_and_compute()
    """
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(4, 1, figsize=(8, 12), sharex=True,
                              gridspec_kw={"height_ratios": [3, 1.5, 3, 1.5]})
    x = results["bin_center"]

    ax = axes[0]
    ax.errorbar(x, results["mu_pred"], yerr=results["se_boot_mu_pred"], fmt="o",
                color="tab:blue", markersize=5, capsize=2, alpha=0.85, label="Inferred")
    ax.errorbar(x, results["mu_true"], yerr=results["se_mu_true"], fmt="D",
                color="black", markersize=4, capsize=2, label="Truth")
    ax.set_ylabel(r"Mean $\mu_x$")
    ax.legend(fontsize=10)
    ax.grid(alpha=0.2)

    ax = axes[1]
    ax.errorbar(x, results["delta_mu"], yerr=results["se_boot_mu_pred"], fmt="o",
                color="tab:blue", markersize=5, capsize=2, alpha=0.85)
    ax.axhline(0, color="black", linestyle="--", linewidth=1)
    ax.set_ylabel(r"$\Delta \mu_x$")
    ax.grid(alpha=0.2)

    ax = axes[2]
    ax.errorbar(x, results["sigma_intrinsic"], yerr=results["se_boot_sigma_intrinsic"], fmt="o",
                color="tab:blue", markersize=5, capsize=2, alpha=0.85, label="Inferred")
    ax.errorbar(x, results["sigma_true"], yerr=results["se_boot_sigma_true"], fmt="D",
                color="black", markersize=4, capsize=2, label="Truth")
    ax.set_ylabel(r"Standard deviation $\sigma_x$")
    ax.grid(alpha=0.2)

    ax = axes[3]
    ax.errorbar(x, results["delta_sigma"], yerr=results["se_boot_sigma_intrinsic"], fmt="o",
                color="tab:blue", markersize=5, capsize=2, alpha=0.85)
    ax.axhline(0, color="black", linestyle="--", linewidth=1)
    ax.set_ylabel(r"$\Delta \sigma_x$")
    ax.set_xlabel(xlabel)
    ax.grid(alpha=0.2)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
    return fig
