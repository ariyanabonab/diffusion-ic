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

sigma_intrinsic is NOT a per-point posterior width. It is the (finite-M
corrected) spread of the posterior MEANS across different points i. It
answers "how much does the model's mean prediction vary from one point to
the next", not "how wide is the model's posterior for a single point".

This module ALSO carries a total-variance-law "sigma_combined" quantity
throughout, for internal diagnostic use, but per the reference doc above
the quantity actually meant to be compared to sigma_true is sigma_intrinsic
alone (see plot_binned_comparison docstring for the full note on this).

Calibration check (sigma_combined) — diagnostic only
-----------------------------------------------------
By the law of total variance, if the model's posterior for point i truly
matches the underlying conditional distribution p(x_i | condition_i):

    Var_i(x_i^true)  ~=  Var_i( E[x_i|condition_i] )  +  E_i[ Var(x_i|condition_i) ]
       sigma_true^2          sigma_intrinsic^2                  sigma2_within

sigma_combined = sqrt(sigma2_intrinsic + sigma2_within) is a valid quantity
under this decomposition, but it is NOT what the reference doc asks for —
it answers a different question ("does mean + width together reproduce the
truth's total spread") than sigma_intrinsic does ("does the diversity of the
model's point estimates alone reproduce the truth's spread"). Keep both
available, but sigma_intrinsic vs sigma_true is the primary comparison.
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
    sigma_intrinsic, sigma2_combined, sigma_combined.
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

    # Diagnostic-only total-variance-law estimate (see module docstring).
    sigma2_combined = sigma2_intrinsic + sigma2_within
    sigma_combined = np.sqrt(sigma2_combined)

    return {
        "xbar_i": xbar_i,
        "v_i": v_i,
        "mu_pred": mu_pred,
        "se_mu_pred": se_mu_pred,
        "sigma2_within": sigma2_within,
        "sigma2_raw": sigma2_raw,
        "sigma2_intrinsic": sigma2_intrinsic,
        "sigma_intrinsic": sigma_intrinsic,
        "sigma2_combined": sigma2_combined,
        "sigma_combined": sigma_combined,
    }


def bootstrap_predicted(pred_draws, n_boot=10_000, ddof_within=1, ddof_between=1, rng=None):
    """
    Hierarchical bootstrap: resample the N points with replacement, keeping
    each resampled point's full set of M draws intact (do NOT resample the
    draws themselves). Matches the doc's prescription.

    pred_draws : ndarray, shape (N, M, *extra)
    Returns se_boot_mu_pred, se_boot_sigma_intrinsic, se_boot_sigma_combined
    — each shape (*extra,).

    sigma_intrinsic and sigma_combined are bootstrapped jointly (from the
    same resampled draws each iteration) rather than combining their SEs
    after the fact, since sigma2_intrinsic and sigma2_within are correlated
    within a resample.
    """
    pred_draws = np.asarray(pred_draws)
    n = pred_draws.shape[0]
    rng = np.random.default_rng() if rng is None else rng

    boot_mu = np.empty((n_boot,) + pred_draws.shape[2:])
    boot_sigma_intrinsic = np.empty((n_boot,) + pred_draws.shape[2:])
    boot_sigma_combined = np.empty((n_boot,) + pred_draws.shape[2:])

    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)          # resample POINTS, not draws
        resampled = pred_draws[idx]                # (N, M, *extra), each point's M draws intact
        stats = predicted_value_stats(resampled, ddof_within=ddof_within, ddof_between=ddof_between)
        boot_mu[b] = stats["mu_pred"]
        boot_sigma_intrinsic[b] = stats["sigma_intrinsic"]
        boot_sigma_combined[b] = stats["sigma_combined"]

    se_boot_mu_pred = boot_mu.std(axis=0, ddof=1)
    se_boot_sigma_intrinsic = boot_sigma_intrinsic.std(axis=0, ddof=1)
    se_boot_sigma_combined = boot_sigma_combined.std(axis=0, ddof=1)

    return se_boot_mu_pred, se_boot_sigma_intrinsic, se_boot_sigma_combined


# ============================================================
# Convenience wrapper: full comparison, true vs. predicted
# ============================================================

def compare_true_vs_predicted(true_values, pred_draws, n_boot=10_000, rng=None):
    """
    true_values : ndarray, shape (N, *extra)
    pred_draws  : ndarray, shape (N, M, *extra)

    Returns a dict with mu_true, sigma_true, mu_pred, sigma_intrinsic,
    sigma_combined, their error bars, and the residuals (delta_mu,
    delta_sigma, delta_sigma_combined) — i.e. everything needed to
    reproduce the four panels in Figure 4 (mu_x, delta_mu_x, sigma_x,
    delta_sigma_x) for a single bin or a single global point, plus the
    diagnostic total-variance-law quantity (sigma_combined) described in
    the module docstring.

    delta_sigma (sigma_intrinsic vs sigma_true) is the primary residual per
    the reference doc; delta_sigma_combined is kept as a secondary
    diagnostic, not a replacement.
    """
    true_stats = true_value_stats(true_values)
    pred_stats = predicted_value_stats(pred_draws)

    se_boot_sigma_true = bootstrap_true_std(true_values, n_boot=n_boot, rng=rng)
    se_boot_mu_pred, se_boot_sigma_intrinsic, se_boot_sigma_combined = bootstrap_predicted(
        pred_draws, n_boot=n_boot, rng=rng
    )

    delta_mu = pred_stats["mu_pred"] - true_stats["mu_true"]
    delta_sigma = pred_stats["sigma_intrinsic"] - true_stats["sigma_true"]
    delta_sigma_combined = pred_stats["sigma_combined"] - true_stats["sigma_true"]

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
        "sigma_combined": pred_stats["sigma_combined"],
        "se_boot_sigma_combined": se_boot_sigma_combined,
        "delta_mu": delta_mu,
        "delta_sigma": delta_sigma,
        "delta_sigma_combined": delta_sigma_combined,
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
    sigma_combined, se_boot_sigma_combined, delta_mu, delta_sigma,
    delta_sigma_combined.
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
            "sigma_intrinsic", "se_boot_sigma_intrinsic",
            "sigma_combined", "se_boot_sigma_combined",
            "delta_mu", "delta_sigma", "delta_sigma_combined"]
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


# ============================================================
# Z-score / coverage calibration (per-point, no population
# variance decomposition needed — matches Legin et al. 2023
# ICdiffusion/results.py)
# ============================================================

def compute_zscores(true_values, pred_draws, ddof=1, sigma_floor=1e-6):
    """
    Per-point z-score: z_i = (x_i^true - mu_i) / sigma_i, where mu_i and
    sigma_i are THAT point's own posterior mean/std across its M draws.
    No aggregation across points is needed to compute this — unlike
    sigma_intrinsic/sigma_combined, there's no finite-M correction and no
    dependence on how points relate to each other.

    true_values : ndarray, shape (N, *extra)
    pred_draws  : ndarray, shape (N, M, *extra)
    sigma_floor : minimum allowed per-point posterior std before dividing.
                  With small M (e.g. 20 draws), a handful of points can have
                  a near-zero estimated sigma_i just from sampling noise,
                  which makes z_i blow up and dominates any pooled std(z)
                  computed downstream (std is not robust to outliers). This
                  floor bounds that contribution. Pick it relative to your
                  field's scale (e.g. ~1% of a typical sigma) rather than
                  leaving it at machine epsilon.

    Returns z, shape (N, *extra). A well-calibrated model gives z that is
    ~N(0, 1) when pooled across many points: mean(z) ~ 0, std(z) ~ 1.
    std(z) < 1 -> underdispersed (posteriors too narrow).
    std(z) > 1 -> overdispersed (posteriors too wide).

    Because std(z) itself is sensitive to the few points where sigma_i is
    smallest, always inspect zscore_summary's robust fields (median_z,
    mad_std_z) alongside the naive mean/std before trusting a single number.
    """
    true_values = np.asarray(true_values)
    pred_draws = np.asarray(pred_draws)

    mu = pred_draws.mean(axis=1)
    sigma = pred_draws.std(axis=1, ddof=ddof)
    sigma = np.clip(sigma, sigma_floor, None)

    return (true_values - mu) / sigma


def zscore_summary(z, extreme_thresh=10.0):
    """
    Pooled summary stats for a z-score array (any shape — flatten first).

    mean_z, std_z   : naive moments. std_z ~ 1 under calibration, but is NOT
                       robust — a handful of points with near-zero posterior
                       sigma can inflate std_z arbitrarily (division by a
                       small number). Always check the robust fields too.
    median_z        : robust center, should be ~0 under calibration.
    mad_std_z       : median absolute deviation scaled by 1.4826 (the
                       constant that makes it a consistent estimator of std
                       under normality). Not pulled around by outliers the
                       way std_z is — this is the number to trust first.
    std_z_trimmed   : std after clipping z to [-10, 10] (winsorizing).
                       Bounds the outlier contribution without discarding
                       points.
    frac_extreme    : fraction of points with |z| > extreme_thresh. If this
                       is more than a small fraction of a percent, std_z is
                       likely dominated by a handful of near-degenerate
                       posteriors rather than reflecting bulk calibration.
    """
    z = np.asarray(z).ravel()
    med = float(np.median(z))
    return {
        "mean_z": float(np.mean(z)),
        "std_z": float(np.std(z, ddof=1)),
        "median_z": med,
        "mad_std_z": float(1.4826 * np.median(np.abs(z - med))),
        "std_z_trimmed": float(np.std(np.clip(z, -10, 10), ddof=1)),
        "frac_extreme": float(np.mean(np.abs(z) > extreme_thresh)),
        "n": int(z.size),
    }


def plot_zscore_histogram(z, save_path=None, title=None, xlim=7):
    """
    Reproduces the Legin et al. calibration plot: histogram of pooled
    z-scores vs. a standard normal overlay. The legend reports both the
    naive std and the robust mad_std, since they can disagree substantially
    when a small fraction of points have near-zero posterior sigma.

    xlim : histogram/plot range. Values outside +/-xlim are still counted
    in the reported stats (via zscore_summary on the full z) but not shown,
    since a few extreme outliers can otherwise flatten the visible bulk.
    """
    import matplotlib.pyplot as plt
    import scipy.stats as stats

    z = np.asarray(z).ravel()
    s = zscore_summary(z)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(np.clip(z, -xlim, xlim), bins=100, density=True, color='#82A8D1',
            label=(f"Inferred (std={s['std_z']:.2f}, "
                    f"robust std={s['mad_std_z']:.2f}, "
                    f"{s['frac_extreme']*100:.2f}% |z|>10)"))
    x = np.linspace(-xlim, xlim, 500)
    ax.plot(x, stats.norm.pdf(x, 0., 1.0), linestyle='--', color='k', lw=1, label='N(0,1)')
    ax.set_xlabel(r'$z = (x_{\rm true} - \mu_{\rm pred}) / \sigma_{\rm pred}$')
    ax.set_ylabel('Probability density')
    ax.set_xlim([-xlim, xlim])
    ax.legend(fontsize=8)
    if title:
        ax.set_title(title, fontsize=11)
    ax.grid(alpha=0.2)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
    return fig


def zscore_by_bin(y_values, true_values, pred_draws, n_bins=40,
                   binning="quantile", min_points=10, ddof=1, sigma_floor=1e-6):
    """
    Bins points by y (same convention as bin_and_compute) and reports both
    std(z) and the robust mad_std(z) per bin — the Figure-4-style version
    of the coverage check. std_z ~ 1 (and mad_std_z ~ 1) across all bins
    means calibration holds uniformly in y; departures show exactly where
    (in density) the model is mis-calibrated. Comparing std_z to mad_std_z
    per bin also flags which bins, if any, are outlier-dominated.

    y_values    : ndarray, shape (V,)
    true_values : ndarray, shape (V,)
    pred_draws  : ndarray, shape (V, M)

    Returns dict with bin_center, n_points, mean_z, std_z, median_z,
    mad_std_z, frac_extreme (one entry per surviving bin).
    """
    y_values = np.asarray(y_values)
    true_values = np.asarray(true_values)
    pred_draws = np.asarray(pred_draws)

    z_all = compute_zscores(true_values, pred_draws, ddof=ddof, sigma_floor=sigma_floor)  # (V,)

    if binning == "quantile":
        edges = np.quantile(y_values, np.linspace(0, 1, n_bins + 1))
        edges = np.unique(edges)
    elif binning == "linear":
        edges = np.linspace(y_values.min(), y_values.max(), n_bins + 1)
    else:
        raise ValueError("binning must be 'quantile' or 'linear'")

    bin_idx = np.digitize(y_values, edges[1:-1], right=False)

    results = {"bin_center": [], "n_points": [], "mean_z": [], "std_z": [],
               "median_z": [], "mad_std_z": [], "frac_extreme": []}
    n_bins_actual = len(edges) - 1
    for b in range(n_bins_actual):
        mask = bin_idx == b
        n_pts = int(mask.sum())
        if n_pts < min_points:
            continue
        zb = z_all[mask]
        s = zscore_summary(zb)
        results["bin_center"].append(0.5 * (edges[b] + edges[b + 1]))
        results["n_points"].append(n_pts)
        results["mean_z"].append(s["mean_z"])
        results["std_z"].append(s["std_z"])
        results["median_z"].append(s["median_z"])
        results["mad_std_z"].append(s["mad_std_z"])
        results["frac_extreme"].append(s["frac_extreme"])

    return {k: np.array(v) for k, v in results.items()}


def plot_zscore_by_bin(results, xlabel="Present-day overdensity y", save_path=None):
    """
    Plots std(z) and the robust mad_std(z) (top) and mean(z)/median(z)
    (bottom) vs. bin center, with 1.0 / 0.0 reference lines. This is the
    coverage analogue of the sigma_x / delta_mu panels in
    plot_binned_comparison, but requires no sigma_true at all. Naive and
    robust curves are both shown so a discrepancy between them (a sign of
    outlier-dominated bins) is visible directly.
    """
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 1, figsize=(8, 6), sharex=True,
                              gridspec_kw={"height_ratios": [2, 1]})
    x = results["bin_center"]

    ax = axes[0]
    ax.plot(x, results["std_z"], 'o', color='#82A8D1', markersize=5, alpha=0.6, label='std(z)')
    ax.plot(x, results["mad_std_z"], 's', color='tab:green', markersize=5, label='robust std(z) (MAD)')
    ax.axhline(1.0, color='k', linestyle='--', linewidth=1, label='Calibrated (=1)')
    ax.set_ylabel(r'std($z$)')
    ax.legend(fontsize=8)
    ax.grid(alpha=0.2)

    ax = axes[1]
    ax.plot(x, results["mean_z"], 'o', color='#82A8D1', markersize=5, alpha=0.6, label='mean(z)')
    ax.plot(x, results["median_z"], 's', color='tab:green', markersize=5, label='median(z)')
    ax.axhline(0.0, color='k', linestyle='--', linewidth=1)
    ax.set_ylabel(r'$z$ center')
    ax.set_xlabel(xlabel)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.2)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
    return fig


def plot_binned_comparison(results, xlabel="Present-day overdensity y", save_path=None):
    """
    Four-panel plot matching Figure 4: mean (top), delta-mean, stdev,
    delta-stdev — truth in black diamonds, inferred (sigma_intrinsic, the
    quantity the reference doc specifies) in blue circles, with the
    diagnostic-only sigma_combined overlaid in green squares for reference.
    The delta panel plots delta_sigma (sigma_intrinsic vs truth), matching
    the reference doc; delta_sigma_combined is available in `results` if
    you want to plot it separately as a diagnostic.
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
    ax.errorbar(x, results["sigma_combined"], yerr=results["se_boot_sigma_combined"], fmt="s",
                color="tab:green", markersize=5, capsize=2, alpha=0.4, label="Inferred (diagnostic, mean+width)")
    ax.errorbar(x, results["sigma_true"], yerr=results["se_boot_sigma_true"], fmt="D",
                color="black", markersize=4, capsize=2, label="Truth")
    ax.set_ylabel(r"Standard deviation $\sigma_x$")
    ax.legend(fontsize=9)
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
