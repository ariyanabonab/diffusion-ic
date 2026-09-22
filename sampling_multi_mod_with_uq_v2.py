import argparse
import torch
import numpy as np
from score_models import ScoreModel, NCSNpp
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import Pk_library as PKL

# NEW: intrinsic variance decomposition (see intrinsic_variance_combined.py, same folder)
from intrinsic_variance_combinedv2 import (
    compare_true_vs_predicted, bin_and_compute, plot_binned_comparison,
    compute_zscores, zscore_summary, plot_zscore_histogram,
    zscore_by_bin, plot_zscore_by_bin
)

# ==================== ARGUMENT PARSING ====================

parser = argparse.ArgumentParser()
parser.add_argument('--checkpoint_run', type=int, default=88,
                    help='Run number where checkpoint is located') 
parser.add_argument('--checkpoint_file', type=str, default='checkpoint_8.2732e+04_050.pt', 
                    help='Checkpoint filename')
parser.add_argument('--checkpoint_number', type=int, default=None,
                    help='Checkpoint epoch number for glob-based loading (e.g., 10 matches checkpoint_*_010.pt)')
parser.add_argument('--load_checkpoint', action='store_true', default=False,
                    help='Whether to load from a checkpoint')
parser.add_argument('--output_run', type=int, default=None,
                    help='prints an output number run') # output is usually 200+
parser.add_argument('--sample_indices', type=int, nargs='+', default=[950, 960, 970, 980, 990],
                    help='Which validation samples to test (e.g., --sample_indices 950 951 952)')
parser.add_argument('--steps', type=int, default=500,
                    help='Number of sampling steps')
parser.add_argument('--n_draws', type=int, default=20,
                    help='Number of stochastic draws per simulation for error estimation')
args = parser.parse_args()

print("="*60)
print("SAMPLING ONLY - MULTIPLE SAMPLES")
print("="*60)

# ==================== CONFIGURATION ====================

checkpoint_run = args.checkpoint_run
checkpoint_file = args.checkpoint_file
checkpoint_number = args.checkpoint_number
output_run = args.output_run
sample_indices = args.sample_indices
steps = args.steps
n_draws = args.n_draws

B = 1
C = 1
dimensions = [64, 64, 64]
box_size = 25.0  # CAMELS box size in Mpc/h

print(f"\nLoading checkpoint from run_{checkpoint_run}/{checkpoint_file}")
print(f"Sample indices: {sample_indices}")
print(f"Number of samples: {len(sample_indices)}")
print(f"Diffusion steps: {steps}")

# ==================== RECREATE MODEL ====================

net=NCSNpp(
    channels=C,
    nf=64, #64,
    ch_mult=[2, 2, 2, 2],# [2,2,2,2]
    sigma_min=0.01,
    sigma_max=400, #1000,
    dropout=0.0,
    fir=False,
    attention=True, #False,
    dimensions=3,
    padding_mode="circular",
    condition=('input',), 
    condition_input_channels=1,
).to('cuda')

model = ScoreModel(model=net, sigma_min=0.01, sigma_max=400, device="cuda")

# ==================== LOAD CHECKPOINT ====================

if args.checkpoint_number is not None:
    import glob
    pattern = f'/work/hdd/bdne/abonab/run_{checkpoint_run}/checkpoint_*_{args.checkpoint_number:03d}.pt'
    matches = glob.glob(pattern)
    if not matches:
        raise FileNotFoundError(f"No checkpoint found matching: {pattern}")
    checkpoint_path = matches[0]
else:
    checkpoint_path = f'/work/hdd/bdne/abonab/run_{checkpoint_run}/{checkpoint_file}'

checkpoint = torch.load(checkpoint_path)

# ==================== CREATE OUTPUT DIR ====================

save_dir = f'/work/hdd/bdne/abonab/run_{output_run}'
os.makedirs(save_dir, exist_ok=True)

# Inspect checkpoint structure
print(f"\nCheckpoint type: {type(checkpoint)}")
if isinstance(checkpoint, dict):
    print(f"Checkpoint keys: {list(checkpoint.keys())}")
    
    if 'model_state_dict' in checkpoint:
        print("Loading from 'model_state_dict'")
        model.model.load_state_dict(checkpoint['model_state_dict'])
    elif 'state_dict' in checkpoint:
        print("Loading from 'state_dict'")
        model.model.load_state_dict(checkpoint['state_dict'])
    elif 'model' in checkpoint:
        print("Loading from 'model'")
        model.model.load_state_dict(checkpoint['model'])
    else:
        print("Checkpoint appears to be a state dict directly")
        model.model.load_state_dict(checkpoint)
else:
    print("Checkpoint is not a dict, treating as state dict")
    model.model.load_state_dict(checkpoint)

model.model.eval()

print("✓ Model loaded successfully!")

# ==================== LOAD DATA ====================

print("\nLoading data...")
fd_all = np.load('/work/hdd/bdne/abonab/fd_64_mcdm_norm.npy')
ic_all = np.load('/work/hdd/bdne/abonab/ic_all_norm.npy')

# ==================== PROCESS MULTIPLE SAMPLES ====================

# Storage for aggregate statistics
all_cross_power_norms = []
all_pk_ratios = []
all_variance_recoveries = []
all_mean_cross_corrs = []
all_median_cross_corrs = []
all_max_cross_corrs = []

# NEW: storage for the intrinsic-variance UQ decomposition
all_draw_predictions = []   # will hold one (n_draws, 64,64,64) array per sample
all_truth_fields = []       # will hold one (64,64,64) truth array per sample
all_fd_fields = []          # will hold one (64,64,64) FD array per sample — the binning variable y

print("\n" + "="*60)
print(f"PROCESSING {len(sample_indices)} SAMPLES")
print("="*60)

for idx, sample_idx in enumerate(sample_indices):
    print(f"\n[{idx+1}/{len(sample_indices)}] Processing sample {sample_idx}...")
    
    fd_example = fd_all[sample_idx]  # (64, 64, 64)
    ic_example = ic_all[sample_idx]  # (64, 64, 64)
    
    fd_tensor = torch.tensor(fd_example).unsqueeze(0).unsqueeze(0).float().to('cuda')
    truth = ic_example
    draw_cross_power_norms = []
    draw_pk_ratios = []
    draw_variance_recoveries = []
    draw_predictions = []   # NEW: full 3D field for every draw of this sample

    for draw in range(n_draws):
        print(f"  Draw {draw+1}/{n_draws}...", end='\r')
        with torch.no_grad():
            samples = model.sample(condition=[fd_tensor], shape=[B, C, *dimensions], steps=steps)
        prediction = samples[0, 0].cpu().numpy()
        draw_predictions.append(prediction.copy())   # NEW: keep the raw field, not just its power spectrum

        delta_pred = (prediction - np.mean(prediction)).astype(np.float32)
        delta_true = (truth - np.mean(truth)).astype(np.float32)
        Pk = PKL.XPk([delta_pred, delta_true], BoxSize=box_size, axis=0, MAS=['CIC','CIC'], threads=1)
        cross_power_norm = Pk.XPk[:,0,0] / np.sqrt(Pk.Pk[:,0,0] * Pk.Pk[:,0,1])
        draw_cross_power_norms.append(cross_power_norm)
        draw_pk_ratios.append(np.mean(Pk.Pk[:,0,0] / Pk.Pk[:,0,1]))
        draw_variance_recoveries.append((prediction.var() / truth.var()) * 100)

    print(f"  All {n_draws} draws complete.        ")
    draw_cross_power_norms = np.array(draw_cross_power_norms)  # (n_draws, n_k_bins)
    mean_r_k = np.mean(draw_cross_power_norms, axis=0)
    std_r_k  = np.std(draw_cross_power_norms, axis=0)
    cross_power_norm = mean_r_k 

    # NEW: stack this sample's draws into one array and save + accumulate for UQ
    draw_predictions = np.stack(draw_predictions, axis=0)  # (n_draws, 64,64,64)
    np.save(f'{save_dir}/draw_fields_sample_{sample_idx}.npy', draw_predictions)
    all_draw_predictions.append(draw_predictions)
    all_truth_fields.append(truth)
    all_fd_fields.append(fd_example.copy())   # NEW: keep the FD field for binning

    k_report = np.logspace(np.log10(Pk.k3D.min()), np.log10(Pk.k3D.max()), 35)
    indices = [np.argmin(np.abs(Pk.k3D - k_target)) for k_target in k_report]
    unique_indices = list(dict.fromkeys(indices))

    k_values_at_report = [Pk.k3D[i] for i in unique_indices]
    r_values_at_report = [cross_power_norm[i] for i in unique_indices]

    with open(f'{save_dir}/r_at_kpoints_sample_{sample_idx}.txt', 'w') as f:
        f.write(f"Sample {sample_idx}\n")
        f.write(f"{'k [h/Mpc]':<15} {'Scale [Mpc]':<15} {'r(k)':<10}\n")
        f.write("-"*40 + "\n")
        for k, r in zip(k_values_at_report, r_values_at_report):
            scale = 2*np.pi/k
            f.write(f"{k:<15.4f} {scale:<15.2f} {r:<10.4f}\n")

    all_cross_power_norms.append(draw_cross_power_norms)
    all_pk_ratios.append(np.mean(draw_pk_ratios))
    all_variance_recoveries.append(np.mean(draw_variance_recoveries))
    all_mean_cross_corrs.append(np.mean(mean_r_k))
    all_median_cross_corrs.append(np.median(mean_r_k))
    all_max_cross_corrs.append(np.max(mean_r_k))
    np.save(f'{save_dir}/checkpoint_draws_sample_{sample_idx}.npy', draw_cross_power_norms)

    f, axs = plt.subplots(1, 3, figsize=(18, 5))
    axs[0].imshow(np.mean(prediction, axis=0), cmap='viridis')
    axs[0].set_title(f'Prediction ({steps} steps)')
    axs[0].axis('off')
    axs[1].imshow(np.mean(truth, axis=0), cmap='viridis')
    axs[1].set_title('True IC', fontsize=12)
    axs[1].axis('off')
    axs[2].imshow(np.mean(fd_example, axis=0), cmap='viridis')
    axs[2].set_title('Input FD (z=0)', fontsize=12)
    axs[2].axis('off')
    plt.tight_layout()
    plt.savefig(f'{save_dir}/prediction_comparison_sample_{sample_idx}.png', dpi=150, bbox_inches='tight')
    plt.close()

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].loglog(Pk.k3D, Pk.Pk[:,0,0], 'b-', label='Prediction', linewidth=2)
    axes[0].loglog(Pk.k3D, Pk.Pk[:,0,1], 'r--', label='True IC', linewidth=2)
    axes[0].set_xlabel('k [h/Mpc]', fontsize=12)
    axes[0].set_ylabel('P(k) [(Mpc/h)³]', fontsize=12)
    axes[0].set_title(f'3D Power Spectrum - Sample {sample_idx}', fontsize=14)
    axes[0].legend(fontsize=11)
    axes[0].grid(True, alpha=0.3)

    axes[1].semilogx(Pk.k3D, cross_power_norm, 'k-', linewidth=2)
    axes[1].fill_between(Pk.k3D, cross_power_norm - std_r_k, cross_power_norm + std_r_k,
                      alpha=0.3, color='gray', label=f'±1σ ({n_draws} draws)')
    axes[1].set_xlabel('k [h/Mpc]', fontsize=12)
    axes[1].set_ylabel('r(k) = P_XY / √(P_XX P_YY)', fontsize=12)
    axes[1].set_title(f'Cross-Power - Sample {sample_idx}', fontsize=14)
    axes[1].set_ylim([0, 1.1])
    axes[1].axhline(y=1.0, color='gray', linestyle='--', alpha=0.5, label='Perfect')
    axes[1].legend(fontsize=10)
    axes[1].grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{save_dir}/pk_analysis_sample_{sample_idx}.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Sample {sample_idx} complete!")

# ==================== AGGREGATE STATISTICS ====================

print("\n" + "="*60)
print("COMPUTING AGGREGATE STATISTICS")
print("="*60)

all_draws_flat = np.concatenate(all_cross_power_norms, axis=0)
mean_cross_power_norms_arr = np.array([np.mean(x, axis=0) for x in all_cross_power_norms])

all_pk_ratios = np.array(all_pk_ratios)
all_variance_recoveries = np.array(all_variance_recoveries)
all_median_cross_corrs = np.array(all_median_cross_corrs)
all_mean_cross_corrs = np.array(all_mean_cross_corrs)
all_max_cross_corrs = np.array(all_max_cross_corrs)

mean_cross_power = np.mean(mean_cross_power_norms_arr, axis=0)
std_cross_power  = np.std(mean_cross_power_norms_arr, axis=0)

within_sample_var = np.mean([np.var(x, axis=0) for x in all_cross_power_norms], axis=0)
between_sample_var = std_cross_power**2
total_std = np.sqrt(within_sample_var + between_sample_var)

p16_model    = np.percentile(all_draws_flat, 16, axis=0)
p84_model    = np.percentile(all_draws_flat, 84, axis=0)
median_model = np.median(all_draws_flat, axis=0)

print(f"\nVariance Recovery:")
print(f"  Mean: {np.mean(all_variance_recoveries):.1f}% ± {np.std(all_variance_recoveries):.1f}%")

with open(f'{save_dir}/aggregate_metrics.txt', 'w') as f:
    f.write(f"Checkpoint: run_{checkpoint_run}/{checkpoint_file}\n")
    f.write(f"Number of samples: {len(sample_indices)}\n")
    f.write(f"Sample indices: {sample_indices}\n")
    f.write(f"Diffusion steps: {steps}\n")
    f.write(f"{'='*60}\n\n")
    f.write(f"Variance Recovery:\n")
    f.write(f"  Mean: {np.mean(all_variance_recoveries):.1f}% ± {np.std(all_variance_recoveries):.1f}%\n\n")
    f.write(f"Mean Cross-Correlation r(k):\n")
    f.write(f"  Mean: {np.mean(all_mean_cross_corrs):.4f} ± {np.std(all_mean_cross_corrs):.4f}\n")

# ==================== NEW: INTRINSIC VARIANCE (UQ) ====================
# Quick whole-field sanity check: treats each of your n_sims test fields as
# one "point," pooling all its voxels together. This is NOT the voxel-binned
# version that reproduces Figure 4 (that needs binning by FD/overdensity —
# a separate step) — this just confirms the corrected formula runs cleanly
# on your existing sample set before you build the binned version.

print("\n" + "="*60)
print("INTRINSIC VARIANCE (whole-field sanity check)")
print("="*60)

all_draw_predictions_arr = np.stack(all_draw_predictions, axis=0)  # (n_sims, n_draws, 64,64,64)
truth_fields_arr = np.stack(all_truth_fields, axis=0)              # (n_sims, 64,64,64)

# n_boot kept small here since bootstrapping full (64,64,64) fields is
# memory-heavy — 10,000 replicates only makes sense once you've reduced
# down to per-bin scalars (see intrinsic_variance_combined.py's note on Figure 4).
uq_result = compare_true_vs_predicted(truth_fields_arr, all_draw_predictions_arr, n_boot=200)

# Each entry in uq_result is a per-voxel (64,64,64) map; average over
# voxels to get single reportable numbers for this sanity check.
summary = {k: float(np.mean(v)) for k, v in uq_result.items()}

print(f"mu_true:          {summary['mu_true']:.6e}")
print(f"sigma_true:       {summary['sigma_true']:.6e}")
print(f"mu_pred:          {summary['mu_pred']:.6e}")
print(f"sigma_intrinsic:  {summary['sigma_intrinsic']:.6e}")
print(f"sigma_combined:   {summary['sigma_combined']:.6e}  (diagnostic only)")
print(f"delta_mu:         {summary['delta_mu']:.6e}")
print(f"delta_sigma:      {summary['delta_sigma']:.6e}")
print(f"delta_sigma_combined: {summary['delta_sigma_combined']:.6e}  (diagnostic only)")

with open(f'{save_dir}/intrinsic_variance.txt', 'w') as f:
    f.write("Whole-field sanity check (voxel-averaged; NOT the binned Figure-4 version)\n")
    f.write("="*60 + "\n\n")
    for k, v in summary.items():
        f.write(f"{k}: {v:.6e}\n")

print(f"✓ Saved: {save_dir}/intrinsic_variance.txt")

# Quick comparison plot: true vs. predicted, mean and stdev, with error bars.
# This is a single aggregate point (all voxels/samples averaged together) —
# NOT the binned-by-overdensity version of Figure 4, just a fast visual
# sanity check of the same two numbers in the text file above.
fig, axes = plt.subplots(1, 2, figsize=(9, 4))

axes[0].errorbar([0], [summary['mu_true']], yerr=[summary['se_mu_true']],
                  fmt='ko', markersize=8, capsize=5, label='Truth')
axes[0].errorbar([1], [summary['mu_pred']], yerr=[summary['se_boot_mu_pred']],
                  fmt='o', color='tab:blue', markersize=8, capsize=5, label='Inferred')
axes[0].set_xticks([0, 1])
axes[0].set_xticklabels(['Truth', 'Inferred'])
axes[0].set_ylabel(r'Mean $\mu_x$', fontsize=12)
axes[0].set_title('Mean', fontsize=13)
axes[0].grid(True, alpha=0.3)

axes[1].errorbar([0], [summary['sigma_true']], yerr=[summary['se_boot_sigma_true']],
                  fmt='ko', markersize=8, capsize=5, label='Truth')
axes[1].errorbar([1], [summary['sigma_intrinsic']], yerr=[summary['se_boot_sigma_intrinsic']],
                  fmt='o', color='tab:blue', markersize=8, capsize=5, label='Inferred')
axes[1].errorbar([2], [summary['sigma_combined']], yerr=[summary['se_boot_sigma_combined']],
                  fmt='s', color='tab:green', markersize=8, capsize=5, alpha=0.5, label='Inferred (diagnostic)')
axes[1].set_xticks([0, 1, 2])
axes[1].set_xticklabels(['Truth', 'Inferred', 'Inferred\n(diagnostic)'])
axes[1].set_ylabel(r'Standard Deviation $\sigma_x$', fontsize=12)
axes[1].set_title('Stdev', fontsize=13)
axes[1].legend(fontsize=9)
axes[1].grid(True, alpha=0.3)

plt.suptitle(f'Whole-field UQ sanity check ({len(sample_indices)} samples, {n_draws} draws)', fontsize=13)
plt.tight_layout()
plt.savefig(f'{save_dir}/intrinsic_variance_summary.png', dpi=150, bbox_inches='tight')
plt.close()
print(f"✓ Saved: {save_dir}/intrinsic_variance_summary.png")

# ==================== NEW: BINNED UQ (Figure 4 style) ====================
# Pools every voxel from every test field, bins by its FD value (present-day
# overdensity), and applies the same true-vs-predicted comparison
# independently within each bin. This is what actually reproduces the
# multi-point spread in Figure 4 — the whole-field check above was just one
# point of this curve, averaged over everything.

print("\n" + "="*60)
print("BINNED UQ (Figure 4 style)")
print("="*60)

y_all = np.concatenate([f.flatten() for f in all_fd_fields])            # (n_sims*64^3,)
true_all = np.concatenate([f.flatten() for f in all_truth_fields])      # (n_sims*64^3,)
pred_all = np.concatenate(
    [d.reshape(d.shape[0], -1).T for d in all_draw_predictions], axis=0
)  # (n_sims*64^3, n_draws)

print(f"Total pooled voxels: {y_all.shape[0]:,}")

# n_boot=300 per bin keeps runtime reasonable across 40 bins; raise once
# you're happy with the bin edges and want tighter final error bars.
binned = bin_and_compute(y_all, true_all, pred_all, n_bins=40,
                          binning='quantile', min_points=10, n_boot=300)

np.savez(f'{save_dir}/binned_uq_results.npz', **binned)
print(f"✓ Saved: {save_dir}/binned_uq_results.npz  ({len(binned['bin_center'])} bins survived)")

plot_binned_comparison(binned, xlabel='Present-day overdensity y (FD field value)',
                        save_path=f'{save_dir}/binned_uq_figure4.png')
print(f"✓ Saved: {save_dir}/binned_uq_figure4.png")

# ==================== NEW: Z-SCORE CALIBRATION (Legin et al. style) ====================

print("\n" + "="*60)
print("Z-SCORE CALIBRATION")
print("="*60)

# Diagnostic check FIRST: how contaminated is the pooled sigma distribution
# by near-degenerate posteriors (sigma_i ~ 0 from only n_draws draws)? This
# tells you whether to trust std_z at face value or lean on the robust
# fields (median_z, mad_std_z) in zscore_summary instead.
sigma_per_voxel = pred_all.std(axis=1, ddof=1)
print(f"min per-voxel posterior sigma: {sigma_per_voxel.min():.6e}")
print(f"fraction of voxels with sigma < 1e-3: {np.mean(sigma_per_voxel < 1e-3)*100:.4f}%")
print(f"fraction of voxels with sigma < 1e-6: {np.mean(sigma_per_voxel < 1e-6)*100:.4f}%")

# Global (pooled) check — no binning
z_pooled = compute_zscores(true_all, pred_all)   # true_all: (V,), pred_all: (V, M)
z_stats = zscore_summary(z_pooled)
print(f"\nPooled z-score:")
print(f"  mean={z_stats['mean_z']:.4f}, std={z_stats['std_z']:.4f}  (naive, sensitive to outliers)")
print(f"  median={z_stats['median_z']:.4f}, robust_std(MAD)={z_stats['mad_std_z']:.4f}  (robust)")
print(f"  std_trimmed(clip to +/-10)={z_stats['std_z_trimmed']:.4f}")
print(f"  fraction |z| > 10: {z_stats['frac_extreme']*100:.4f}%")
print(f"  n={z_stats['n']:,}")
print("  std ~ 1 -> calibrated; < 1 -> underdispersed; > 1 -> overdispersed")
print("  If naive std and robust std disagree a lot, trust the robust one —")
print("  the naive std is likely dominated by a small number of near-zero-sigma voxels.")

plot_zscore_histogram(z_pooled, save_path=f'{save_dir}/zscore_histogram.png',
                       title=f'Pooled z-score calibration ({len(sample_indices)} samples, {n_draws} draws)')
print(f"✓ Saved: {save_dir}/zscore_histogram.png")

# Binned by y — the Figure-4-style breakdown (now includes robust fields too)
z_binned = zscore_by_bin(y_all, true_all, pred_all, n_bins=40,
                          binning='quantile', min_points=10)
np.savez(f'{save_dir}/zscore_binned.npz', **z_binned)
plot_zscore_by_bin(z_binned, xlabel='Present-day overdensity y (FD field value)',
                    save_path=f'{save_dir}/zscore_by_bin.png')
print(f"✓ Saved: {save_dir}/zscore_by_bin.png")

print(f"\nDONE! All results in: {save_dir}/")
