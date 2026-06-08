import argparse
import torch
import numpy as np
from score_models import ScoreModel, NCSNpp
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import Pk_library as PKL

# ==================== ARGUMENT PARSING ====================

parser = argparse.ArgumentParser()
parser.add_argument('--checkpoint_run', type=int, default=48,
                    help='Run number where checkpoint is located') 
parser.add_argument('--checkpoint_file', type=str, default='checkpoint_8.3722e+04_020.pt', # gas
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

print("\nCreating model architecture...")
net = NCSNpp(
    channels=C, 
    nf=32, 
    ch_mult=[2, 2, 2, 1], 
    dimensions=3,
    dropout=0.1, 
    condition=('input',), 
    condition_input_channels=1
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
#checkpoint_path = f'/work/hdd/bdne/abonab/run_{checkpoint_run}/{checkpoint_file}'
#print(f"\nLoading weights from: {checkpoint_path}")

checkpoint = torch.load(checkpoint_path)

# ==================== CREATE OUTPUT DIR ====================

save_dir = f'/work/hdd/bdne/abonab/run_{output_run}'
os.makedirs(save_dir, exist_ok=True)

# Inspect checkpoint structure
print(f"\nCheckpoint type: {type(checkpoint)}")
if isinstance(checkpoint, dict):
    print(f"Checkpoint keys: {list(checkpoint.keys())}")
    
    # Try different possible keys
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
        # Checkpoint might be the state dict itself
        print("Checkpoint appears to be a state dict directly")
        model.model.load_state_dict(checkpoint)
else:
    # Not a dict, probably IS the state dict
    print("Checkpoint is not a dict, treating as state dict")
    model.model.load_state_dict(checkpoint)

model.model.eval()

print("✓ Model loaded successfully!")

# ==================== LOAD DATA ====================

print("\nLoading data...")
#fd_all = np.load('/work/hdd/bdne/abonab/fd_cdmgas_1P_64_norm.npy') # 1P!
#ic_all = np.load('/work/hdd/bdne/abonab/ic_1P_30_64_norm.npy') # 1P!
fd_all = np.load('/work/hdd/bdne/abonab/fd_gas_1P_30_64_norm.npy')
ic_all = np.load('/work/hdd/bdne/abonab/ic_1P_30_64_norm.npy')

# fd gas set: # fd_gas_CV_norm.npy, fd_gas_1P_30_64_norm, 
# fd cdm set: # fd_cdm_CV_norm.npy, fd_cdm_1P_30_64_norm, 
# cv set is : fd_cdmgas_CV_norm.npy, ic_combined_CV_norm_reordered.npy
# 1P set is : fd_cdmgas_1P_64_norm.npy, ic_1P_30_64_norm.npy
# ==================== PROCESS MULTIPLE SAMPLES ====================

# Storage for aggregate statistics
all_cross_power_norms = []
all_pk_ratios = []
all_variance_recoveries = []
all_mean_cross_corrs = []
all_median_cross_corrs = []
all_max_cross_corrs = []

print("\n" + "="*60)
print(f"PROCESSING {len(sample_indices)} SAMPLES")
print("="*60)

for idx, sample_idx in enumerate(sample_indices):
    print(f"\n[{idx+1}/{len(sample_indices)}] Processing sample {sample_idx}...")
    
    fd_example = fd_all[sample_idx]  # (64, 64, 64)
    ic_example = ic_all[sample_idx]  # (64, 64, 64)
    
    # Add batch and channel dimensions
    fd_tensor = torch.tensor(fd_example).unsqueeze(0).unsqueeze(0).float().to('cuda')
    truth = ic_example
    draw_cross_power_norms = []
    draw_pk_ratios = []
    draw_variance_recoveries = []

    for draw in range(n_draws):
        print(f"  Draw {draw+1}/{n_draws}...", end='\r')
        with torch.no_grad():
            samples = model.sample(condition=[fd_tensor], shape=[B, C, *dimensions], steps=steps)
        prediction = samples[0, 0].cpu().numpy()
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
    
        # Generate prediction
        #print(f"  Generating prediction ({steps} steps)...")
        #with torch.no_grad():
        #    samples = model.sample(condition=[fd_tensor], shape=[B, C, *dimensions], steps=steps)
        
        #prediction = samples[0, 0].cpu().numpy()
        #truth = ic_example
        
        # Compute power spectrum
        #print(f"  Computing power spectrum...")
        #delta_pred = (prediction - np.mean(prediction)).astype(np.float32)
        #delta_true = (truth - np.mean(truth)).astype(np.float32)
        
        #Pk = PKL.XPk([delta_pred, delta_true], BoxSize=box_size, axis=0, MAS=['CIC','CIC'], threads=1)
        #cross_power_norm = Pk.XPk[:,0,0] / np.sqrt(Pk.Pk[:,0,0] * Pk.Pk[:,0,1])
    
    k_report = np.logspace(np.log10(Pk.k3D.min()), np.log10(Pk.k3D.max()), 35)

    # Get unique indices only
    indices = [np.argmin(np.abs(Pk.k3D - k_target)) for k_target in k_report]
    unique_indices = list(dict.fromkeys(indices))  # removes duplicates, preserves order

    k_values_at_report = [Pk.k3D[i] for i in unique_indices]
    r_values_at_report = [cross_power_norm[i] for i in unique_indices]

    with open(f'{save_dir}/r_at_kpoints_sample_{sample_idx}.txt', 'w') as f:
        f.write(f"Sample {sample_idx}\n")
        f.write(f"{'k [h/Mpc]':<15} {'Scale [Mpc]':<15} {'r(k)':<10}\n")
        f.write("-"*40 + "\n")
        for k, r in zip(k_values_at_report, r_values_at_report):
            scale = 2*np.pi/k
            f.write(f"{k:<15.4f} {scale:<15.2f} {r:<10.4f}\n")

    # Store statistics
    all_cross_power_norms.append(draw_cross_power_norms)  # full (n_draws, n_k_bins)
    all_pk_ratios.append(np.mean(draw_pk_ratios))
    all_variance_recoveries.append(np.mean(draw_variance_recoveries))
    all_mean_cross_corrs.append(np.mean(mean_r_k))
    all_median_cross_corrs.append(np.median(mean_r_k))
    all_max_cross_corrs.append(np.max(mean_r_k))
    np.save(f'{save_dir}/checkpoint_draws_sample_{sample_idx}.npy', draw_cross_power_norms)
    
    # Save individual visualization with FD (1x3 grid)
    f, axs = plt.subplots(1, 3, figsize=(18, 5))
    
    # Prediction IC
    #axs[0].imshow(np.mean(prediction[:5], axis=0), cmap='viridis') mean of first 5 slices
    axs[0].imshow(np.mean(prediction, axis=0), cmap='viridis')
    axs[0].set_title(f'Prediction ({steps} steps)')
    axs[0].axis('off')

    
    # True IC
    #axs[1].imshow(np.mean(truth[:5], axis=0), cmap='viridis')
    axs[1].imshow(np.mean(truth, axis=0), cmap='viridis')
    axs[1].set_title('True IC', fontsize=12)
    axs[1].axis('off')
    
    # Input FD
    #axs[2].imshow(np.mean(fd_example[:5], axis=0), cmap='viridis')
    axs[2].imshow(np.mean(fd_example, axis=0), cmap='viridis')
    axs[2].set_title('Input FD (z=0)', fontsize=12)
    axs[2].axis('off')
    
    plt.tight_layout()
    plt.savefig(f'{save_dir}/prediction_comparison_sample_{sample_idx}.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # Save individual power spectrum plot with semilogx for cross-power
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].loglog(Pk.k3D, Pk.Pk[:,0,0], 'b-', label='Prediction', linewidth=2)
    axes[0].loglog(Pk.k3D, Pk.Pk[:,0,1], 'r--', label='True IC', linewidth=2)
    axes[0].set_xlabel('k [h/Mpc]', fontsize=12)
    axes[0].set_ylabel('P(k) [(Mpc/h)³]', fontsize=12)
    axes[0].set_title(f'3D Power Spectrum - Sample {sample_idx}', fontsize=14)
    axes[0].legend(fontsize=11)
    axes[0].grid(True, alpha=0.3)
    
    axes[1].semilogx(Pk.k3D, cross_power_norm, 'k-', linewidth=2)
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
    
# --- 10-point cross-power annotation plot ---
# --- Combined cross-power plot for all samples ---
fig, ax = plt.subplots(figsize=(10, 5))
for i, (cpn, sample_idx) in enumerate(zip(all_cross_power_norms, sample_indices)):
    ax.semilogx(Pk.k3D, np.mean(cpn, axis=0), linewidth=1.5, alpha=0.7, label=f'Sample {sample_idx}')

ax.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5, label='Perfect')
ax.set_xlabel('k [h/Mpc]', fontsize=12)
ax.set_ylabel('r(k)', fontsize=12)
ax.set_title(f'Cross-Power Spectrum - All Samples', fontsize=13)
ax.set_ylim([0, 1.1])
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(f'{save_dir}/cross_power_kpoints_all_samples.png', dpi=150, bbox_inches='tight')
plt.close()
# In the per-sample loop, replace the individual pk plot block:
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
variance_recovery = (prediction.var() / truth.var()) * 100  # compute before plotting
    
axes[0].loglog(Pk.k3D, Pk.Pk[:,0,0], 'b-', label='Prediction', linewidth=2)
axes[0].loglog(Pk.k3D, Pk.Pk[:,0,1], 'r--', label='True IC', linewidth=2)
axes[0].set_xlabel('k [h/Mpc]', fontsize=12)
axes[0].set_ylabel('P(k) [(Mpc/h)³]', fontsize=12)
axes[0].set_title(f'3D Power Spectrum - Sample {sample_idx}', fontsize=14)
axes[0].legend(fontsize=11)
axes[0].grid(True, alpha=0.3)
# Add variance recovery text box
axes[0].text(0.03, 0.05, f'Variance Recovery: {variance_recovery:.1f}%',
transform=axes[0].transAxes, fontsize=11, verticalalignment='bottom', 
bbox=dict(boxstyle='round,pad=0.4', facecolor='lightyellow', alpha=0.8))

axes[1].semilogx(Pk.k3D, cross_power_norm, 'k-', linewidth=2)
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
k_nyquist = Pk.k3D.max()
k_highlight_targets = np.arange(0.5, k_nyquist, 0.5)  # 0.5, 1.0, 1.5, 2.0, ...

# Snap each target to the nearest actual k bin
highlight_idx = [np.argmin(np.abs(Pk.k3D - k_t)) for k_t in k_highlight_targets]
# Deduplicate while preserving order
highlight_idx = list(dict.fromkeys(highlight_idx))

k_pts = Pk.k3D[highlight_idx]
r_pts = cross_power_norm[highlight_idx]

fig, ax = plt.subplots(figsize=(12, 5))
ax.semilogx(Pk.k3D, cross_power_norm, 'k-', linewidth=2)
ax.scatter(k_pts, r_pts, color='red', s=60, zorder=5)
for k, r in zip(k_pts, r_pts):
    ax.annotate(f'k={k:.1f}\nr={r:.2f}', xy=(k, r),
                xytext=(0, 12), textcoords='offset points',
                fontsize=7, ha='center', color='red')
ax.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5)
ax.set_xlabel('k [h/Mpc]', fontsize=12)
ax.set_ylabel('r(k)', fontsize=12)
ax.set_title(f'Cross-Power Spectrum - Sample {sample_idx}', fontsize=13)
ax.set_ylim([0, 1.1])
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(f'{save_dir}/cross_power_kpoints_sample_{sample_idx}.png', dpi=150, bbox_inches='tight')
plt.close()

# ==================== AGGREGATE STATISTICS ====================

print("\n" + "="*60)
print("COMPUTING AGGREGATE STATISTICS")
print("="*60)

# Convert to numpy arrays
# all_cross_power_norms is a list of (n_draws, n_k_bins) arrays — one per simulation
all_draws_flat = np.concatenate(all_cross_power_norms, axis=0)  # (n_sims*n_draws, n_k_bins)
# Per-simulation mean r(k)
mean_cross_power_norms_arr = np.array([np.mean(x, axis=0) for x in all_cross_power_norms])  # (n_sims, n_k_bins)

all_pk_ratios = np.array(all_pk_ratios)
all_variance_recoveries = np.array(all_variance_recoveries)
all_median_cross_corrs = np.array(all_median_cross_corrs)
all_mean_cross_corrs = np.array(all_mean_cross_corrs)
all_max_cross_corrs = np.array(all_max_cross_corrs)

mean_cross_power = np.mean(mean_cross_power_norms_arr, axis=0)
std_cross_power  = np.std(mean_cross_power_norms_arr, axis=0)

# Percentile bands from the full draw distribution
p16_model    = np.percentile(all_draws_flat, 16, axis=0)
p84_model    = np.percentile(all_draws_flat, 84, axis=0)
median_model = np.median(all_draws_flat, axis=0)
    #all_cross_power_norms = np.array(all_cross_power_norms)  # shape: (n_samples, n_k_bins)
    #all_pk_ratios = np.array(all_pk_ratios)
    #all_variance_recoveries = np.array(all_variance_recoveries)
    #all_median_cross_corrs = np.array(all_median_cross_corrs)
    #all_mean_cross_corrs = np.array(all_mean_cross_corrs)
    #all_max_cross_corrs = np.array(all_max_cross_corrs)

    # Compute mean and std across samples
    #mean_cross_power = np.mean(all_cross_power_norms, axis=0)
    #median_cross_power = np.median(all_cross_power_norms, axis=0)
    #std_cross_power = np.std(all_cross_power_norms, axis=0)

# ==================== AGGREGATE PLOTS ====================

print("\nCreating aggregate plots...")

# Plot: Mean cross-power spectrum with error bars
fig, ax = plt.subplots(1, 1, figsize=(10, 6))
ax.plot(Pk.k3D, mean_cross_power, 'k-', linewidth=2, label='Mean')
ax.fill_between(Pk.k3D, 
                mean_cross_power - std_cross_power, 
                mean_cross_power + std_cross_power, 
                alpha=0.3, color='gray', label='±1 std')
ax.set_xscale('log')
ax.set_xlabel('k [h/Mpc]', fontsize=12)
ax.set_ylabel('r(k) = P_XY / √(P_XX P_YY)', fontsize=12)
ax.set_title(f'Mean Normalized Cross-Power Spectrum ({len(sample_indices)} samples)', fontsize=14)
ax.set_ylim([0, 1.1])
ax.axhline(y=1.0, color='red', linestyle='--', alpha=0.5, label='Perfect correlation')
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(f'{save_dir}/mean_cross_power_spectrum.png', dpi=150, bbox_inches='tight')
plt.close()

# Plot: All individual cross-power spectra overlaid
fig, ax = plt.subplots(1, 1, figsize=(10, 6))
for i, sample_idx in enumerate(sample_indices):
    ax.plot(Pk.k3D, np.mean(all_cross_power_norms[i], axis=0), alpha=0.5, linewidth=1, label=f'Sample {sample_idx}')
ax.plot(Pk.k3D, mean_cross_power, 'k-', linewidth=3, label='Mean')
ax.set_xscale('log')
ax.set_xlabel('k [h/Mpc]', fontsize=12)
ax.set_ylabel('r(k)', fontsize=12)
ax.set_title(f'All Cross-Power Spectra ({len(sample_indices)} samples)', fontsize=14)
ax.set_ylim([0, 1.1])
ax.axhline(y=1.0, color='red', linestyle='--', alpha=0.3)
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(f'{save_dir}/all_cross_power_spectra.png', dpi=150, bbox_inches='tight')
plt.close()

print(f"✓ Saved: {save_dir}/mean_cross_power_spectrum.png")
print(f"✓ Saved: {save_dir}/all_cross_power_spectra.png")

# ==================== PRINT AGGREGATE METRICS ====================

print(f"Statistics For True IC:\n")
print(f"  Min:  {np.min(truth):.4f}\n")
print(f"  Max:  {np.max(truth):.4f}\n\n")
print(f"Statistics For Prediction:\n")
print(f"  Min:  {np.min(prediction):.4f}\n")
print(f"  Max:  {np.max(prediction):.4f}\n\n")

print(f"\n{'='*60}")
print(f"AGGREGATE STATISTICS ({len(sample_indices)} samples)")
print(f"{'='*60}")

print(f"\nVariance Recovery:")
print(f"  Mean: {np.mean(all_variance_recoveries):.1f}% ± {np.std(all_variance_recoveries):.1f}%")
print(f"  Min:  {np.min(all_variance_recoveries):.1f}%")
print(f"  Max:  {np.max(all_variance_recoveries):.1f}%")

print(f"\nMean P(k) Ratio (pred/true):")
print(f"  Mean: {np.mean(all_pk_ratios):.4f} ± {np.std(all_pk_ratios):.4f}")
print(f"  Min:  {np.min(all_pk_ratios):.4f}")
print(f"  Max:  {np.max(all_pk_ratios):.4f}")

print(f"\nMean Cross-Correlation r(k):")
print(f"  Mean: {np.mean(all_mean_cross_corrs):.4f} ± {np.std(all_mean_cross_corrs):.4f}")
print(f"  Min:  {np.min(all_mean_cross_corrs):.4f}")
print(f"  Max:  {np.max(all_mean_cross_corrs):.4f}")

print(f"\nMedian Cross-Correlation r(k):")
print(f"  Mean: {np.mean(all_median_cross_corrs):.4f} ± {np.std(all_median_cross_corrs):.4f}")
print(f"  Min:  {np.min(all_median_cross_corrs):.4f}")
print(f"  Max:  {np.max(all_median_cross_corrs):.4f}")

print(f"\nMax Cross-Correlation r(k):")
print(f"  Mean: {np.mean(all_max_cross_corrs):.4f} ± {np.std(all_max_cross_corrs):.4f}")
print(f"  Min:  {np.min(all_max_cross_corrs):.4f}")
print(f"  Max:  {np.max(all_max_cross_corrs):.4f}")

print(f"\n{'='*60}\n")

# ==================== SAVE AGGREGATE METRICS ====================

with open(f'{save_dir}/aggregate_metrics.txt', 'w') as f:
    f.write(f"Checkpoint: run_{checkpoint_run}/{checkpoint_file}\n")
    f.write(f"Number of samples: {len(sample_indices)}\n")
    f.write(f"Sample indices: {sample_indices}\n")
    f.write(f"Diffusion steps: {steps}\n")
    f.write(f"{'='*60}\n\n")
    
    f.write(f"AGGREGATE STATISTICS\n")
    f.write(f"{'='*60}\n\n")
    
    f.write(f"Variance Recovery:\n")
    f.write(f"  Mean: {np.mean(all_variance_recoveries):.1f}% ± {np.std(all_variance_recoveries):.1f}%\n")
    f.write(f"  Min:  {np.min(all_variance_recoveries):.1f}%\n")
    f.write(f"  Max:  {np.max(all_variance_recoveries):.1f}%\n\n")
    
    f.write(f"Mean P(k) Ratio (pred/true):\n")
    f.write(f"  Mean: {np.mean(all_pk_ratios):.4f} ± {np.std(all_pk_ratios):.4f}\n")
    f.write(f"  Min:  {np.min(all_pk_ratios):.4f}\n")
    f.write(f"  Max:  {np.max(all_pk_ratios):.4f}\n\n")
    
    f.write(f"Mean Cross-Correlation r(k):\n")
    f.write(f"  Mean: {np.mean(all_mean_cross_corrs):.4f} ± {np.std(all_mean_cross_corrs):.4f}\n")
    f.write(f"  Min:  {np.min(all_mean_cross_corrs):.4f}\n")
    f.write(f"  Max:  {np.max(all_mean_cross_corrs):.4f}\n\n")

    f.write(f"Median Cross-Correlation r(k):\n")
    f.write(f"  Mean: {np.mean(all_median_cross_corrs):.4f} ± {np.std(all_median_cross_corrs):.4f}\n")
    f.write(f"  Min:  {np.min(all_median_cross_corrs):.4f}\n")
    f.write(f"  Max:  {np.max(all_median_cross_corrs):.4f}\n\n")
    
    f.write(f"Max Cross-Correlation r(k):\n")
    f.write(f"  Mean: {np.mean(all_max_cross_corrs):.4f} ± {np.std(all_max_cross_corrs):.4f}\n")
    f.write(f"  Min:  {np.min(all_max_cross_corrs):.4f}\n")
    f.write(f"  Max:  {np.max(all_max_cross_corrs):.4f}\n\n")
    
    f.write(f"{'='*60}\n")
    f.write(f"INDIVIDUAL SAMPLE RESULTS\n")
    f.write(f"{'='*60}\n\n")
    
    for i, sample_idx in enumerate(sample_indices):
        f.write(f"Sample {sample_idx}:\n")
        f.write(f"  Variance Recovery: {all_variance_recoveries[i]:.1f}%\n")
        f.write(f"  Mean P(k) ratio: {all_pk_ratios[i]:.4f}\n")
        f.write(f"  Mean r(k): {all_mean_cross_corrs[i]:.4f}\n")
        f.write(f"  Max r(k): {all_max_cross_corrs[i]:.4f}\n")
        f.write(f"  Median r(k): {all_median_cross_corrs[i]:.4f}\n\n")

# ==================== PARAMETER-LABELED CROSS-POWER PLOTS ====================

param_groups = [
    {
        'name': 'omega_m',
        'label': r'$\Omega_m$',
        'title': r'Cross-Power Spectrum vs. $\Omega_m$',
        'filename': 'cross_power_by_omega_m.png',
        'map': {
            sample_indices[0]: 0.3,
            sample_indices[1]: 0.4,
            sample_indices[2]: 0.5,
            sample_indices[3]: 0.2,
            sample_indices[4]: 0.1,
        }
    },
    {
        'name': 'sigma_8',
        'label': r'$\sigma_8$',
        'title': r'Cross-Power Spectrum vs. $\sigma_8$',
        'filename': 'cross_power_by_sigma8.png',
        'map': {
            sample_indices[5]:  0.8,
            sample_indices[6]:  0.9,
            sample_indices[7]:  1.0,
            sample_indices[8]:  0.7,
            sample_indices[9]:  0.6,
        }
    },
    {
        'name': 'ASN1',
        'label': r'$A_{\rm SN1}$',
        'title': r'Cross-Power Spectrum vs. $A_{\rm SN1}$ (Galactic Wind Energy)',
        'filename': 'cross_power_by_ASN1.png',
        'map': {
            sample_indices[10]: 1.0,
            sample_indices[11]: 2.0,
            sample_indices[12]: 4.0,
            sample_indices[13]: 0.5,
            sample_indices[14]: 0.25,
        }
    },
    {
        'name': 'AAGN1',
        'label': r'$A_{\rm AGN1}$',
        'title': r'Cross-Power Spectrum vs. $A_{\rm AGN1}$ (AGN Kinetic Mode Energy)',
        'filename': 'cross_power_by_AAGN1.png',
        'map': {
            sample_indices[15]: 1.0,
            sample_indices[16]: 2.0,
            sample_indices[17]: 4.0,
            sample_indices[18]: 0.5,
            sample_indices[19]: 0.25,
        }
    },
    {
        'name': 'ASN2',
        'label': r'$A_{\rm SN2}$',
        'title': r'Cross-Power Spectrum vs. $A_{\rm SN2}$ (Galactic Wind Speed)',
        'filename': 'cross_power_by_ASN2.png',
        'map': {
            sample_indices[20]: 2.0,
            sample_indices[21]: 2.82,
            sample_indices[22]: 4.0,
            sample_indices[23]: 1.41,
            sample_indices[24]: 1.0,
        }
    },
    {
        'name': 'AAGN2',
        'label': r'$A_{\rm AGN2}$',
        'title': r'Cross-Power Spectrum vs. $A_{\rm AGN2}$ (AGN Kinetic Mode Burstiness)',
        'filename': 'cross_power_by_AAGN2.png',
        'map': {
            sample_indices[25]: 1.0,
            sample_indices[26]: 1.4,
            sample_indices[27]: 2.0,
            sample_indices[28]: 0.7,
            sample_indices[29]: 0.5,
        }
    },
]

for group in param_groups:
    # Only plot samples that were actually run
    available = [(i, s) for i, s in enumerate(sample_indices) if s in group['map']]
    if not available:
        continue

    fig, ax = plt.subplots(figsize=(10, 5))
    for i, sample_idx in available:
        param_val = group['map'][sample_idx]
        ax.semilogx(Pk.k3D, np.mean(all_cross_power_norms[i], axis=0), linewidth=1.8, alpha=0.85,
                    label=f"{group['label']} $= {param_val}$")

    ax.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5, label='Perfect')
    ax.set_xlabel('k [h/Mpc]', fontsize=12)
    ax.set_ylabel('r(k)', fontsize=12)
    ax.set_title(group['title'], fontsize=13)
    ax.set_ylim([0, 1.1])
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{save_dir}/{group['filename']}", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: {save_dir}/{group['filename']}")

print("\n✓ All parameter-labeled cross-power plots complete.")


# ==================== LINEAR BASELINE (FD x True IC) ====================

print("\nComputing linear baseline cross-power (FD x True IC)...")
all_linear_r = []

for sample_idx in sample_indices:
    fd   = fd_all[sample_idx].astype(np.float32)
    ic   = ic_all[sample_idx].astype(np.float32)
    delta_fd = (fd - np.mean(fd)).astype(np.float32)
    delta_ic = (ic - np.mean(ic)).astype(np.float32)
    Pk_lin = PKL.XPk([delta_fd, delta_ic], BoxSize=box_size, axis=0, MAS=['CIC','CIC'], threads=1)
    linear_r = Pk_lin.XPk[:,0,0] / np.sqrt(Pk_lin.Pk[:,0,0] * Pk_lin.Pk[:,0,1])
    all_linear_r.append(linear_r)

all_linear_r = np.array(all_linear_r)
mean_linear_r = np.mean(all_linear_r, axis=0)

# Overlay plot: diffusion model vs linear baseline
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(Pk.k3D, mean_cross_power, 'b-', linewidth=2, label='Diffusion Reconstruction(Mean)')
ax.fill_between(Pk.k3D, mean_cross_power - std_cross_power, mean_cross_power + std_cross_power,
                alpha=0.2, color='blue')
ax.plot(Pk.k3D, mean_linear_r, 'r--', linewidth=2, label='Linear Reconstruction: FD × IC Cross-Power (Mean)')
ax.axhline(1.0, color='gray', linestyle='--', alpha=0.5)
ax.set_xscale('log')
ax.set_xlabel('k [h/Mpc]', fontsize=12)
ax.set_ylabel('r(k)', fontsize=12)
ax.set_title('Diffusion Model vs. Linear Baseline', fontsize=14)
ax.set_ylim([0, 1.1])
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(f'{save_dir}/diffusion_vs_linear_baseline.png', dpi=150, bbox_inches='tight')
plt.close()
print(f"✓ Saved: {save_dir}/diffusion_vs_linear_baseline.png")

# Percentile band plot
p16_model   = np.percentile(all_draws_flat, 16, axis=0)
p84_model   = np.percentile(all_draws_flat, 84, axis=0)
median_model  = np.median(all_draws_flat, axis=0)
median_linear = np.median(all_linear_r, axis=0)

fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(Pk.k3D, median_model, 'b-', linewidth=2, label='Diffusion (median)')
ax.fill_between(Pk.k3D, p16_model, p84_model, alpha=0.25, color='blue',
                label=f'Diffusion 16-84th pct ({len(sample_indices)} sims x {n_draws} draws)')
ax.plot(Pk.k3D, median_linear, 'r--', linewidth=2, label='Linear (median across sims)')
# No fill_between for linear
ax.axhline(1.0, color='gray', linestyle='--', alpha=0.5)
ax.set_xscale('log')
ax.set_xlabel('k [h/Mpc]', fontsize=12)
ax.set_ylabel('r(k)', fontsize=12)
ax.set_title('Diffusion vs. Linear (median, diffusion 16-84th percentile)', fontsize=13)
ax.set_ylim([0, 1.1])
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(f'{save_dir}/diffusion_vs_linear_percentile.png', dpi=150, bbox_inches='tight')
plt.close()

import matplotlib.cm as cm

param_groups_colorbar = [
    ('omega_m', r'$\Omega_m$',    {sample_indices[0]:0.3, sample_indices[1]:0.4, sample_indices[2]:0.5, sample_indices[3]:0.2, sample_indices[4]:0.1}),
    ('sigma_8', r'$\sigma_8$',    {sample_indices[5]:0.8, sample_indices[6]:0.9, sample_indices[7]:1.0, sample_indices[8]:0.7, sample_indices[9]:0.6}),
    ('ASN1',    r'$A_{\rm SN1}$', {sample_indices[10]:1.0, sample_indices[11]:2.0, sample_indices[12]:4.0, sample_indices[13]:0.5, sample_indices[14]:0.25}),
    ('AAGN1',   r'$A_{\rm AGN1}$',{sample_indices[15]:1.0, sample_indices[16]:2.0, sample_indices[17]:4.0, sample_indices[18]:0.5, sample_indices[19]:0.25}),
    ('ASN2',    r'$A_{\rm SN2}$', {sample_indices[20]:2.0, sample_indices[21]:2.82, sample_indices[22]:4.0, sample_indices[23]:1.41, sample_indices[24]:1.0}),
    ('AAGN2',   r'$A_{\rm AGN2}$',{sample_indices[25]:1.0, sample_indices[26]:1.4, sample_indices[27]:2.0, sample_indices[28]:0.7, sample_indices[29]:0.5}),
]

for name, label, pmap in param_groups_colorbar:
    available = [(i, s) for i, s in enumerate(sample_indices) if s in pmap]
    if not available:
        continue

    param_vals = [pmap[s] for _, s in available]
    vmin, vmax = min(param_vals), max(param_vals)
    cmap = cm.viridis
    norm = plt.Normalize(vmin=vmin, vmax=vmax)

    for fig_suffix, r_arrays, title_suffix, use_mean in [
        ('model',   all_cross_power_norms, 'Diffusion', True),
        ('linear',  all_linear_r,          'Linear',    False),
    ]:
        fig, ax = plt.subplots(figsize=(10, 5))
        for i, sample_idx in available:
            color = cmap(norm(pmap[sample_idx]))
            y = np.mean(r_arrays[i], axis=0) if use_mean else r_arrays[i]
            ax.semilogx(Pk.k3D, y, linewidth=1.8, color=color)
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        plt.colorbar(sm, ax=ax, label=label)
        ax.axhline(1.0, color='gray', linestyle='--', alpha=0.5)
        ax.set_xlabel('k [h/Mpc]', fontsize=12)
        ax.set_ylabel('r(k)', fontsize=12)
        ax.set_title(f'{title_suffix}: r(k) vs. {label}', fontsize=13)
        ax.set_ylim([0, 1.1])
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f'{save_dir}/colorbar_{name}_{fig_suffix}.png', dpi=150, bbox_inches='tight')
        plt.close()
        print(f"✓ Saved: {save_dir}/colorbar_{name}_{fig_suffix}.png")

# ==================== r(k) AT k=1 h/Mpc ====================

print(f"\n{'='*60}")
print(f"r(k) AT k ~ 1 h/Mpc")
print(f"{'='*60}")

k_target = 1.0
k1_idx = np.argmin(np.abs(Pk.k3D - k_target))
k1_actual = Pk.k3D[k1_idx]

r_at_k1 = [np.mean(all_cross_power_norms[i], axis=0)[k1_idx] for i in range(len(sample_indices))]

print(f"Nearest k bin to 1.0: k = {k1_actual:.4f} h/Mpc")
print(f"r(k~1) per sample:")
for i, sample_idx in enumerate(sample_indices):
    print(f"  Sample {sample_idx}: r = {r_at_k1[i]:.4f}")
print(f"\nMean r(k~1):   {np.mean(r_at_k1):.4f}")
print(f"Std  r(k~1):   {np.std(r_at_k1):.4f}")
print(f"Min  r(k~1):   {np.min(r_at_k1):.4f}")
print(f"Max  r(k~1):   {np.max(r_at_k1):.4f}")

with open(f'{save_dir}/aggregate_metrics.txt', 'a') as f:
    f.write(f"\nr(k) AT k ~ 1 h/Mpc (nearest bin: {k1_actual:.4f})\n")
    f.write(f"{'='*60}\n")
    for i, sample_idx in enumerate(sample_indices):
        f.write(f"  Sample {sample_idx}: {r_at_k1[i]:.4f}\n")
    f.write(f"Mean: {np.mean(r_at_k1):.4f} ± {np.std(r_at_k1):.4f}\n")
    f.write(f"Min:  {np.min(r_at_k1):.4f}\n")
    f.write(f"Max:  {np.max(r_at_k1):.4f}\n")

print(f"✓ Metrics saved to {save_dir}/aggregate_metrics.txt")
print(f"\nDONE! All results in: {save_dir}/")
