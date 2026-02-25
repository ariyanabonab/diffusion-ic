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
parser.add_argument('--checkpoint_run', type=int, default=33,
                    help='Run number where checkpoint is located')
parser.add_argument('--checkpoint_file', type=str, default='checkpoint_1.3806e+05_005.pt',
                    help='Checkpoint filename')
parser.add_argument('--output_run', type=int, default=30,
                    help='Run number for saving outputs')
parser.add_argument('--sample_indices', type=int, nargs='+', default=[950, 960, 970, 980, 990],
                    help='Which validation samples to test (e.g., --sample_indices 950 951 952)')
parser.add_argument('--steps', type=int, default=250,
                    help='Number of sampling steps')
args = parser.parse_args()

print("="*60)
print("SAMPLING ONLY - MULTIPLE SAMPLES")
print("="*60)

# ==================== CONFIGURATION ====================

checkpoint_run = args.checkpoint_run
checkpoint_file = args.checkpoint_file
output_run = args.output_run
sample_indices = args.sample_indices
steps = args.steps

B = 1
C = 1
dimensions = [64, 64, 64]
box_size = 25.0  # CAMELS box size in Mpc/h

print(f"\nLoading checkpoint from run_{checkpoint_run}/{checkpoint_file}")
print(f"Saving results to run_{output_run}")
print(f"Sample indices: {sample_indices}")
print(f"Number of samples: {len(sample_indices)}")
print(f"Diffusion steps: {steps}")

# ==================== RECREATE MODEL ====================

print("\nCreating model architecture...")
net = NCSNpp(
    channels=C, 
    nf=32, 
    ch_mult=[2, 2, 1, 1], 
    dimensions=3,
    dropout=0.1, 
    condition=('input',), 
    condition_input_channels=1
).to('cuda')

model = ScoreModel(model=net, sigma_min=0.01, sigma_max=5.0, device="cuda")

# ==================== LOAD CHECKPOINT ====================

checkpoint_path = f'/work/hdd/bdne/abonab/run_{checkpoint_run}/{checkpoint_file}'
print(f"\nLoading weights from: {checkpoint_path}")

checkpoint = torch.load(checkpoint_path)

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

# ==================== CREATE OUTPUT DIR ====================

save_dir = f'/work/hdd/bdne/abonab/run_{output_run}'
os.makedirs(save_dir, exist_ok=True)

# ==================== LOAD DATA ====================

print("\nLoading data...")
fd_all = np.load('/work/hdd/bdne/abonab/downsampled_fd_64_camels_normalized_mod.npy')
ic_all = np.load('/work/hdd/bdne/abonab/downsampled_ic_64_camels_normalized_mod.npy')

# ==================== PROCESS MULTIPLE SAMPLES ====================

# Storage for aggregate statistics
all_cross_power_norms = []
all_pk_ratios = []
all_variance_recoveries = []
all_mean_cross_corrs = []
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
    
    # Generate prediction
    print(f"  Generating prediction ({steps} steps)...")
    with torch.no_grad():
        samples = model.sample(condition=[fd_tensor], shape=[B, C, *dimensions], steps=steps)
    
    prediction = samples[0, 0].cpu().numpy()
    truth = ic_example
    
    # Compute power spectrum
    print(f"  Computing power spectrum...")
    delta_pred = (prediction - np.mean(prediction)).astype(np.float32)
    delta_true = (truth - np.mean(truth)).astype(np.float32)
    
    Pk = PKL.XPk([delta_pred, delta_true], BoxSize=box_size, axis=0, MAS=['CIC','CIC'], threads=1)
    cross_power_norm = Pk.XPk[:,0,0] / np.sqrt(Pk.Pk[:,0,0] * Pk.Pk[:,0,1])
    
    # Store statistics
    all_cross_power_norms.append(cross_power_norm)
    all_pk_ratios.append(np.mean(Pk.Pk[:,0,0] / Pk.Pk[:,0,1]))
    all_variance_recoveries.append((prediction.var() / truth.var()) * 100)
    all_mean_cross_corrs.append(np.mean(cross_power_norm))
    all_max_cross_corrs.append(np.max(cross_power_norm))
    
    # Save individual visualization with FD (1x3 grid)
    f, axs = plt.subplots(1, 3, figsize=(18, 5))
    
    # Prediction IC
    axs[0].imshow(np.mean(prediction[:5], axis=0), cmap='viridis')
    axs[0].set_title(f'Prediction ({steps} steps)')
    axs[0].axis('off')
    
    # True IC
    axs[1].imshow(np.mean(truth[:5], axis=0), cmap='viridis')
    axs[1].set_title('True IC', fontsize=12)
    axs[1].axis('off')
    
    # Input FD
    axs[2].imshow(np.mean(fd_example[:5], axis=0), cmap='viridis')
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

# ==================== AGGREGATE STATISTICS ====================

print("\n" + "="*60)
print("COMPUTING AGGREGATE STATISTICS")
print("="*60)

# Convert to numpy arrays
all_cross_power_norms = np.array(all_cross_power_norms)  # shape: (n_samples, n_k_bins)
all_pk_ratios = np.array(all_pk_ratios)
all_variance_recoveries = np.array(all_variance_recoveries)
all_mean_cross_corrs = np.array(all_mean_cross_corrs)
all_max_cross_corrs = np.array(all_max_cross_corrs)

# Compute mean and std across samples
mean_cross_power = np.mean(all_cross_power_norms, axis=0)
std_cross_power = np.std(all_cross_power_norms, axis=0)

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
    ax.plot(Pk.k3D, all_cross_power_norms[i], alpha=0.5, linewidth=1, label=f'Sample {sample_idx}')
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
        f.write(f"  Max r(k): {all_max_cross_corrs[i]:.4f}\n\n")

print(f"✓ Metrics saved to {save_dir}/aggregate_metrics.txt")
print(f"\nDONE! All results in: {save_dir}/")
# # Quick test
# # Quick augmentation test
# #   print("\nTesting augmentation...")
# #    test_dataset = Dataset(start_idx=0, length=10)
# #    sample1 = test_dataset[0]
# #    sample2 = test_dataset[0]
# #    print("Are samples identical?", torch.allclose(sample1[0], sample2[0]))
# #    print("This should be False if augmentation is working")

# # Extract prediction
# prediction = samples[0, 0].cpu().numpy()  # (64, 64, 64)
# truth = ic_example  # (64, 64, 64)

# # ADD THESE DIAGNOSTIC LINES HERE:
# print(f"\n{'='*60}")
# print(f"STATISTICS CHECK - Sample {sample_idx}")
# print(f"{'='*60}")
# print(f"Truth IC    - mean: {truth.mean():.6f}, std: {truth.std():.6f}, var: {truth.var():.6f}")
# print(f"Prediction  - mean: {prediction.mean():.6f}, std: {prediction.std():.6f}, var: {prediction.var():.6f}")
# print(f"Input FD    - mean: {fd_example.mean():.6f}, std: {fd_example.std():.6f}, var: {fd_example.var():.6f}")
# print(f"Std ratio (pred/truth): {prediction.std() / truth.std():.3f}")
# print(f"Var ratio (pred/truth): {prediction.var() / truth.var():.3f}")
# print(f"{'='*60}\n")

# # ==================== BASIC VISUALIZATION ====================