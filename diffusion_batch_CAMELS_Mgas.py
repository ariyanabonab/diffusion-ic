# !pip install jax==0.7.0
# !pip -q install torch score_models h5py tqdm
# !pip install score_models
# !pip install jaxpm

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--run_number', type=int, default=0, 
                    help='Run number for saving checkpoints')
parser.add_argument('--load_checkpoint', action='store_true',
                    help='Load from checkpoint and skip training')
parser.add_argument('--checkpoint_run', type=int, default=None,
                    help='Run number of checkpoint to load (if different from run_number)')
parser.add_argument('--checkpoint_epoch', type=int, default=5,
                    help='Which epoch checkpoint to load')
args = parser.parse_args()
run_number = args.run_number
print(f"Run number: {run_number}")

import torch
from torch.utils.data import TensorDataset
from score_models import ScoreModel, EnergyModel, MLP, NCSNpp
import shutil, os
import numpy as np
import score_models
print(score_models.__file__)
import torch
from torch.utils.data import TensorDataset
from score_models import ScoreModel, EnergyModel, MLP, NCSNpp
import shutil, os
import numpy as np
print(torch.cuda.is_available())

# Import for power spectrum and cross-correlation
import Pk_library as PKL
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

class Dataset(torch.utils.data.Dataset):
    def __init__(self, start_idx, length, ic_stats=None, fd_stats=None):
        self.start_idx = start_idx
        self.ic_stats = ic_stats
        self.fd_stats = fd_stats
        self.length = length
        self.__name__ = 'dataset'
        
        # loading then putting into memory
        print("Loading data into memory...")
        all_ics = np.load(f'/work/hdd/bdne/abonab/downsampled_ic_64_hydrogas_norm.npy')
        all_fds = np.load(f'/work/hdd/bdne/abonab/downsampled_fd_64_hydrogas_norm.npy')
        
        # slicing this to desired range
        self.ics = all_ics[start_idx:start_idx+length]
        self.fds = all_fds[start_idx:start_idx+length]
        
        # Converting to torch tensors and moving to GPU
        self.ics = torch.from_numpy(self.ics).float().cuda()
        self.fds = torch.from_numpy(self.fds).float().cuda()

    def __len__(self):
        return self.length

    def __getitem__(self, i):
    # Get tensors (already on GPU)
        initial_c = self.ics[i]  # Shape: (64, 64, 64), on GPU
        final_d = self.fds[i]    # Shape: (64, 64, 64), on GPU
    
    # Random augmentation choices
        flip_axis = np.random.choice([0, 1, 2])
        rot_axis = np.random.choice([0, 1, 2])
        rot_amount = np.random.choice([0, 1, 2, 3])  # 0, 90, 180, 270 degrees
    
        # Flip using torch (not numpy!)
        initial_c = torch.flip(initial_c, dims=[flip_axis])
        final_d = torch.flip(final_d, dims=[flip_axis])
    
        # Rotate using torch (not numpy!)
        initial_c = torch.rot90(initial_c, k=rot_amount, dims=[rot_axis, (rot_axis+1)%3])
        final_d = torch.rot90(final_d, k=rot_amount, dims=[rot_axis, (rot_axis+1)%3])
    
        # Add channel dimension
        initial_c = initial_c.unsqueeze(0)  # Shape: (1, 64, 64, 64)
        final_d = final_d.unsqueeze(0)
    
        return initial_c, final_d

# ==================== AUGMENTATION TEST ====================
print("\n" + "="*60)
print("TESTING AUGMENTATION")
print("="*60)

test_dataset = Dataset(start_idx=0, length=10)

# Get the same sample twice
sample1_ic, sample1_fd = test_dataset[0]
sample2_ic, sample2_fd = test_dataset[0]

# Check if they're identical
ic_identical = torch.allclose(sample1_ic, sample2_ic)
fd_identical = torch.allclose(sample1_fd, sample2_fd)

print(f"\nSame sample loaded twice:")
print(f"  IC identical? {ic_identical}")
print(f"  FD identical? {fd_identical}")

if ic_identical and fd_identical:
    print("  ❌ AUGMENTATION IS NOT WORKING - samples are identical!")
else:
    print("  ✓ AUGMENTATION IS WORKING - samples differ!")

print("="*60 + "\n")

# ==================== MODEL SETUP ====================
B = 8
C = 1
dimensions = [64, 64, 64]
box_size = 25.0  # CAMELS box size in Mpc/h 

print("Creating model...")

print(f"Batch size: {B}, Channels: {C}, Dimensions: {dimensions}")
# possibly change the channel multiplier to [2,2,1,1,1] like in ronan's paper ?

net = NCSNpp(channels=C, nf=32, ch_mult=[2, 2, 2, 1], dimensions=3, dropout=0.1,
             condition=('input',), condition_input_channels=1).to('cuda')
model = ScoreModel(model=net, sigma_min=0.01, sigma_max=250, device="cuda") # increase sigma_max

save_dir = f'/work/hdd/bdne/abonab/run_{run_number}'
os.makedirs(save_dir, exist_ok=True)

# train 2-3 epochs at a time
# for loop that trains for 5 epochs at a time, then runs or copy and past code from test script,
# print out cross power and power spectrum as a function of epochs
# then goes back to training
# once i have saved the actual numbers themselves with np.save, we will have 20 ish files 
# epoch 5 k, can load in jupyter notebook then plot on the same axis

# ==================== TRAINING OR LOADING ====================

if args.load_checkpoint:
    # LOAD FROM CHECKPOINT
    checkpoint_run = args.checkpoint_run if args.checkpoint_run is not None else run_number
    checkpoint_path = f'/work/hdd/bdne/abonab/run_{checkpoint_run}/model_epoch_{args.checkpoint_epoch}.pth'
    
    print(f"Loading checkpoint from: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path)
    model.model.load_state_dict(checkpoint['model_state_dict'])
    model.model.eval()
    print(f"Successfully loaded model from epoch {checkpoint['epoch']}")
    
    losses = None  # No training losses
    
else:
    # TRAIN FROM SCRATCH
    train_dataset = Dataset(start_idx=0, length=950)
    val_dataset = Dataset(start_idx=950, length=50)
    
    import time
    start_time = time.time()
    losses = model.fit(train_dataset, epochs=60, batch_size=B, learning_rate=1e-4,
                       checkpoints_directory=f'/work/hdd/bdne/abonab/run_42', val_dataset=val_dataset)#), model_checkpoint=10)
    end_time = time.time()
    print(f"Finished training after {(end_time - start_time) / 3600:.3f} hours.")
    
    # Plot losses
    print(losses)
    plt.figure(figsize=(10, 6))
    plt.plot(losses[0], label='train')
    plt.plot(losses[1], label='val')
    plt.title('Training & Validation Loss')
    plt.xlabel('epochs')
    plt.ylabel('loss values')
    plt.legend()
    plt.savefig(f'{save_dir}/training_loss_{run_number}.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Loss plot saved to {save_dir}/training_loss_{run_number}.png")

# ==================== SAMPLING AND ANALYSIS ====================

# Load the full arrays
fd_all = np.load('/work/hdd/bdne/abonab/downsampled_fd_64_hydrogas_norm.npy')
ic_all = np.load('/work/hdd/bdne/abonab/downsampled_ic_64_hydrogas_norm.npy') # can replace with z=127 files

print(f"\nfd_all shape: {fd_all.shape}")
print(f"ic_all shape: {ic_all.shape}")

# Use a validation sample
sample_idx = 950  # First validation sample (valid range: 950-999)
fd_example = fd_all[sample_idx]  # Shape: (64, 64, 64)
ic_example = ic_all[sample_idx]  # Shape: (64, 64, 64)

# Add batch and channel dimensions
fd_tensor = torch.tensor(fd_example[None, None, :]).float().to('cuda')
print(f"fd_tensor shape: {fd_tensor.shape}")

# Generate prediction
print("\nGenerating prediction...")
with torch.no_grad():
    samples = model.sample(condition=[fd_tensor], shape=[B, C, *dimensions], steps=500)

# Extract prediction
prediction = samples[0, 0].cpu().numpy()  # (64, 64, 64)
truth = ic_example  # (64, 64, 64)

# ==================== BASIC VISUALIZATION ====================

f, axs = plt.subplots(1, 2, figsize=(12, 5))
axs[0].imshow(np.mean(prediction[:5], axis=0))
axs[0].set_title('Prediction')

axs[1].imshow(np.mean(truth[:5], axis=0))
axs[1].set_title('True IC')

plt.savefig(f'{save_dir}/prediction_comparison_{run_number}.png', dpi=150, bbox_inches='tight')
plt.close()
print(f"Prediction plot saved to {save_dir}/prediction_comparison_{run_number}.png")

# ==================== POWER SPECTRUM & CROSS-CORRELATION ====================

print("\nComputing power spectrum and cross-correlation...")

# Prepare overdensity fields (delta = rho/rho_mean - 1)
delta_pred = (prediction / np.mean(prediction)).astype(np.float32)
delta_true = (truth / np.mean(truth)).astype(np.float32)

# Compute 3D Power Spectrum
Pk = PKL.XPk([delta_pred, delta_true], BoxSize=box_size, axis=0, MAS=['CIC','CIC'], threads=1)

# Compute Cross-Correlation Function
CCF = PKL.XPk(delta_pred, delta_true, BoxSize=box_size, MAS=['CIC', 'CIC'], axis=0, threads=1)
# cross power spectrum is better
# same package ####PKL.XPk 

# ADD THIS
# plot of cross power spectrum normalized from each individual field
# <XY> = \int dX dY X*Y
# field X and field Y
#quantity = <XY> / sqrt(<XX> <YY>)
# when they are equal (xx=yy) = 1, on small scales should drop to 0, but large scales
# x should be as close to y as possible
# if we can get as close to 1, this is a success !
# k ~ 2pi / r
# change axis on right to axis on left on x axis

# Plot Power Spectrum and Cross-Correlation
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# 3D Power Spectrum
axes[0].loglog(Pk.k3D, Pk.Pk[:,0,0], 'b-', label='Prediction', linewidth=2)
axes[0].loglog(Pk.k3D, Pk.Pk[:,0,1], 'r--', label='True IC', linewidth=2)
axes[0].set_xlabel('k [h/Mpc]', fontsize=12)
axes[0].set_ylabel('P(k) [(Mpc/h)³]', fontsize=12)
axes[0].set_title('3D Power Spectrum (Monopole)', fontsize=14)
axes[0].legend(fontsize=11)
axes[0].grid(True, alpha=0.3)

# Cross-Correlation Function # quantity function
axes[1].plot(CCF.r3D, CCF.xi[:,0], 'k-', linewidth=2) # possibly change second axis on this to quantity
axes[1].set_xlabel('r [Mpc/h]', fontsize=12) # this should be k [h/Mpc]
axes[1].set_ylabel('ξ(r)', fontsize=12)
axes[1].set_title('3D Cross-Correlation Function (Monopole)', fontsize=14)
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(f'{save_dir}/pk_ccf_analysis_{run_number}.png', dpi=150, bbox_inches='tight')
plt.close()
print(f"Power spectrum and CCF plot saved to {save_dir}/pk_ccf_analysis_{run_number}.png")

# ==================== METRICS ====================

print(f"\n{'='*60}")
print(f"Analysis Metrics (Sample {sample_idx})")
print(f"{'='*60}")

# Basic statistics
print(f"\nBasic Statistics:")
print(f"  IC:         min={truth.min():.3f}, max={truth.max():.3f}, std={truth.std():.3f}")
print(f"  Prediction: min={prediction.min():.3f}, max={prediction.max():.3f}, std={prediction.std():.3f}")

# Variance recovery
var_true = truth.var()
var_pred = prediction.var()
variance_recovery = (var_pred / var_true) * 100
print(f"\nVariance Recovery: {variance_recovery:.1f}%")

# Power spectrum metrics
mean_pk_ratio = np.mean(Pk.Pk[:,0,0] / Pk.Pk[:,0,1])
print(f"\nPower Spectrum:")
print(f"  Mean P(k) ratio (pred/true): {mean_pk_ratio:.4f}")

# Cross-correlation metrics
max_ccf = np.max(CCF.xi[:,0])
print(f"\nCross-Correlation:")
print(f"  Max CCF: {max_ccf:.4f}")

print(f"{'='*60}\n")

# Save metrics to file
with open(f'{save_dir}/metrics_{run_number}.txt', 'w') as f:
    if args.load_checkpoint:
        checkpoint_run = args.checkpoint_run if args.checkpoint_run is not None else run_number
        f.write(f"Loaded from run {checkpoint_run}, epoch {args.checkpoint_epoch}\n")
    else:
        f.write(f"Trained fresh for 5 epochs\n")
    f.write(f"Sample index: {sample_idx}\n")
    f.write(f"{'='*60}\n\n")
    f.write(f"Basic Statistics:\n")
    f.write(f"  IC:         min={truth.min():.3f}, max={truth.max():.3f}, std={truth.std():.3f}\n")
    f.write(f"  Prediction: min={prediction.min():.3f}, max={prediction.max():.3f}, std={prediction.std():.3f}\n")
    f.write(f"\nVariance Recovery: {variance_recovery:.1f}%\n")
    f.write(f"\nPower Spectrum:\n")
    f.write(f"  Mean P(k) ratio (pred/true): {mean_pk_ratio:.4f}\n")
    f.write(f"\nCross-Correlation:\n")
    f.write(f"  Max CCF: {max_ccf:.4f}\n")

print(f"Metrics saved to {save_dir}/metrics_{run_number}.txt")
print(f"\nAll done! Results saved to {save_dir}/")
quantity = Pk.XPk[:,0,0] / np.sqrt(Pk.Pk[:,0,0]*Pk.Pk[:,0,1])
print(quantity)