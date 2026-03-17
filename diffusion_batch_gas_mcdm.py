# !pip install jax==0.7.0
# !pip -q install torch score_models h5py tqdm
# !pip install score_models
# !pip install jaxpm

import argparse
parser = argparse.ArgumentParser()
# parser.add_argument('--run_number', type=int, default=0, 
#                     help='Run number for saving checkpoints') # can be overridden through .sh script
# parser.add_argument('--checkpoint_file', type=str, default='checkpoint_6.9958e+04_015.pt', # change
# parser.add_argument('--load_checkpoint', action='store_true', default=False,
#                     help='Whether to load from a checkpoint')
#                     help='Checkpoint filename') # not sure if this is better or the integer number
# parser.add_argument('--checkpoint_number', type=int, default=15, # integer number checkpoint
#                     help='Checkpoint Number')
# parser.add_argument('--checkpoint_run', type=int, default=None,
#                     help='Run number of checkpoint to load (if different from run_number)')
# parser.add_argument('--checkpoint_epoch', type=int, default=5,
#                     help='Which epoch checkpoint to load')
parser.add_argument('--run_number', type=int, default=0, 
                     help='Run number for saving checkpoints')
parser.add_argument('--checkpoint_file', type=str, default='checkpoint_6.9958e+04_015.pt',
                    help='Checkpoint filename')
parser.add_argument('--checkpoint_number', type=int, default=15,
                    help='Checkpoint Number')
parser.add_argument('--checkpoint_run', type=int, default=None,
                    help='Run number of checkpoint to load (if different from run_number)')
parser.add_argument('--checkpoint_epoch', type=int, default=5,
                    help='Which epoch checkpoint to load')
parser.add_argument('--load_checkpoint', action='store_true', default=False,
                    help='Whether to load from a checkpoint')
args = parser.parse_args()
run_number = args.run_number
checkpoint_number = args.checkpoint_number
print(f"Run number: {run_number}")
print(f"Checkpoint number: {checkpoint_number}")

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
        all_ics = np.load(f'/work/hdd/bdne/abonab/ic_all_norm.npy')
        all_fds = np.load(f'/work/hdd/bdne/abonab/fd_64_gas_mcdm.npy')
        
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
model = ScoreModel(model=net, sigma_min=0.01, sigma_max=400, device="cuda") # increase sigma_max

save_dir = f'/work/hdd/bdne/abonab/run_{run_number}'
os.makedirs(save_dir, exist_ok=True)

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
    
    if checkpoint_number == 0:
        # TRAIN FROM SCRATCH
        checkpoint_number = None

    train_dataset = Dataset(start_idx=0, length=950)
    val_dataset = Dataset(start_idx=950, length=50)
    
    import time
    start_time = time.time()
    losses = model.fit(train_dataset, epochs=100, batch_size=B, learning_rate=1e-4,
                       checkpoints_directory=f'/work/hdd/bdne/abonab/run_50', val_dataset=val_dataset, model_checkpoint=checkpoint_number)
    end_time = time.time()
    print(f"Finished training after {(end_time - start_time) / 3600:.3f} hours.")
    
    # Plot losses
    # fix this so losses are plotted with checkpoints
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
    with open(f'{save_dir}/losses.txt', 'a') as f:
        np.savetxt(f, losses.T, delimiter=',')
    # this will save the epochs and show as columns / rows of the loss values i can use in a jp
    # from the batch job specify what checkpoint to run at

# ==================== SAMPLING AND ANALYSIS ====================
