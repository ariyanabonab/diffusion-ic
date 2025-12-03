%load_ext autoreload
%autoreload 2
!pip install jax==0.7.0
!pip -q install torch score_models h5py tqdm
!pip install score_models
!pip install jaxpm
import os
import jax
import jax.numpy as jnp
import jax_cosmo as jc
from jax.experimental.ode import odeint
from jaxpm.painting import cic_paint
from jaxpm.pm import linear_field, lpt, make_ode_fn
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

class Dataset(torch.utils.data.Dataset):
    def __init__(self, start_idx, length, ic_stats=None, fd_stats=None):
        self.start_idx = start_idx
        self.length = length
        self.ic_stats = ic_stats
        self.fd_stats = fd_stats
    
    def __len__(self):
        return self.length
    # using the lower res 32,32,32 data (not _hr)
    def __getitem__(self, i):
        i += self.start_idx
        initial_c = np.load(f'ic_{i}.npy')
        final_d = np.load(f'fd_{i}.npy')
        
        # normalizing IC and FD separately
        if self.ic_stats is not None:
            initial_c = (initial_c - self.ic_stats['mean']) / self.ic_stats['std']
        if self.fd_stats is not None:
            final_d = (final_d - self.fd_stats['mean']) / self.fd_stats['std']
        
        initial_c = torch.tensor(initial_c).to('cuda')
        final_d = torch.tensor(final_d).to('cuda')
        
        return initial_c, final_d 
# we have less 64,64,64 data, so i used the 50 .npy files we have, but we could change this to
# length 900 and 100 for the 32,32,32 data. we just have to modify the _hr names above in the script to run it
train_dataset = Dataset(start_idx=0, length=900)
val_dataset = Dataset(start_idx=900, length=100)

from score_models import ScoreModel, EnergyModel, NCSNpp, MLP, DDPM
from torch.utils.data import Dataset # Import Dataset
import torch # Import torch
B = 8 # changed from 3
C = 1
dimensions = [32, 32, 32]

# NN Architectures support a Unet with 1D convolutions for time series input data
net = NCSNpp(channels=C, nf=128, ch_mult=[2, 2, 2, 2], dimensions=3, condition = ('input',), condition_input_channels = 1,).to('cuda')
model = ScoreModel(model=net, sigma_min=0.01, sigma_max=50, device="cuda") # changed beta_max from 40 to 10 # changing beta_min=1e-2, beta_max=10,to sigma_max and sigma_min
losses = model.fit(train_dataset, val_dataset, epochs=10, batch_size=B, learning_rate=2e-4, checkpoints_directory='/work/hdd/bdne/abonab/run_69') 

import matplotlib.pyplot as plt
print(losses)
plt.plot(losses[0], label='train') # training loss, not validation loss
plt.plot(losses[1], label='val')
plt.title('Training & Validation Loss')
plt.xlabel('epochs')
plt.ylabel('loss values')
plt.legend()

# correct code input & conditioning
i = 1000
fd_example = np.load('/work/hdd/bdne/abonab/' + f'fd_{i}.npy')
ic_example = np.load('/work/hdd/bdne/abonab/' + f'ic_{i}.npy')

ic_example = torch.tensor([ic_example]).to('cuda')  # Convert IC to tensor
fd_example = torch.tensor([fd_example]).to('cuda') # Converting FD to tensor

B = 1 # maybe change to 8?
C = 1
dimensions = [32, 32, 32]

# Condition on fd to output ic
samples = model.sample(condition=[fd_example], shape=[B, C, *dimensions], steps=500)
#output image slices
x = samples[0, 0].cpu().numpy()          # Prediction
y = np.load(f'/work/hdd/bdne/abonab/ic_1000.npy')[0]   #ic

f, axs = plt.subplots(1, 2)
axs[0].imshow(np.mean(x[:5], axis=0))
axs[0].set_title('Prediction')

axs[1].imshow(np.mean(y[:5], axis=0))
axs[1].set_title('True IC')
plt.show()

print(f"IC: min={y.min():.3f}, max={y.max():.3f}, std={y.std():.3f}")
print(f"Prediction: min={x.min():.3f}, max={x.max():.3f}, std={x.std():.3f}")
# calculate the variance
var_true = y.var()
var_pred = x.var()
variance_recovery = (var_pred / var_true) * 100
print(f"Variance recovery: {variance_recovery:.1f}%") # our target is 90-100% recovery.
