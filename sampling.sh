#!/bin/bash
#SBATCH --mem=64g
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --ntasks-per-socket=1
#SBATCH --cpus-per-task=100
#SBATCH --partition=ghx4
#SBATCH --time=00:35:00
#SBATCH --job-name=sample_only
#SBATCH --account=bdne-dtai-gh
#SBATCH --gpus-per-node=1
#SBATCH --gpu-bind=verbose,closest

echo "job is starting on `hostname`"
cd /work/hdd/bdne/abonab/notebooks

module load cuda/12.6.1

source ~/.bashrc
conda activate icdiff

# Sample from run_28 checkpoint, save to run_30
python ./sampling_multi_mod.py --checkpoint_run 38 --output_run 171 --sample_indices 966 967 968 --steps 500