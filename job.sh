#!/bin/bash
#SBATCH --mem=64g
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --ntasks-per-socket=1
#SBATCH --cpus-per-task=100
#SBATCH --partition=ghx4
#SBATCH --time=24:00:00
#SBATCH --job-name=pytorch
#SBATCH --account=bdne-dtai-gh
#SBATCH --gpus-per-node=1
#SBATCH --gpu-bind=verbose,closest

echo "job is starting on `hostname`"
cd /work/hdd/bdne/abonab/notebooks

module load cuda/12.6.1

source ~/.bashrc
conda activate icdiff

run_number=38
python ./diffusion_batch_CAMELS_hydro.py --run_number=$run_number