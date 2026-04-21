#!/bin/bash
#SBATCH --job-name=1P   # Job name
#SBATCH --array=0-29         # Job array range for lhid
#SBATCH --nodes=1               # Number of nodes
#SBATCH --ntasks=4            # Number of tasks
#SBATCH --time=04:00:00         # Time limit
#SBATCH --partition=cpu  # Partition name
#SBATCH --account=bdne-delta-cpu  # Account name
#SBATCH --output=/work/hdd/bdne/maho3/jobout/%x_%A_%a.out  # Output file for each array task
#SBATCH --error=/work/hdd/bdne/maho3/jobout/%x_%A_%a.out   # Error file for each array task


# set -e

# SLURM_ARRAY_TASK_ID=0

source ~/.bashrc
conda activate cmass
idx=$SLURM_ARRAY_TASK_ID
echo $idx


# Command to run for each lhid
cd /u/maho3/git/diffusion-ic/scripts

suite=1P

python ./process_camels.py --suite=$suite --idx=$idx
