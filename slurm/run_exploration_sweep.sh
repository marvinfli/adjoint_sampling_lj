#!/bin/bash
#SBATCH --job-name=adjoint_exploration_sweep
#SBATCH --account=sitanc_lab
#SBATCH --partition=gpu,seas_gpu
#SBATCH --gres=gpu:nvidia_h100_80gb_hbm3:1
#SBATCH --mem=100G
#SBATCH --time=10:00:00
#SBATCH --output=logs/exploration_sweep_%j.out
#SBATCH --error=logs/exploration_sweep_%j.err
#SBATCH --array=0-2  # 3 different noise_scale values
#SBATCH --requeue

# Create logs directory if it doesn't exist
mkdir -p logs

# Print job information
echo "Job ID: $SLURM_JOB_ID"
echo "Job name: $SLURM_JOB_NAME"
echo "Array task ID: $SLURM_ARRAY_TASK_ID"
echo "Node: $SLURM_NODELIST"
echo "Start time: $(date)"

# Load conda environment
source ~/.bashrc
conda activate adjoint_sampling

# Change to the project directory
cd /n/holylabs/LABS/sitanc_lab/Users/mfli/adjoint_sampling_lj

# Set environment variables
export CUDA_VISIBLE_DEVICES=0
export PROJECTROOT=/n/holylabs/LABS/sitanc_lab/Users/mfli/adjoint_sampling_lj

# Define noise_scale values (geometric sequence from 1e-3 to 1e-1)
noise_scales=(0.001 0.01 0.1)
noise_scale=${noise_scales[$SLURM_ARRAY_TASK_ID]}

echo "Running with noise_scale=${noise_scale}"

# Create meaningful wandb name that includes the noise scale
wandb_name="lennard_jones_exploration_noise_${noise_scale}"

# Run the experiment with exploration parameter sweep
python train.py \
    experiment=lennard_jones_with_exploration \
    exploration.noise_scale=${noise_scale} \
    wandb_name="${wandb_name}" \
    use_wandb=true

echo "Job completed at: $(date)" 
