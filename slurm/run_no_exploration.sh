#!/bin/bash
#SBATCH --job-name=adjoint_no_exploration
#SBATCH --account=flows
#SBATCH --partition=gpu
#SBATCH --qos=flows_high
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32GB
#SBATCH --time=24:00:00
#SBATCH --output=logs/no_exploration_%j.out
#SBATCH --error=logs/no_exploration_%j.err

# Create logs directory if it doesn't exist
mkdir -p logs

# Print job information
echo "Job ID: $SLURM_JOB_ID"
echo "Job name: $SLURM_JOB_NAME"
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

# Run the experiment with no exploration (standard behavior)
python train.py \
    experiment=lennard_jones \
    wandb_name="lennard_jones_no_exploration_baseline" \
    use_wandb=true

echo "Job completed at: $(date)" 