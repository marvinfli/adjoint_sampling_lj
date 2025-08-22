#!/bin/bash
#SBATCH --job-name=adjoint_exploration_sweep_new
#SBATCH --account=sitanc_lab
#SBATCH --partition=gpu,seas_gpu
#SBATCH --gres=gpu:nvidia_h100_80gb_hbm3:1
#SBATCH --mem=100G
#SBATCH --time=10:00:00
#SBATCH --output=logs/exploration_sweep_new_%j.out
#SBATCH --error=logs/exploration_sweep_new_%j.err
#SBATCH --array=0-2  # 3 different start_multiplier values
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

# Define start_multiplier values to sweep
start_multipliers=(2 4 16)
start_multiplier=${start_multipliers[$SLURM_ARRAY_TASK_ID]}

echo "Running with start_multiplier=${start_multiplier}"

# Create meaningful wandb name that includes the start multiplier
wandb_name="lennard_jones_exploration_startmul_${start_multiplier}"

# Run the experiment with exploration parameter sweep
python train.py \
    experiment=lennard_jones_keely_param_with_exploration \
    exploration.start_multiplier=${start_multiplier} \
    wandb_name="${wandb_name}" \
    use_wandb=true

echo "Job completed at: $(date)" 
