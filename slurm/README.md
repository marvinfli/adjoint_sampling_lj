# SLURM Scripts for Adjoint Sampling Experiments

This directory contains SLURM batch scripts for running adjoint sampling experiments on the cluster.

## Scripts

### 1. `run_no_exploration.sh`
Runs the baseline Lennard-Jones experiment with no exploration (standard adjoint sampling).

**Command:**
```bash
sbatch slurm/run_no_exploration.sh
```

**Configuration:**
- Experiment: `lennard_jones` 
- Exploration: None (default behavior)
- WandB name: `lennard_jones_no_exploration_baseline`
- Runtime: 24 hours
- Resources: 1 H100 80GB GPU, 8 CPUs, 100GB RAM

### 2. `run_exploration_sweep.sh`
Runs a parameter sweep over exploration noise_scale values using SLURM job arrays.

**Command:**
```bash
sbatch slurm/run_exploration_sweep.sh
```

**Configuration:**
- Experiment: `lennard_jones_with_exploration`
- Exploration: SomeStepwiseNoise with varying `noise_scale`
- Noise scale values: 0.001, 0.01, 0.1 (geometric sequence)
- WandB names: `lennard_jones_exploration_noise_{value}` (e.g., `lennard_jones_exploration_noise_0.001`)
- Runtime: 48 hours per job
- Resources: 1 H100 80GB GPU, 8 CPUs, 100GB RAM per job
- Total jobs: 3 (one for each noise_scale value)

## Usage

1. **Submit single job (no exploration):**
   ```bash
   cd /n/holylabs/LABS/sitanc_lab/Users/mfli/adjoint_sampling_lj
   sbatch slurm/run_no_exploration.sh
   ```

2. **Submit parameter sweep (exploration):**
   ```bash
   cd /n/holylabs/LABS/sitanc_lab/Users/mfli/adjoint_sampling_lj
   sbatch slurm/run_exploration_sweep.sh
   ```

3. **Check job status:**
   ```bash
   squeue -u $USER
   ```

4. **View logs:**
   ```bash
   # For no exploration
   cat logs/no_exploration_<job_id>.out
   
   # For exploration sweep (array_id: 0, 1, or 2)
   cat logs/exploration_sweep_<job_id>_<array_id>.out
   ```

## Results

All experiments will log to WandB with semantic names:
- Baseline: `lennard_jones_no_exploration_baseline`
- Exploration: `lennard_jones_exploration_noise_0.001`, `lennard_jones_exploration_noise_0.01`, `lennard_jones_exploration_noise_0.1`

## Notes

- Both scripts assume the `adjoint_sampling` conda environment is available
- Scripts are configured to use H100 80GB GPUs via `sitanc_lab` account on `gpu,seas_gpu` partitions
- Logs are saved to the `logs/` directory (created automatically)
- The exploration sweep uses SLURM job arrays for parallel execution
- Modify the `noise_scales` array in `run_exploration_sweep.sh` to change parameter values
- Jobs include `--requeue` for automatic resubmission if preempted
- Adjust SLURM directives as needed for your cluster configuration 