# Exploration Strategies for Adjoint Sampling

This directory contains configuration files for different exploration strategies that can be used to modify the SDE integration behavior during adjoint sampling.

## Overview

The exploration framework allows you to modify the drift and diffusion terms during SDE integration by:
- **Temperature scaling**: Multiply the drift by a temperature factor ≥ 1
- **Noise addition**: Add extra noise to the diffusion term

## Available Strategies

### 1. Default (No Config)
- **Exploration**: `null` (None) - this is the default when no exploration config is specified
- **Use case**: Standard adjoint sampling behavior (default)
- **Note**: When exploration is None, the system uses standard SDE integration without any modifications

### 2. NoExploration (`no_exploration.yaml`)
- **Temperature**: Always 1.0 (no scaling)
- **Noise addition**: Always 0.0 (no extra noise)
- **Use case**: Equivalent to default but using explicit NoExploration class

### 3. SomeStepwiseNoise (`stepwise_noise.yaml`)
- **Temperature**: Always 1.0 (no scaling)
- **Noise addition**: `noise_scale` for t > threshold

## Usage

### In Experiment Configs

Add the exploration strategy to your experiment configuration:

```yaml
defaults:
  - override /exploration: stepwise_noise  # or no_exploration, or default

# Optional: Override parameters inline
exploration:
  _target_: adjoint_sampling.components.exploration.SomeStepwiseNoise
  noise_scale: 0.2

# Or explicitly set to None for standard behavior
exploration: null
```

### Running Experiments

```bash
# With stepwise noise exploration
python train.py experiment=lennard_jones_with_exploration

# With no exploration (standard behavior) - uses default exploration=null
python train.py experiment=lennard_jones_no_exploration

# Explicitly override to None from command line
python train.py experiment=lennard_jones_with_exploration exploration=null

# Override exploration parameters from command line
python train.py experiment=lennard_jones_with_exploration exploration.noise_scale=0.15
```

## Mathematical Details

The exploration framework modifies the SDE integration in `euler_maruyama_step`:

### Standard SDE step:
```
x_{t+dt} = x_t + f(x_t, t) * dt + g(t) * sqrt(dt) * noise
```

### With Exploration:
```
x_{t+dt} = x_t + temperature(t) * f(x_t, t) * dt + (g(t) + noise_addition(t)) * sqrt(dt) * noise
```

Where:
- `temperature(t)`: Time-dependent temperature scaling (≥ 1)
- `noise_addition(t)`: Time-dependent additional noise scaling
- `t` ∈ [0, 1]: Normalized time during SDE integration

## Creating Custom Exploration Strategies

To create your own exploration strategy:

1. Inherit from `BaseExploration` in `adjoint_sampling/components/exploration.py`
2. Implement `temperature(t)` and `noise_addition(t)` methods
3. Create a corresponding YAML config file
4. Use it in your experiment configurations

Example:
```python
class CustomExploration(BaseExploration):
    def __init__(self, temp_scale=2.0):
        super().__init__()
        self.temp_scale = temp_scale
    
    def temperature(self, t):
        # Linear increase from 1 to temp_scale
        return 1.0 + (self.temp_scale - 1.0) * t
    
    def noise_addition(self, t):
        # Sinusoidal noise addition
        return 0.1 * torch.sin(math.pi * t)
``` 