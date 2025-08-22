# Copyright (c) Meta Platforms, Inc. and affiliates.

import math
from abc import ABC, abstractmethod
from typing import Callable
import torch


class BaseExploration(ABC):
    """
    Base class for exploration strategies in adjoint sampling.
    Provides temperature and noise addition mappings as functions of time t.
    """
    
    def __init__(self):
        # Epoch scheduling defaults (no-op by default)
        self._current_epoch: int = 0
        self._end_epoch: int | None = None
    
    @abstractmethod
    def temperature(self, t: torch.Tensor) -> torch.Tensor:
        """
        Map time t (between 0 and 1) to temperature (float >= 1).
        
        Args:
            t: Time tensor with values between 0 and 1
            
        Returns:
            Temperature tensor with values >= 1
        """
        pass
    
    @abstractmethod 
    def noise_addition(self, t: torch.Tensor) -> torch.Tensor:
        """
        Map time t (between 0 and 1) to additional noise scaling.
        
        Args:
            t: Time tensor with values between 0 and 1
            
        Returns:
            Noise addition tensor (can be any float value)
        """
        pass
    
    def compute_drift(self, f: torch.Tensor, t: torch.Tensor, dt: float) -> torch.Tensor:
        """
        Compute exploration-modified drift term.
        
        Args:
            f: Drift function output
            t: Time tensor
            dt: Time step
            
        Returns:
            Modified drift: temperature(t) * f * dt
        """
        temp = self.temperature(t)
        return temp * f * dt
    
    def compute_diffusion(self, t: torch.Tensor, sde, dt: float, noise: torch.Tensor) -> torch.Tensor:
        """
        Compute exploration-modified diffusion term.
        
        Args:
            t: Time tensor
            sde: SDE object with g(t) method
            dt: Time step
            noise: Random noise tensor
            
        Returns:
            Modified diffusion: sde.g(t) * noise_addition(t) * sqrt(dt) * noise
        """
        g_t = sde.g(t)
        noise_multiplier = self.noise_addition(t)
        # Apply epoch-dependent factor (default 1.0)
        epoch_factor = self._get_epoch_factor_tensor(t, noise)
        return g_t * (1+(noise_multiplier-1) * epoch_factor) * math.sqrt(dt) * noise

    def set_epoch(self, epoch: int, end_epoch: int | None = None) -> None:
        """Update internal epoch state for epoch-dependent scheduling.

        Args:
            epoch: Current training epoch (0-indexed)
            end_epoch: Final epoch (exclusive upper bound or last epoch+1). If provided,
                       overrides any previously set value.
        """
        self._current_epoch = max(0, int(epoch))
        if end_epoch is not None:
            self._end_epoch = max(1, int(end_epoch))

    def _get_epoch_factor(self) -> float:
        """Scalar factor applied to diffusion multiplier for epoch scheduling (default 1.0)."""
        return 1.0

    def _get_epoch_factor_tensor(self, t: torch.Tensor, noise: torch.Tensor | None) -> torch.Tensor:
        factor = self._get_epoch_factor()
        device = t.device if torch.is_tensor(t) else (noise.device if torch.is_tensor(noise) else None)
        dtype = t.dtype if torch.is_tensor(t) else (noise.dtype if torch.is_tensor(noise) else torch.float32)
        return torch.as_tensor(factor, device=device, dtype=dtype)


class NoExploration(BaseExploration):
    """
    No exploration strategy - standard temperature=1 and no additional noise.
    """
    
    def temperature(self, t: torch.Tensor) -> torch.Tensor:
        """Always return temperature of 1."""
        return torch.ones_like(t)
    
    def noise_addition(self, t: torch.Tensor) -> torch.Tensor:
        """Always return noise addition of 0."""
        return torch.zeros_like(t)

class SomeStepwiseNoise(BaseExploration):
    """
    Temperature=1, but with some noise addition.
    """
    
    def __init__(
        self,
        noise_scale: float = None,
        start_multiplier: float = 2.0,
        end_multiplier: float = 1.0,
        decay_power: float = 1.0,
        end_epoch: int | None = None,
        epoch_start_factor: float | None = None,
    ):
        """
        Args:
            noise_scale: (Deprecated) Backward-compat alias for start_multiplier.
            start_multiplier: Multiplier at t=0 (default 2.0).
            end_multiplier: Multiplier at t=1 (default 1.0).
            decay_power: Shape of decay from start->end over t in [0,1].
                         1.0 is linear; >1 front-loads (faster early), <1 back-loads.
        """
        super().__init__()
        if noise_scale is not None:
            start_multiplier = noise_scale
        self.start_multiplier = float(start_multiplier)
        self.end_multiplier = float(end_multiplier)
        self.decay_power = float(decay_power)
        # Epoch scheduling parameters
        self._end_epoch = int(end_epoch) if end_epoch is not None else None
        # Default epoch-start factor equals start_multiplier if not provided
        self.epoch_start_factor = float(epoch_start_factor) if epoch_start_factor is not None else self.start_multiplier
    
    def temperature(self, t: torch.Tensor) -> torch.Tensor:
        """Always return temperature of 1."""
        return torch.ones_like(t)
    
    def noise_addition(self, t: torch.Tensor) -> torch.Tensor:
        """Return multiplicative noise multiplier that decays from start to end.

        Schedules a multiplier m(t) over t ∈ [0, 1]:
            m(t) = end + (start - end) * (1 - t^decay_power)
        So m(0)=start and m(1)=end. Default is 2 -> 1 linearly.
        """
        t_clamped = torch.clamp(t, 0.0, 1.0)
        start = torch.as_tensor(self.start_multiplier, device=t_clamped.device, dtype=t_clamped.dtype)
        end = torch.as_tensor(self.end_multiplier, device=t_clamped.device, dtype=t_clamped.dtype)
        power = torch.as_tensor(self.decay_power, device=t_clamped.device, dtype=t_clamped.dtype)
        return end + (start - end) * (1 - t_clamped.pow(power))

    def _get_epoch_factor(self) -> float:
        # If no end epoch known, do not scale across epochs
        if self._end_epoch is None or self._end_epoch <= 0:
            return 1.0
        # Linear decay from epoch_start_factor to 1.0 across epochs [0, end_epoch]
        progress = min(max(self._current_epoch / float(self._end_epoch), 0.0), 1.0)
        return self.epoch_start_factor * (1.0 - progress)        

