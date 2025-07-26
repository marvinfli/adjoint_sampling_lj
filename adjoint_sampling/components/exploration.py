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
        pass
    
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
            Modified diffusion: (sde.g(t) + noise_addition(t)) * sqrt(dt) * noise
        """
        g_t = sde.g(t)
        noise_add = self.noise_addition(t)        
        return (g_t + noise_add) * math.sqrt(dt) * noise


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
    Stepwise noise exploration - temperature=1, but with some noise addition.
    This is just an example - you can modify the noise schedule as needed.
    """
    
    def __init__(self, noise_scale: float = 0.1):
        """
        Args:
            noise_scale: Amount of additional noise to add
        """
        super().__init__()
        self.noise_scale = noise_scale
    
    def temperature(self, t: torch.Tensor) -> torch.Tensor:
        """Always return temperature of 1."""
        return torch.ones_like(t)
    
    def noise_addition(self, t: torch.Tensor) -> torch.Tensor:
        """Return stepwise noise addition."""
        # Add noise_scale when t > step_threshold, otherwise 0
        return self.noise_scale
