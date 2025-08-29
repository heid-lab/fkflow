import numpy as np
import torch

# Adapted from https://filterpy.readthedocs.io/en/latest/monte_carlo/resampling.html

class Resampler():
    def __init__(self, resample_method: str = "multinomial"):
        self.resample_method = resample_method

    def resample(self, weights, num_samples):
        if self.resample_method == "multinomial":
            return self.multinomial_resample(weights, num_samples)
        elif self.resample_method == "stratified":
            return self.stratified_resample(weights, num_samples)
        elif self.resample_method == "residual":
            return self.residual_resample(weights, num_samples)
        elif self.resample_method == "systematic":
            return self.systematic_resample(weights, num_samples)
        else:
            raise ValueError(f"Unknown resampling method: {self.resample_method}")
        
    def multinomial_resample(self, weights, num_samples):
        return torch.multinomial(weights, num_samples, replacement=True)
    
    def _handle_batched(self, weights, num_samples, resample_fn):
        """Helper to handle 1D/2D weights uniformly."""
        if weights.dim() == 1:
            return resample_fn(weights, num_samples)
        
        # Batch processing for 2D
        batch_size = weights.shape[0]
        return torch.stack([resample_fn(weights[i], num_samples) for i in range(batch_size)])

    def residual_resample(self, weights, num_samples):
        """Efficient vectorized residual resampling for particle filters."""
        
        def _resample_1d(w, n):
            w = w / w.sum()
            num_copies = torch.floor(n * w).long()
            
            deterministic_indices = torch.repeat_interleave(
                torch.arange(len(w), device=w.device), num_copies
            )
            
            remaining = n - len(deterministic_indices)
            if remaining <= 0:
                return deterministic_indices
            
            residual = torch.clamp(w - num_copies.float() / n, min=0)
            if residual.sum() == 0:
                return deterministic_indices
            
            residual_indices = torch.multinomial(residual, remaining, replacement=True)
            return torch.cat([deterministic_indices, residual_indices])
        
        return self._handle_batched(weights, num_samples, _resample_1d)

    def stratified_resample(self, weights, num_samples):
        """Efficient stratified resampling for particle filters."""
        
        def _resample_1d(w, n):
            w = w / w.sum()
            cumulative_sum = torch.cumsum(w, dim=0)
            
            positions = (torch.rand(n, device=w.device) + 
                        torch.arange(n, device=w.device, dtype=torch.float)) / n
            
            indexes = torch.searchsorted(cumulative_sum, positions)
            return torch.clamp(indexes, 0, len(w) - 1)
        
        return self._handle_batched(weights, num_samples, _resample_1d)

    def systematic_resample(self, weights, num_samples):
        """Efficient systematic resampling for particle filters."""
        
        def _resample_1d(w, n):
            w = w / w.sum()
            cumulative_sum = torch.cumsum(w, dim=0)
            
            positions = (torch.rand(1, device=w.device) + 
                        torch.arange(n, device=w.device, dtype=torch.float)) / n
            
            indexes = torch.searchsorted(cumulative_sum, positions)
            return torch.clamp(indexes, 0, len(w) - 1)
        
        return self._handle_batched(weights, num_samples, _resample_1d)