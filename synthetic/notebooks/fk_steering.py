"""
Feynman-Kac Steering for Flow Models

This module implements FK steering for synthetic 2D distribution experiments,
extracted and adapted from the main FlowModule._fk_generate method to work
with the guided flow notebook interface.

Key Features:
- Pure deterministic flow (no SDE terms, unlike hypercube implementation)
- Multiple resampling strategies (multinomial, stratified, systematic, residual)
- Event-based potential scheduling (sum, harmonic_sum, difference)
- Compatible with synthetic notebook experiment framework
- Gradient-based potential steering option

The implementation adapts FK steering to work with 2D synthetic distributions
in the guided flow experiments, providing a bridge between the FK steering
theory and the existing guided flow experimental framework.

Usage:
    # Create FK wrapper for a trained model
    fk_wrapper = FKSteeringWrapper(
        model=trained_model,
        x1_dist=target_distribution,
        num_samples=8,
        fk_steering_temperature=1.0
    )
    
    # Generate trajectory for visualization
    trajectory = fk_wrapper.generate_trajectory(x0_sampler, ode_cfg)
    
    # Or use the evaluation function for notebook compatibility
    trajectory = evaluate_fk(x0_sampler, x1_sampler, model, x1_dist, fk_params, ode_cfg)

Adapted from the original FlowModule implementation for compatibility with
the "On the Guidance of Flow Matching" experimental framework.
"""

import torch
from typing import Optional, Callable
from functools import partial


# Resampling methods - copied from resampling.py
def _ensure_batched(weights: torch.Tensor) -> tuple[torch.Tensor, bool]:
    """Ensure weights tensor is 2D (B, S) format for resampling operations."""
    if weights.dim() == 1:
        return weights.unsqueeze(0), True
    if weights.dim() != 2:
        raise ValueError(f"weights must be 1D or 2D, got shape {weights.shape}")
    return weights, False


def multinomial_resample(weights: torch.Tensor, num_samples: int) -> torch.Tensor:
    weights_B_S, squeezed = _ensure_batched(weights)
    B, S = weights_B_S.shape
    # normalize
    probs_B_S = weights_B_S / (weights_B_S.sum(dim=1, keepdim=True) + 1e-12)
    idx_B_S = torch.stack([
        torch.multinomial(probs_B_S[b], num_samples, replacement=True)
        for b in range(B)
    ], dim=0)
    return idx_B_S


def stratified_resample(weights: torch.Tensor, num_samples: int) -> torch.Tensor:
    weights_B_S, _ = _ensure_batched(weights)
    B, S = weights_B_S.shape
    probs_B_S = weights_B_S / (weights_B_S.sum(dim=1, keepdim=True) + 1e-12)
    cdf_B_S = torch.cumsum(probs_B_S, dim=1)
    # positions: (B, num_samples) stratified in [0,1]
    # u_b ~ U[0,1/S], then positions = (u_b + k)/num_samples
    u_B_1 = torch.rand(B, 1, device=weights_B_S.device) / num_samples
    positions_B_Sr = u_B_1 + (torch.arange(num_samples, device=weights_B_S.device).float().unsqueeze(0) / num_samples)
    # searchsorted per row
    # torch.searchsorted expects (S,) so loop per batch
    idx_rows = []
    for b in range(B):
        idx_rows.append(torch.searchsorted(cdf_B_S[b], positions_B_Sr[b], right=False).clamp_max(S - 1))
    return torch.stack(idx_rows, dim=0)


def systematic_resample(weights: torch.Tensor, num_samples: int) -> torch.Tensor:
    weights_B_S, _ = _ensure_batched(weights)
    B, S = weights_B_S.shape
    probs_B_S = weights_B_S / (weights_B_S.sum(dim=1, keepdim=True) + 1e-12)
    cdf_B_S = torch.cumsum(probs_B_S, dim=1)
    # single offset shared per batch element
    u_B_1 = torch.rand(B, 1, device=weights_B_S.device) / num_samples
    positions_B_Sr = u_B_1 + (torch.arange(num_samples, device=weights_B_S.device).float().unsqueeze(0) / num_samples)
    idx_rows = []
    for b in range(B):
        idx_rows.append(torch.searchsorted(cdf_B_S[b], positions_B_Sr[b], right=False).clamp_max(S - 1))
    return torch.stack(idx_rows, dim=0)


def residual_resample(weights: torch.Tensor, num_samples: int) -> torch.Tensor:
    weights_B_S, _ = _ensure_batched(weights)
    B, S = weights_B_S.shape
    probs_B_S = weights_B_S / (weights_B_S.sum(dim=1, keepdim=True) + 1e-12)

    out_idx = []
    for b in range(B):
        probs = probs_B_S[b]
        num_copies = torch.floor(num_samples * probs).long()
        deterministic_indices = torch.repeat_interleave(torch.arange(S, device=weights.device), num_copies)
        remaining = num_samples - deterministic_indices.numel()
        if remaining > 0:
            residual = (probs - num_copies.float() / num_samples).clamp_min(0)
            if residual.sum() > 0:
                res_idx = torch.multinomial(residual, remaining, replacement=True)
                idx_b = torch.cat([deterministic_indices, res_idx])
            else:
                # fallback if residual vanished numerically
                idx_b = deterministic_indices
                # pad if needed
                if idx_b.numel() < num_samples:
                    pad = idx_b.new_empty(0, dtype=idx_b.dtype)
                    idx_b = torch.cat([idx_b, pad])
        else:
            idx_b = deterministic_indices[:num_samples]
        # If due to rounding, idx_b can be shorter; pad by repeating last index
        if idx_b.numel() < num_samples:
            pad = idx_b[-1].repeat(num_samples - idx_b.numel())
            idx_b = torch.cat([idx_b, pad])
        out_idx.append(idx_b)
    return torch.stack(out_idx, dim=0)


def resample(weights: torch.Tensor, num_samples: int, method: str = "stratified") -> torch.Tensor:
    """
    Resample particle indices based on normalized weights using specified method.
    
    Args:
        weights: Unnormalized weights tensor (B, S) or (S,)
        num_samples: Number of samples to resample
        method: Resampling method ('multinomial', 'stratified', 'systematic', 'residual')
        
    Returns:
        torch.Tensor: Resampled indices (B, num_samples)
    """
    if method == "multinomial":
        return multinomial_resample(weights, num_samples)
    if method == "stratified":
        return stratified_resample(weights, num_samples)
    if method == "systematic":
        return systematic_resample(weights, num_samples)
    if method == "residual":
        return residual_resample(weights, num_samples)
    raise ValueError(f"Unknown resampling method: {method}")


class FKSteeringWrapper:
    """
    Wraps any MLP model with FK steering algorithm (pure deterministic flow).
    
    This wrapper implements FK steering for 2D synthetic distributions without
    SDE noise terms (unlike the hypercube implementation). It uses pure deterministic
    flow with periodic resampling based on potential-weighted importance weights.
    
    The implementation follows the event-based FK algorithm with configurable:
    - Resampling strategies (multinomial, stratified, systematic, residual)
    - Potential schedulers (sum, harmonic_sum, difference)  
    - Optional gradient nudging against potential gradients
    
    Args:
        model: Trained flow model that predicts velocity field v_t(x,t)
        x1_dist: Target distribution providing potential function via get_J(x)
        num_samples: Number of particles for particle filter
        num_steps: Number of flow integration steps
        fk_steering_temperature: Temperature β for FK steering (higher = more aggressive)
        fk_potential_scheduler: How to schedule potentials ("sum", "harmonic_sum", "difference")
        gradient_factor: Optional gradient nudging factor against ∇U (None to disable)
        resample_method: Resampling strategy ("multinomial", "stratified", "systematic", "residual")
        resample_freq: Resampling frequency (lower = more FK-like)
        scale: Scaling factor for potential function
    """
    
    def __init__(
        self,
        model,
        x1_dist,
        num_samples: int = 8,
        num_steps: int = 32, 
        fk_steering_temperature: float = 1.0,
        fk_potential_scheduler: str = "sum",
        gradient_factor: Optional[float] = None,
        resample_method: str = "residual", 
        resample_freq: int = 2,
        scale: float = 1.0,
    ):
        self.model = model
        self.x1_dist = x1_dist
        self.num_samples = num_samples
        self.num_steps = num_steps
        self.fk_steering_temperature = fk_steering_temperature
        self.fk_potential_scheduler = fk_potential_scheduler
        self.gradient_factor = gradient_factor
        self.resample_method = resample_method
        self.resample_freq = resample_freq
        self.scale = scale
        
        # Create potential function: J = λ*U, so U = J (since λ=1)
        self.potential_fn = lambda x: self.scale * self.x1_dist.get_J(x)
    
    def generate_trajectory(self, x0_sampler, ode_cfg):
        """
        Generate FK steering trajectory in notebook format [T, BS, D] using pure deterministic flow.
        
        This method implements the core FK steering algorithm for 2D synthetic distributions.
        Unlike the hypercube SDE version, this uses pure deterministic flow integration
        with optional gradient nudging and periodic resampling based on potential weights.
        
        Args:
            x0_sampler: Function that samples initial conditions x0_sampler(batch_size) -> Tensor
            ode_cfg: ODE configuration object with batch_size, device, etc.
            
        Returns:
            torch.Tensor: Trajectory of shape [T, BS, D] where T=num_steps, BS=batch_size, D=2
        """
        BS = ode_cfg.batch_size  # Total particles we want
        S = self.num_samples     # Particles per batch item
        assert BS % S == 0, f"batch_size ({BS}) must be divisible by num_samples ({S})"
        B = BS // S              # Number of batch items
        D = 2                    # 2D distributions in notebook
        K = self.num_steps
        dt = 1.0 / max(K - 1, 1)
        device = ode_cfg.device

        # Initialize positions and drift - sample S independent priors for each of B batch items
        pos_current_S_B_D = torch.stack(
            [x0_sampler(B).to(device) for _ in range(S)],
            dim=0,
        )  # (S, B, D)

        with torch.no_grad():
            t0_B_1 = torch.zeros(B, 1, device=device)
            drift_S_B_D = torch.stack(
                [self.model(torch.cat([pos_current_S_B_D[s], t0_B_1], dim=1)) for s in range(S)],
                dim=0,
            )

        # Track trajectory for notebook format - flatten to show all S*B particles
        trajectory = [pos_current_S_B_D.view(S * B, D)]  # Start with all particles [S*B, D]

        # Event-based FK reweighting (match FKSteeringSDE logic)
        log_weights_S_B = torch.zeros(S, B, device=device)
        log_weights_cum_S_B = torch.zeros(S, B, device=device)
        prev_U_S_B = None

        # Number of steering events before the final “closure” (used by harmonic_sum)
        if self.resample_freq is not None and self.resample_freq > 0:
            L = K // self.resample_freq
        else:
            L = 0
        if self.fk_potential_scheduler == "harmonic_sum" and L > 0:
            H_L = torch.sum(1.0 / torch.arange(1, L + 1, device=device, dtype=torch.float32))
        else:
            H_L = None

        ell = 0          # counts steering events
        t_steering = 0.0 # last steering time

        for k in range(K - 1):
            t = k / (K - 1)
            t_next = (k + 1) / (K - 1)

            # Pure deterministic flow update (no SDE terms)
            pos_new_S_B_D = pos_current_S_B_D + dt * drift_S_B_D

            # Gradient nudging against ∇U
            if self.gradient_factor is not None and self.gradient_factor > 0:
                pos_new_S_B_D = pos_new_S_B_D.clone()
                for s in range(S):
                    pos_slice = pos_new_S_B_D[s].detach().clone().requires_grad_(True)
                    U_B = self.potential_fn(pos_slice)
                    if U_B.requires_grad:
                        (grad,) = torch.autograd.grad(
                            U_B.sum(), pos_slice, retain_graph=False, create_graph=False
                        )
                        pos_new_S_B_D[s] = pos_slice - self.gradient_factor * dt * grad

            # Next drift and precompute one-shot potentials
            with torch.no_grad():
                tnext_B_1 = torch.full((B, 1), t_next, device=device)
                drift_S_B_D = torch.stack(
                    [self.model(torch.cat([pos_new_S_B_D[s], tnext_B_1], dim=1)) for s in range(S)],
                    dim=0,
                )
                # One-shot position for potential lookahead (not used at the last step)
                pos_one_shot_S_B_D = pos_new_S_B_D + (1.0 - t_next) * drift_S_B_D
                U_oneshot_S_B = torch.stack(
                    [self.potential_fn(pos_one_shot_S_B_D[s]) for s in range(S)],
                    dim=0,
                )

            # Periodic resampling (with final closure at terminal time)
            do_event = (
                self.resample_freq is not None
                and (
                    ((k + 1) % self.resample_freq == 0)  # regular event
                    or (k == K - 2)                      # force final closure at terminal time
                )
            )
            if do_event:
                is_final = (k == K - 2)

                # Choose which potential to use at event
                if is_final:
                    U_evt_S_B = torch.stack(
                        [self.potential_fn(pos_new_S_B_D[s]) for s in range(S)],
                        dim=0,
                    )
                else:
                    U_evt_S_B = U_oneshot_S_B

                # Update weights based on scheduler, event-based
                if self.fk_potential_scheduler == "sum":
                    if is_final:
                        # Close: total log-weight equals -T * U at terminal
                        log_weights_S_B = (
                            -self.fk_steering_temperature * U_evt_S_B
                            - log_weights_cum_S_B
                        )
                    else:
                        # Integrate potential over time since last event
                        log_weights_S_B = (
                            log_weights_S_B
                            - self.fk_steering_temperature * U_evt_S_B * (t - t_steering)
                        )
                    log_weights_cum_S_B += log_weights_S_B

                elif self.fk_potential_scheduler == "harmonic_sum":
                    if is_final:
                        log_weights_S_B = (
                            -self.fk_steering_temperature * U_evt_S_B
                            - log_weights_cum_S_B
                        )
                    else:
                        if H_L is not None and L > ell:
                            log_weights_S_B = log_weights_S_B - (
                                self.fk_steering_temperature * (U_evt_S_B / ((L - ell) * H_L))
                            )
                    log_weights_cum_S_B += log_weights_S_B

                elif self.fk_potential_scheduler == "difference":
                    if prev_U_S_B is None:
                        log_weights_S_B = -self.fk_steering_temperature * U_evt_S_B
                    else:
                        log_weights_S_B = self.fk_steering_temperature * (prev_U_S_B - U_evt_S_B)
                    prev_U_S_B = U_evt_S_B
                else:
                    raise ValueError(f"Unknown scheduler: {self.fk_potential_scheduler}")

                t_steering = t
                ell += 1

                with torch.no_grad():
                    stable = log_weights_S_B - log_weights_S_B.max(dim=0, keepdim=True)[0]
                    G_S_B = torch.exp(stable)
                    G_S_B = G_S_B / (G_S_B.sum(dim=0, keepdim=True) + 1e-12)

                    # Use proper resampling method
                    indices_B_S = resample(G_S_B.T, num_samples=S, method=self.resample_method)

                gather_idx_S_B_1 = indices_B_S.T.unsqueeze(-1)
                pos_new_S_B_D = torch.gather(
                    pos_new_S_B_D, dim=0, index=gather_idx_S_B_1.expand(-1, -1, D)
                )
                drift_S_B_D = torch.gather(
                    drift_S_B_D, dim=0, index=gather_idx_S_B_1.expand(-1, -1, D)
                )
                log_weights_S_B = torch.gather(log_weights_S_B, dim=0, index=indices_B_S.T)
                log_weights_cum_S_B = torch.gather(
                    log_weights_cum_S_B, dim=0, index=indices_B_S.T
                )

            pos_current_S_B_D = pos_new_S_B_D

            # Add to trajectory (all particles for visualization)
            trajectory.append(pos_current_S_B_D.view(S * B, D))

        # Convert to notebook format [T, S*B, D]
        return torch.stack(trajectory, dim=0)


def evaluate_fk(x0_sampler, x1_sampler, model, x1_dist, fk_params, ode_cfg):
    """
    FK evaluation function that matches notebook's evaluate() signature.
    
    This is the main entry point for running FK steering in the synthetic experiments,
    providing compatibility with the existing guided flow notebook interface.
    
    Args:
        x0_sampler: Initial distribution sampler function
        x1_sampler: Target distribution sampler (unused, kept for API compatibility)
        model: Trained flow model
        x1_dist: Target distribution object with get_J(x) method
        fk_params: Dictionary of FK parameters (num_samples, temperature, etc.)
        ode_cfg: ODE configuration object
        
    Returns:
        torch.Tensor: Generated trajectory [T, BS, D] for visualization
    """
    
    fk_wrapper = FKSteeringWrapper(
        model=model,
        x1_dist=x1_dist,
        num_samples=fk_params.get('num_samples', 8),
        num_steps=ode_cfg.num_steps,
        fk_steering_temperature=fk_params.get('fk_steering_temperature', 1.0),
        fk_potential_scheduler=fk_params.get('fk_potential_scheduler', 'sum'),
        gradient_factor=fk_params.get('gradient_factor', None),
        resample_method=fk_params.get('resample_method', 'stratified'),
        resample_freq=fk_params.get('resample_freq', 2),
        scale=fk_params.get('scale', 1.0),
    )
    
    return fk_wrapper.generate_trajectory(x0_sampler, ode_cfg)


# Extension for GuideFnConfig to support FK parameters
def create_fk_guide_cfg(dist_pair, scale=1.0, num_samples=8, 
                       fk_steering_temperature=1.0, fk_potential_scheduler='sum',
                       gradient_factor=None, resample_method='residual', 
                       resample_freq=2, ode_cfg=None, **kwargs):
    """Create a GuideFnConfig for FK steering"""
    from guided_flow.config.sampling import GuideFnConfig, ODEConfig
    
    if ode_cfg is None:
        ode_cfg = ODEConfig(t_end=1.0, num_steps=100)
    
    # Create base config
    cfg = GuideFnConfig(
        dist_pair=dist_pair,
        guide_type='fk',
        scale=scale,
        ode_cfg=ode_cfg,
        **kwargs
    )
    
    # Add FK-specific parameters
    cfg.fk_params = {
        'num_samples': num_samples,
        'fk_steering_temperature': fk_steering_temperature,
        'fk_potential_scheduler': fk_potential_scheduler,
        'gradient_factor': gradient_factor,
        'resample_method': resample_method,
        'resample_freq': resample_freq,
        'scale': scale,
    }
    
    return cfg