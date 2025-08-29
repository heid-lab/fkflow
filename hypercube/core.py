import os
import torch
from tqdm import tqdm
from itertools import product
from math import log
from typing import Tuple, Optional, Union

from torchcfm.conditional_flow_matching import SchrodingerBridgeConditionalFlowMatcher
from torchcfm.models import MLP
import random


def set_random_seed(seed):
    """Set random seed for reproducibility across all libraries"""
    if seed is not None:
        random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            torch.mps.manual_seed(seed)


def get_device(device="auto"):
    """Get the appropriate device (cuda, mps, or cpu)"""
    if device == "auto":
        if torch.cuda.is_available():
            return "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
        else:
            return "cpu"
    return device


def sample_uniform_hypercube(
    n_samples, dim, cube_range=(-1, 1), device="cpu", generator=None
):
    """Sample uniformly from [-r, r]^d hypercube"""
    low, high = cube_range
    return (
        torch.rand(n_samples, dim, device=device, generator=generator) * (high - low)
        + low
    )


def sample_corner_gaussians(
    n_samples, dim, cube_range=(-2, 2), std=0.2, device="cpu", generator=None
):
    """Sample from equal mixture of Gaussians at hypercube corners"""
    low, high = cube_range

    corners = list(product([low, high], repeat=dim))
    n_corners = len(corners)

    corner_indices = torch.randint(0, n_corners, (n_samples,), generator=generator)

    corner_centers = torch.tensor(corners, device=device, dtype=torch.float32)
    centers = corner_centers[corner_indices]

    noise = torch.randn(n_samples, dim, device=device, generator=generator) * std
    return centers + noise


def indicator_potential(x, dim, threshold=0.0, log_weight=None):
    """
    Indicator potential function for steering toward target region [threshold, ∞)^d.
    
    Returns 0 for samples in the target region and log_weight for samples outside.
    This creates a sharp preference for the target region.
    
    Args:
        x: Input samples of shape (N, D)
        dim: Problem dimension (for computing default log_weight)
        threshold: Target region boundary (default: 0.0)
        log_weight: Log potential for samples outside target (default: log(2^dim - 1))
        
    Returns:
        torch.Tensor: Potential values of shape (N,)
    """
    in_target = torch.all(x >= threshold, dim=1)
    # Allow equal probabilities by passing None as arg (50% target peak, 50% spread over the others)
    if log_weight is None:
        n = 2**dim
        log_weight = log(n - 1)
    return torch.where(in_target, 0.0, log_weight)


def distance_potential(x, dim, threshold=0.0, weight_scale=1.0, exponent=1.0):
    """
    Distance-based potential function for steering toward target region [threshold, ∞)^d.
    
    Returns potential proportional to distance from the target region, raised to
    a given exponent. This creates a smooth gradient toward the target region.
    
    Args:
        x: Input samples of shape (N, D)
        dim: Problem dimension (unused, kept for API consistency)
        threshold: Target region boundary (default: 0.0)
        weight_scale: Scaling factor for the potential (default: 1.0)
        exponent: Power to raise distances to (1.0=linear, 2.0=quadratic, etc.)
        
    Returns:
        torch.Tensor: Potential values of shape (N,)
    """
    distances_per_dim = torch.clamp(threshold - x, min=0.0)
    powered_distances = torch.pow(distances_per_dim, exponent)
    total_distance = torch.sum(powered_distances, dim=1)
    return weight_scale * total_distance


def compute_success_rate(samples, threshold=0.0):
    """Compute percentage of samples in [threshold, ∞)^d"""
    in_target = torch.all(samples >= threshold, dim=1)
    return in_target.float().mean().item() * 100


def sample_corner_gaussians_biased(
    n_samples: int,
    dim: int,
    cube_range: Tuple[float, float] = (-2.0, 2.0),
    std: float = 0.2,
    positive_prob: float = 0.5,
    device: Union[str, torch.device] = "cpu",
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    """Sample from a corner-Gaussian mixture with configurable weight on positive corner"""
    low, high = cube_range

    corners = list(product([low, high], repeat=dim))
    n_corners = len(corners)
    corner_centers = torch.tensor(corners, device=device, dtype=torch.float32)

    positive_corner = torch.full((dim,), high, device=device, dtype=torch.float32)
    pos_idx = torch.argmin(
        torch.sum(torch.abs(corner_centers - positive_corner), dim=1)
    ).item()

    weights = torch.full(
        (n_corners,), (1.0 - positive_prob) / (n_corners - 1), device=device
    )
    weights[pos_idx] = positive_prob

    idx = torch.multinomial(
        weights, num_samples=n_samples, replacement=True, generator=generator
    )
    centers = corner_centers[idx]

    noise = torch.randn(n_samples, dim, device=device, generator=generator) * std
    return centers + noise


def sample_target_region_gaussians(
    n_samples: int,
    dim: int,
    threshold: float = 0.0,
    cube_range: Tuple[float, float] = (-2.0, 2.0),
    std: float = 0.2,
    device: Union[str, torch.device] = "cpu",
    generator: Optional[torch.Generator] = None,
    max_attempts: int = 10000,
) -> torch.Tensor:
    """Sample from the single target region Gaussian at (high, high, ..., high)"""
    low, high = cube_range

    target_center = torch.full((dim,), high, device=device, dtype=torch.float32)

    # Sample from Gaussian centered at the target corner
    centers = target_center.unsqueeze(0).expand(n_samples, -1)
    noise = torch.randn(n_samples, dim, device=device, generator=generator) * std
    samples = centers + noise

    return samples


@torch.no_grad()
def wasserstein_2_to_target(
    samples: torch.Tensor,
    threshold: float = 0.0,
    cube_range: Tuple[float, float] = (-2.0, 2.0),
    std: float = 0.2,
    n_projections: int = 256,
    seed: Optional[int] = 3,
    squared: bool = False,
) -> float:
    """Approximate 2-Wasserstein distance to target region Gaussians using sliced Wasserstein"""
    if samples.ndim != 2:
        raise ValueError("samples must be of shape (N, D)")
    N, D = samples.shape
    device = samples.device

    gen = None
    if seed is not None:
        gen = torch.Generator(device=device)
        gen.manual_seed(seed)

    target = sample_target_region_gaussians(
        n_samples=N,
        dim=D,
        threshold=threshold,
        cube_range=cube_range,
        std=std,
        device=device,
        generator=gen,
    )

    dirs = torch.randn(n_projections, D, device=device, generator=gen)
    dirs = dirs / dirs.norm(dim=1, keepdim=True).clamp_min(1e-12)

    proj_samples = samples @ dirs.T
    proj_target = target @ dirs.T

    proj_samples_sorted, _ = torch.sort(proj_samples, dim=0)
    proj_target_sorted, _ = torch.sort(proj_target, dim=0)

    w2_sq_per_proj = ((proj_samples_sorted - proj_target_sorted) ** 2).mean(dim=0)
    w2_sq = w2_sq_per_proj.mean()

    return float(w2_sq.item() if squared else torch.sqrt(w2_sq).item())


def stratified_multinomial(
    input: torch.Tensor,
    num_samples: int,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    """Stratified resampling with same interface as torch.multinomial"""
    if num_samples < 0:
        raise ValueError("num_samples must be non-negative")

    device = input.device
    last_dim = input.shape[-1]
    flat = input.reshape(-1, last_dim).to(dtype=torch.float32)

    if (flat < 0).any():
        raise RuntimeError("probabilities contain negative values")

    row_sums = flat.sum(dim=-1)
    if (row_sums <= 0).any():
        raise RuntimeError(
            "invalid multinomial distribution (sum of probabilities <= 0)"
        )

    probs = flat / row_sums.clamp_min(1e-20).unsqueeze(-1)
    cdf = probs.cumsum(dim=-1).clamp(max=1.0)

    B = flat.shape[0]

    if num_samples == 0:
        out = torch.empty((B, 0), dtype=torch.long, device=device)
        return out.reshape(*input.shape[:-1], 0)

    u0 = torch.rand(B, 1, dtype=torch.float32, device=device, generator=generator)
    u = (
        torch.arange(num_samples, device=device, dtype=torch.float32) + u0
    ) / num_samples

    idx = torch.searchsorted(cdf, u, right=False).to(dtype=torch.long)

    return idx.reshape(*input.shape[:-1], num_samples)


class FKSteeringSDE:
    """
    Feynman-Kac steering for Stochastic Differential Equations.
    
    This class implements FK steering to guide samples from an initial distribution
    toward regions with low potential energy. It uses a particle filter approach
    with resampling based on potential-weighted importance weights.
    
    Args:
        flow_model: Neural network that predicts the velocity field v_t(x,t)
        score_model: Neural network that predicts the score function ∇log p_t(x)
        potential_fn: Function that computes potential U(x) for steering
        sigma: Noise level for the SDE
        num_samples: Number of particles for the particle filter
        fk_steering_temperature: Temperature parameter β for FK steering (higher = more aggressive)
        fk_potential_scheduler: How to schedule potentials ("sum", "harmonic_sum", "difference")
        resample_freq: How often to resample particles (lower = more FK-like, higher = more IS-like)
        generator: Random number generator for reproducibility
    """

    def __init__(
        self,
        flow_model,
        score_model,
        potential_fn,
        sigma,
        num_samples=8,
        fk_steering_temperature=1.0,
        fk_potential_scheduler="sum",
        resample_freq=2,
        generator=None,
    ):
        self.flow_model = flow_model
        self.score_model = score_model
        self.potential_fn = potential_fn
        self.sigma = sigma
        self.num_samples = num_samples
        self.fk_steering_temperature = fk_steering_temperature
        self.fk_potential_scheduler = fk_potential_scheduler
        self.resample_freq = resample_freq
        self.generator = generator

    def generate_samples(
        self,
        x0_sampler,
        num_steps=100,
        batch_size=256,
        device=None,
        progress_callback=None,
    ):
        """
        Generate samples using FK steering.
        
        Args:
            x0_sampler: Function that samples initial conditions x0_sampler(batch_size, device)
            num_steps: Number of SDE integration steps
            batch_size: Total batch size (must be divisible by num_samples)
            device: Device to run on (auto-detected if None)
            progress_callback: Optional callback function called at each step
            
        Returns:
            torch.Tensor: Final samples of shape (batch_size, dim)
        """
        if device is None:
            device = next(self.flow_model.parameters()).device

        S = self.num_samples
        assert batch_size % S == 0, "batch_size must be divisible by num_samples"
        B = batch_size // S
        K = num_steps
        L = K // self.resample_freq if self.resample_freq is not None else 0

        dt = 1.0 / (K - 1)

        pos_current_S_B_D = torch.stack(
            [x0_sampler(B, device=device) for _ in range(S)], dim=0
        )

        with torch.no_grad():
            t0_B_1 = torch.zeros(B, 1, device=device)
            drift_S_B_D = torch.stack(
                [
                    self.flow_model(torch.cat([pos_current_S_B_D[s], t0_B_1], dim=1))
                    for s in range(S)
                ],
                dim=0,
            )

        log_weights_S_B = torch.zeros(S, B, device=device)
        log_weights_cum_S_B = torch.zeros(S, B, device=device)

        prev_U_S_B = None

        if (
            self.fk_potential_scheduler == "harmonic_sum"
            and self.resample_freq is not None
        ):
            H_L = torch.sum(
                1.0 / torch.arange(1, L + 1, device=device, dtype=torch.float32)
            )
        else:
            H_L = None
        ell = 0
        t_steering = 0.0
        for k in range(K - 1):
            t = k / (K - 1)
            t_next = (k + 1) / (K - 1)
            sigma_t = self.sigma

            # Update progress if callback provided
            if progress_callback is not None:
                progress_callback()

            with torch.no_grad():
                t_tensor = torch.full((S * B, 1), t, device=device)
                pos_flat = pos_current_S_B_D.view(S * B, -1)
                score_t = self.score_model(torch.cat([pos_flat, t_tensor], dim=1))
                score_t_S_B_D = score_t.view(S, B, -1)

                sde_drift_S_B_D = drift_S_B_D + (sigma_t**2 / 2) * score_t_S_B_D
                diffusion = (
                    sigma_t
                    * (dt**0.5)
                    * torch.randn(
                        pos_current_S_B_D.shape,
                        device=pos_current_S_B_D.device,
                        generator=self.generator,
                    )
                )
                pos_new_S_B_D = pos_current_S_B_D + dt * sde_drift_S_B_D + diffusion

                if k != K - 2:
                    tnext_B_1 = torch.full((B, 1), t_next, device=device)
                    drift_S_B_D = torch.stack(
                        [
                            self.flow_model(
                                torch.cat([pos_new_S_B_D[s], tnext_B_1], dim=1)
                            )
                            for s in range(S)
                        ],
                        dim=0,
                    )

            if (
                self.resample_freq is not None and (k + 1) % self.resample_freq == 0
            ) or (k == K - 2 and self.resample_freq is not None):
                if k != K - 2:
                    pos_one_shot_S_B_D = pos_new_S_B_D + (1.0 - t_next) * drift_S_B_D
                    U_S_B = torch.stack(
                        [self.potential_fn(pos_one_shot_S_B_D[s]) for s in range(S)],
                        dim=0,
                    )
                else:
                    U_S_B = torch.stack(
                        [self.potential_fn(pos_new_S_B_D[s]) for s in range(S)], dim=0
                    )

                if self.fk_potential_scheduler == "sum":
                    if k == K - 2:
                        log_weights_S_B = (
                            -self.fk_steering_temperature * U_S_B - log_weights_cum_S_B
                        )
                    else:
                        log_weights_S_B = (
                            log_weights_S_B
                            - self.fk_steering_temperature * U_S_B * (t - t_steering)
                        )

                    log_weights_cum_S_B += log_weights_S_B

                elif self.fk_potential_scheduler == "harmonic_sum":
                    if k == K - 2:
                        log_weights_S_B = (
                            -self.fk_steering_temperature * U_S_B - log_weights_cum_S_B
                        )
                    else:
                        if H_L is not None:
                            log_weights_S_B = (
                                log_weights_S_B
                                - self.fk_steering_temperature
                                * (U_S_B / ((L - ell) * H_L))
                            )
                    log_weights_cum_S_B += log_weights_S_B

                elif self.fk_potential_scheduler == "difference":
                    if prev_U_S_B is None:
                        log_weights_S_B = -self.fk_steering_temperature * U_S_B
                    else:
                        log_weights_S_B = self.fk_steering_temperature * (
                            prev_U_S_B - U_S_B
                        )

                    prev_U_S_B = U_S_B
                t_steering = t
                ell += 1

                with torch.no_grad():
                    stable = (
                        log_weights_S_B - log_weights_S_B.max(dim=0, keepdim=True)[0]
                    )
                    G_S_B = torch.exp(stable)
                    G_S_B = G_S_B / (G_S_B.sum(dim=0, keepdim=True) + 1e-12)
                    indices_B_S = stratified_multinomial(
                        G_S_B.T, num_samples=S, generator=self.generator
                    )

                gather_idx_S_B_1 = indices_B_S.T.unsqueeze(-1)
                pos_new_S_B_D = torch.gather(
                    pos_new_S_B_D,
                    dim=0,
                    index=gather_idx_S_B_1.expand(-1, -1, pos_new_S_B_D.size(-1)),
                )
                drift_S_B_D = torch.gather(
                    drift_S_B_D,
                    dim=0,
                    index=gather_idx_S_B_1.expand(-1, -1, drift_S_B_D.size(-1)),
                )

                log_weights_S_B = torch.gather(
                    log_weights_S_B, dim=0, index=indices_B_S.T
                )
                log_weights_cum_S_B = torch.gather(
                    log_weights_cum_S_B, dim=0, index=indices_B_S.T
                )

            pos_current_S_B_D = pos_new_S_B_D

        return pos_current_S_B_D.view(S * B, -1)


def train_sb_models(
    dim,
    num_epochs=5000,
    batch_size=256,
    lr=0.01,
    sigma=0.5,
    save_models=True,
    device="auto",
):
    """
    Train Schrödinger Bridge flow and score models for given dimension.
    
    Trains both a velocity field model v_t(x,t) and score model ∇log p_t(x,t)
    using the Schrödinger Bridge objective. The initial distribution is uniform
    hypercube and target distribution is corner Gaussians.
    
    Args:
        dim: Dimension of the problem
        num_epochs: Number of training epochs
        batch_size: Training batch size
        lr: Learning rate
        sigma: Noise level for Schrödinger Bridge
        save_models: Whether to save trained models to disk
        device: Device to train on ("auto", "cuda", "mps", "cpu")
        
    Returns:
        tuple: (flow_model, score_model) trained models
    """

    device = get_device(device)

    print(f"Training dimension {dim} ({num_epochs} epochs)...", end=" ", flush=True)

    model = MLP(dim=dim, time_varying=True, w=128).to(device)
    score_model = MLP(dim=dim, time_varying=True, w=128).to(device)
    optimizer = torch.optim.Adam(
        list(model.parameters()) + list(score_model.parameters()), lr
    )
    FM = SchrodingerBridgeConditionalFlowMatcher(sigma=sigma)

    model.train()
    score_model.train()

    for epoch in tqdm(range(num_epochs), desc=f"Dim {dim}", leave=False, ncols=60):
        optimizer.zero_grad()

        x0 = sample_uniform_hypercube(batch_size, dim, device=device)
        x1 = sample_corner_gaussians(batch_size, dim, device=device)

        t, xt, ut, eps = FM.sample_location_and_conditional_flow(
            x0, x1, return_noise=True
        )
        lambda_t = FM.compute_lambda(t)

        vt = model(torch.cat([xt, t[:, None]], dim=-1))
        st = score_model(torch.cat([xt, t[:, None]], dim=-1))

        flow_loss = torch.mean((vt - ut) ** 2)
        score_loss = torch.mean((lambda_t[:, None] * st + eps) ** 2)
        total_loss = flow_loss + score_loss

        total_loss.backward()
        optimizer.step()

    model.eval()
    score_model.eval()

    if save_models:
        os.makedirs("trained_models", exist_ok=True)
        torch.save(
            {
                "flow_model_state": model.state_dict(),
                "score_model_state": score_model.state_dict(),
                "dim": dim,
                "sigma": sigma,
            },
            f"trained_models/sb_models_dim{dim}.pt",
        )
        print("✓")
    else:
        print("✓")

    return model, score_model


def load_sb_models(dim, sigma=1.0, device="auto"):
    """
    Load pre-trained Schrödinger Bridge models for given dimension.
    
    Args:
        dim: Dimension to load models for
        sigma: Noise level (for compatibility, not currently used in loading)
        device: Device to load models on ("auto", "cuda", "mps", "cpu")
        
    Returns:
        tuple: (flow_model, score_model) loaded in eval mode
        
    Raises:
        FileNotFoundError: If no trained model exists for the given dimension
    """

    # Auto-detect device if needed
    device = get_device(device)

    model_path = f"trained_models/sb_models_dim{dim}.pt"

    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"No trained models found for dimension {dim} at {model_path}"
        )

    model = MLP(dim=dim, time_varying=True, w=128).to(device)
    score_model = MLP(dim=dim, time_varying=True, w=128).to(device)

    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint["flow_model_state"])
    score_model.load_state_dict(checkpoint["score_model_state"])

    model.eval()
    score_model.eval()
    return model, score_model
