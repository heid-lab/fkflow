import math
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import lightning.pytorch as pl
import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch_geometric.data import Batch, Data
from torchdiffeq import odeint

from flow_matching.resampling import Resampler
from flow_matching.utils import (
    chiral_potential,
    get_min_dmae_match_torch,
    get_shortest_path_batched_x_1,
    get_substruct_matches,
    rmsd_loss,
)
from gotennet.models.components.outputs import Atomwise3DOut
from gotennet.models.representation.gotennet import GotenNet


class GraphTimeMLP(nn.Module):
    """
    Multi-layer perceptron that takes per-graph time values and outputs per-graph coefficients.

    Used for time-dependent conditioning in flow matching models.

    Args:
        hidden_dim (int): Hidden dimension size. Defaults to 32.
        num_layers (int): Number of hidden layers. Defaults to 2.
    """

    def __init__(self, hidden_dim=32, num_layers=2):
        super().__init__()
        layers = []
        layers.append(nn.Linear(1, hidden_dim))
        layers.append(nn.ReLU())
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(hidden_dim, 1))
        self.mlp = nn.Sequential(*layers)

    def forward(self, t_G):
        """
        Forward pass through the MLP.

        Args:
            t_G (Tensor): Per-graph time values, shape (num_graphs,) or (num_graphs, 1)

        Returns:
            Tensor: Per-graph coefficients, shape (num_graphs,)
        """
        if t_G.dim() == 1:
            t_G = t_G.unsqueeze(-1)
        coeffs = self.mlp(t_G)  # (num_graphs, 1)
        return coeffs.squeeze(-1)  # (num_graphs,)


def expand_coeffs_to_nodes(coeffs, batch):
    """
    Expand per-graph coefficients to per-node coefficients.

    Maps graph-level values to their corresponding nodes using the batch index tensor.

    Args:
        coeffs (Tensor): Per-graph coefficients, shape (num_graphs,)
        batch (Tensor): Batch index tensor mapping each node to its graph, shape (num_nodes,)

    Returns:
        Tensor: Per-node coefficients, shape (num_nodes,)
    """
    return coeffs[batch]


class FlowModule(pl.LightningModule):
    """
    Flow matching with Feynman-Kac steering for reaction transition state generation.

    This module implements continuous normalizing flows with optional FK steering for generating
    molecular conformations, particularly focused on chemical reaction transition states with
    chirality awareness.

    Args:
        representation (GotenNet): The neural network representation model
        lr (float): Learning rate for optimization. Defaults to 5e-4.
        lr_decay (float): Learning rate decay factor. Defaults to 0.5.
        lr_patience (int): Patience for learning rate scheduler. Defaults to 100.
        lr_minlr (float): Minimum learning rate. Defaults to 1e-6.
        lr_monitor (str): Metric to monitor for LR scheduling.
            Defaults to "validation/ema_val_loss".
        chain_scheduler (Optional[Callable]): Unused; kept for config compatibility.
        weight_decay (float): Weight decay for optimizer. Defaults to 0.01.
        num_steps (int): Number of integration steps for sampling. Defaults to 10.
        num_samples (int): Number of samples for FK steering. Defaults to 1.
        seed (int): Random seed. Defaults to 1.
        filter_clash (bool): Unused; kept for config compatibility.
        pos_fit_idx (int): Position index to fit. Defaults to 1.
        ema_decay (float): Unused; kept for config compatibility.
        dataset_meta (Optional[Dict]): Unused; kept for config compatibility.
        output (Optional[Dict]): Output layer configuration. Defaults to None.
        scheduler (Optional[Callable]): Learning rate scheduler. Defaults to None.
        save_predictions (Optional[bool]): Unused; kept for config compatibility.
        input_contribution (float): Unused; kept for config compatibility.
        task_config (Optional[Dict]): Task configuration. Defaults to None.
        lr_warmup_steps (int): Learning rate warmup steps. Defaults to 0.
        schedule_free (bool): Whether to use schedule-free optimization. Defaults to False.
        use_ema (bool): Whether to use exponential moving average. Defaults to False.
        inference_sampling_method (str): "fk_steering" (particles resampled with the FK
                                       weights) or "median" (num_samples independent ODE
                                       samples, no steering). In both cases the reported
                                       geometry is the final sample closest to the
                                       coordinate-wise median of all samples.
                                       Defaults to "fk_steering".
        fk_potential_fns (Optional[Sequence[Callable]]): FK potentials, each called as
                                    fn(pos_N_3, batch, device) -> per-molecule U;
                                    U is their sum. Defaults to (chiral_potential,).
        fk_potential_scheduler (str): FK potential schedule over the resampling events
                                    j = 0..L-1, with U_j the potential at the one-shot
                                    estimate x + (1 - t) v (at the last step, the
                                    sample itself) and lambda = fk_steering_temperature:
                                    "difference"    log G_j = -lambda (U_j - U_{j-1}), U_{-1} = 0,
                                    "sum"           log G_j = -lambda sum_{i<=j} U_i dt_i,
                                                    dt_i the time since the previous event,
                                    "harmonic_sum"  log G_j = -lambda sum_{i<=j} U_i / ((L-i) H_L),
                                    "max"           log G_j = -lambda max_{i<=j} U_i.
                                    Except for "difference", the last event uses
                                    log G = -lambda U - (sum of the earlier log G), so the
                                    product of all G is exp(-lambda U(x_1)). Defaults to "sum".
        steering_base_variance (float): Noise scale sigma of the SDE, a standard deviation
                                    despite the name (see noise_schedule); 0 gives the
                                    plain ODE. Defaults to 1.0.
        fk_steering_temperature (float): lambda, the strength of the tilt
                                    p_1(x) exp(-lambda U(x)). Defaults to 2e0.
        resample_method (str): Resampling method ("multinomial", "stratified", "residual",
                             "systematic"). Defaults to "stratified".
        resample_freq (int): Frequency of resampling in FK loop, counted in
                             steps inside the steering window. Defaults to 2.
        steering_t_start (float): Resampling events happen only after steps that
                                  end at t >= steering_t_start; the event at the
                                  last step always happens. Defaults to 0.0.
        steering_t_end (Optional[float]): No resampling events after steps that end
                                          at t > steering_t_end, except the last
                                          step. With noise_t_end, events can stop
                                          where the noise stops, since resampling
                                          deterministic trajectories only copies
                                          them. Defaults to None (events up to t = 1).
        steering_final_event (bool): Resample after the last step. With False and
                                     steering_t_end < 1, the particles keep their
                                     diversity to the end and the median
                                     aggregation sees all of them. Defaults to True.
        noise_schedule (str): Noise level sigma_t of the SDE, with
                              sigma = steering_base_variance and t0 = noise_t_start:
                              "linear"       sigma * (1 - t) everywhere,
                              "linear_window" sigma * (1 - t) for t >= t0, else 0,
                              "sqrt_window"  sigma * sqrt((1 - t) / (1 - t0)) for
                                             t >= t0, else 0.
                              Defaults to "linear".
        noise_t_start (Optional[float]): Start of the noise window. Defaults to
                                         steering_t_start.
        noise_t_end (Optional[float]): No noise on steps starting at t >= noise_t_end,
                                       so the end of the trajectory is a plain ODE and
                                       no late noise is left in the final geometry.
                                       Applies to every noise_schedule. Defaults to
                                       None (noise up to t = 1).
        window_num_steps (Optional[int]): If set, the time grid keeps its
                                          num_steps - 1 steps in total but puts
                                          this many of them on [steering_t_start, 1]
                                          and the rest on [0, steering_t_start].
                                          The median method uses the same grid.
                                          Defaults to None (uniform grid).
        save_final_particles (bool): FK steering only. Store on each result the S
                                     final particles before the last resampling,
                                     aligned to the GT (particles_pre, S x Nm x 3),
                                     their summed potential (particle_potential_pre, S)
                                     and the particles after it (particles_post).
                                     Defaults to False.
        **kwargs: Additional keyword arguments.
    """

    def __init__(
        self,
        representation: GotenNet,
        lr: float = 5e-4,
        lr_decay: float = 0.5,
        lr_patience: int = 100,
        lr_minlr: float = 1e-6,
        lr_monitor: str = "validation/ema_val_loss",
        chain_scheduler: Optional[Callable] = None,
        weight_decay: float = 0.01,
        num_steps: int = 10,
        num_samples: int = 1,
        seed: int = 1,
        filter_clash: bool = False,
        pos_fit_idx: int = 1,
        ema_decay: float = 0.9,
        dataset_meta: Optional[Dict[str, Dict[int, Tensor]]] = None,
        output: Optional[Dict] = None,
        scheduler: Optional[Callable] = None,
        save_predictions: Optional[bool] = None,
        input_contribution: float = 1,
        task_config: Optional[Dict] = None,
        lr_warmup_steps: int = 0,
        schedule_free: bool = False,
        use_ema: bool = False,
        inference_sampling_method: str = "fk_steering",  # "fk_steering", "median"
        fk_potential_fns: Optional[Sequence[Callable]] = (chiral_potential,),
        fk_potential_scheduler: str = "sum",  # "difference", "sum", "max", "harmonic_sum"
        steering_base_variance: float = 1.0,
        fk_steering_temperature: float = 2e0,
        resample_method: str = "stratified",  # multinomial, stratified, residual, systematic
        resample_freq: int = 2,  # Frequency of resampling in the Feynman-Kac loop
        steering_t_start: float = 0.0,
        steering_t_end: Optional[float] = None,
        steering_final_event: bool = True,
        noise_schedule: str = "linear",  # "linear", "linear_window", "sqrt_window"
        noise_t_start: Optional[float] = None,
        noise_t_end: Optional[float] = None,
        window_num_steps: Optional[int] = None,
        save_final_particles: bool = False,
        **kwargs,
    ):
        super().__init__()
        self.representation = representation
        if output is not None:
            self.atomwise_3D_out_layer = Atomwise3DOut(
                n_in=representation.hidden_dim,
                n_hidden=output["n_hidden"],
                activation=F.silu,
            )
        else:
            self.atomwise_3D_out_layer = None

        self.lr = lr
        self.lr_decay = lr_decay
        self.lr_patience = lr_patience
        self.lr_monitor = lr_monitor
        self.weight_decay = weight_decay
        self.dataset_meta = dataset_meta

        self.num_steps = num_steps
        self.num_samples = num_samples

        self.use_ema = use_ema
        self.schedule_free = schedule_free
        self.lr_warmup_steps = lr_warmup_steps
        self.lr_minlr = lr_minlr
        self.chain_scheduler = chain_scheduler
        self.input_contribution = input_contribution
        self.save_predictions = save_predictions

        self.seed = seed
        self.filter_clash = filter_clash
        self.pos_fit_idx = pos_fit_idx

        self.scheduler = scheduler

        # Instantiated potentials are objects, which the loggers cannot write to
        # YAML; Hydra's own config.yaml of the run records them.
        self.save_hyperparameters(ignore=["fk_potential_fns"])
        # save results in test_step
        self.results_R = []
        self._fk_diag_Nm = None

        self.inference_sampling_method = inference_sampling_method

        # list() because Hydra passes instantiated potentials in a ListConfig
        self.fk_potential_fns = list(fk_potential_fns) if fk_potential_fns is not None else []
        valid_schedulers = ["difference", "sum", "harmonic_sum", "max"]
        if fk_potential_scheduler not in valid_schedulers:
            raise ValueError(f"fk_potential_scheduler must be one of {valid_schedulers}")
        self.fk_potential_scheduler = fk_potential_scheduler
        self.steering_base_variance = steering_base_variance
        self.fk_steering_temperature = fk_steering_temperature
        self.resample_freq = resample_freq

        valid_noise_schedules = ["linear", "linear_window", "sqrt_window"]
        if noise_schedule not in valid_noise_schedules:
            raise ValueError(f"noise_schedule must be one of {valid_noise_schedules}")
        if not 0.0 <= steering_t_start < 1.0:
            raise ValueError("steering_t_start must be in [0, 1)")
        self.steering_t_start = steering_t_start
        if steering_t_end is not None and not steering_t_start < steering_t_end <= 1.0:
            raise ValueError("steering_t_end must be in (steering_t_start, 1]")
        self.steering_t_end = steering_t_end
        self.steering_final_event = steering_final_event
        self.noise_schedule = noise_schedule
        self.noise_t_start = steering_t_start if noise_t_start is None else noise_t_start
        if not 0.0 <= self.noise_t_start < 1.0:
            raise ValueError("noise_t_start must be in [0, 1)")
        if noise_t_end is not None and not 0.0 < noise_t_end <= 1.0:
            raise ValueError("noise_t_end must be in (0, 1]")
        self.noise_t_end = noise_t_end
        if window_num_steps is not None and not 0 < window_num_steps < num_steps - 1:
            raise ValueError("window_num_steps must be between 1 and num_steps - 2")
        self.window_num_steps = window_num_steps
        self.save_final_particles = save_final_particles
        self._final_pre = None

        self.resampler = Resampler(resample_method=resample_method)

    def _time_grid(self) -> Tensor:
        """Integration grid on [0, 1], shared by the FK loop and the median method."""
        if self.window_num_steps is None:
            return torch.linspace(0, 1, steps=self.num_steps, device=self.device)
        # num_steps - 1 steps in total, window_num_steps of them in the window
        n_pre = self.num_steps - 1 - self.window_num_steps
        pre = torch.linspace(0, self.steering_t_start, steps=n_pre + 1, device=self.device)
        post = torch.linspace(
            self.steering_t_start, 1, steps=self.window_num_steps + 1, device=self.device
        )
        return torch.cat([pre, post[1:]])

    def _event_steps(self, t_T: Tensor) -> set:
        """Steps k after which a resampling event happens.

        Every resample_freq-th step that ends inside the steering window
        [steering_t_start, steering_t_end], plus the last step (unless
        steering_final_event is False). With
        steering_t_start = 0 and no steering_t_end this is (k + 1) % resample_freq == 0.
        """
        last = len(t_T) - 2
        if self.resample_freq is None:
            return set()
        events, n_in_window = set(), 0
        for k in range(len(t_T) - 1):
            t_end = float(t_T[k + 1])
            if self.steering_t_end is not None and t_end > self.steering_t_end + 1e-6:
                continue
            if t_end >= self.steering_t_start - 1e-6:
                n_in_window += 1
                if n_in_window % self.resample_freq == 0:
                    events.add(k)
        if self.steering_final_event:
            events.add(last)
        return events

    def _sigma_t(self, t) -> float:
        """Noise level of the SDE step at time t (see noise_schedule)."""
        t = float(t)
        sigma = self.steering_base_variance
        if self.noise_t_end is not None and t >= self.noise_t_end - 1e-6:
            return 0.0
        if self.noise_schedule == "linear":
            return sigma * (1 - t)
        if t < self.noise_t_start - 1e-6:
            return 0.0
        if self.noise_schedule == "linear_window":
            return sigma * (1 - t)
        return sigma * math.sqrt(max(1 - t, 0.0) / (1 - self.noise_t_start))

    def _sample_prior(self, x_1_N_3, batch):
        # Default to gaussian if no method matches
        x_0_N_3 = torch.randn_like(x_1_N_3, device=self.device)
        return x_0_N_3

    def get_perturbed_flow_point_and_time(self, batch: Data):
        """
        Generate perturbed flow points for training by interpolating between prior and target.

        This method implements the flow matching training procedure by sampling random times
        and interpolating between a prior distribution (x_0) and target conformations (x_1).

        Args:
            batch (Data): PyTorch Geometric batch containing molecular data

        Returns:
            tuple: A tuple containing:
                - x_t_N_3 (Tensor): Interpolated positions at time t, shape (N, 3)
                - dx_dt_N_3 (Tensor): Flow field (velocity), shape (N, 3)
                - t_G (Tensor): Per-graph time values, shape (num_graphs, 1)
        """
        x_1_N_3 = batch.pos[:, self.pos_fit_idx, :]
        x_0_N_3 = self._sample_prior(x_1_N_3, batch)

        t_G = torch.rand(batch.num_graphs, 1, device=self.device)
        t_N = t_G[batch.batch]

        x_1_aligned_N_3 = get_shortest_path_batched_x_1(x_0_N_3, x_1_N_3, batch)

        x_t_N_3 = (1 - t_N) * x_0_N_3 + t_N * x_1_aligned_N_3
        dx_dt_N_3 = x_1_aligned_N_3 - x_0_N_3

        return x_t_N_3, dx_dt_N_3, t_G

    def model_output(self, x_t_N_3, batch: Data, t_G: Tensor) -> Tensor:
        """
        Compute model output (predicted flow field) given positions, time, and batch data.

        Args:
            x_t_N_3 (Tensor): Current positions at time t, shape (N, 3)
            batch (Data): PyTorch Geometric batch containing molecular data
            t_G (Tensor): Per-graph time values, shape (num_graphs, 1)

        Returns:
            Tensor: Predicted flow field/velocity, shape (N, 3)

        Raises:
            ValueError: If atomwise_3D_out_layer is not initialized
        """
        h_N_D, X_N_L_D = self.representation(x_t_N_3, t_G, batch)
        if self.atomwise_3D_out_layer is not None:
            atom_N_3 = self.atomwise_3D_out_layer(h_N_D, X_N_L_D[:, :3, :])
        else:
            # Default to some output, perhaps identity or raise error
            raise ValueError("atomwise_3D_out_layer is not initialized")
        return atom_N_3

    def train_val_step(self, batch: Data) -> Tensor:
        x_t_N_3, dx_dt_N_3, t_G = self.get_perturbed_flow_point_and_time(batch)

        atom_N_3 = self.model_output(x_t_N_3, batch, t_G)

        return rmsd_loss(atom_N_3, dx_dt_N_3)

    def training_step(self, batch: Data, batch_idx: int) -> Tensor:
        loss = self.train_val_step(batch)
        self.log(
            "train/loss",
            loss,
            prog_bar=True,
            on_step=True,
            on_epoch=False,
            batch_size=batch.num_graphs,
        )
        return loss

    def validation_step(self, batch: Data, batch_idx: int) -> Tensor:
        loss = self.train_val_step(batch)
        self.log(
            "validation/val_loss",
            loss,
            prog_bar=True,
            on_step=False,
            on_epoch=True,
            batch_size=batch.num_graphs,
        )
        return loss

    def _calculate_potentials_S_Nm(
        self, positions_S_N_3: Tensor, batch: Batch
    ) -> Tuple[Tensor, Tensor]:
        """
        Calculates multiple potentials by iterating through each sample.
        This is compatible with potential functions that expect (N, 3) input.

        Args:
            positions_S_N_3: Tensor of shape (S, N, 3)
            batch: PyG Batch object

        Returns:
            Tensor of shape (S, Nm) representing the summed potential for each graph.
        """
        with torch.enable_grad() and torch.inference_mode(False):
            S = positions_S_N_3.shape[0]
            Nm = batch.num_graphs
            total_potential_S_Nm = torch.zeros((S, Nm), device=positions_S_N_3.device)
            gradients_S_N_3 = torch.zeros_like(positions_S_N_3, device=positions_S_N_3.device)
            # Iterate through each of the S particles/samples
            for s_idx in range(S):
                pos_N_3 = positions_S_N_3[s_idx, ...].clone()

                # Make sure it requires gradients
                if not pos_N_3.requires_grad:
                    pos_N_3 = pos_N_3.requires_grad_(True)

                slice_gradients_N_3 = torch.zeros_like(pos_N_3, device=positions_S_N_3.device)

                # Calculate all specified potentials for this single sample
                for potential_fn in self.fk_potential_fns:
                    # potential_fn expects (N, 3) and returns (Nm,)
                    potential_for_sample_Nm = potential_fn(pos_N_3, batch, device=self.device)
                    # Only compute gradients if the potential actually depends on positions
                    if potential_for_sample_Nm.requires_grad:
                        grads = torch.autograd.grad(
                            outputs=potential_for_sample_Nm.sum(),  # Sum to get scalar
                            inputs=(pos_N_3,),
                            retain_graph=True,
                            allow_unused=True,
                        )
                        if grads[0] is not None:
                            slice_gradients_N_3 += grads[0]
                    total_potential_S_Nm[s_idx, :] += potential_for_sample_Nm

                gradients_S_N_3[s_idx, ...] = slice_gradients_N_3

        return total_potential_S_Nm, gradients_S_N_3

    def _get_resampling_indices(self, G_t_S_Nm, batch):
        """
        Get resampling indices based on the resampling method.

        Args:
            G_t_S_Nm: Resampling weights (S, Nm)
            batch: Batch object

        Returns:
            sampled_indices_Nm_S: Sampled indices (Nm, S)
        """
        G_Nm_S = G_t_S_Nm.T
        sampled_indices_Nm_S = self.resampler.resample(G_Nm_S, self.num_samples)
        return sampled_indices_Nm_S

    def _reindex_resampled_positions(self, positions_S_N_3, sampled_indices_Nm_S, batch):
        """
        Reindex the resampled positions based on the sampled indices.

        Args:
            positions_S_N_3: Current particle positions (S, N, 3)
            sampled_indices_Nm_S: Sampled indices (Nm, S)
            batch: Batch object

        Returns:
            pos_new_S_N_3: Resampled positions (S, N, 3)
        """
        output_pos_slices = []
        batch_batch = batch.batch.cpu()

        for s_target_idx in range(self.num_samples):
            s_original_indices_for_graphs_Nm = sampled_indices_Nm_S[:, s_target_idx]
            s_original_indices_for_nodes_Nm = s_original_indices_for_graphs_Nm[batch_batch]
            current_target_slice_N_3 = positions_S_N_3[
                s_original_indices_for_nodes_Nm,
                torch.arange(batch.num_nodes, device=self.device),
                :,
            ]
            output_pos_slices.append(current_target_slice_N_3)

        pos_new_S_N_3 = torch.stack(output_pos_slices, dim=0)
        return pos_new_S_N_3

    def _flow_and_diffuse_with_precomputed_drift_step(
        self,
        pos_current_S_N_3: Tensor,
        drift_S_N_3: Tensor,
        batch: Batch,
        t: float,
        dt: float,
    ) -> Tensor:
        """
        One Euler-Maruyama step of the SDE with the same marginals as the flow ODE,

            x += [v + sigma_t^2 / (2 (1 - t)) * (t v - x)] dt + sigma_t sqrt(dt) eps,

        where v is the precomputed drift (the flow velocity). With sigma_t = 0 this is
        the Euler step of the ODE. No update at t = 1.
        Returns:
            pos_new_S_N_3: Updated positions after flow and diffusion (S, N, 3)
        """
        sigma_t_val = self._sigma_t(t)
        if 1 - t < 1e-6:
            pos_new_S_N_3 = pos_current_S_N_3
        else:
            term1_factor = 1 - (sigma_t_val**2) / (2 * (1 - t)) * dt
            term2_factor = ((2 * (1 - t) + t * sigma_t_val**2) / (2 * (1 - t))) * dt
            diffusion = sigma_t_val * np.sqrt(dt) * torch.randn_like(pos_current_S_N_3)

            pos_new_S_N_3 = (
                term1_factor * pos_current_S_N_3 + term2_factor * drift_S_N_3 + diffusion
            )
        return pos_new_S_N_3

    def _calculate_drift_S_N_3(
        self,
        pos_current_S_N_3: Tensor,
        batch: Batch,
        ode_func: Callable[[float, Tensor], Tensor],
        t: float,
    ) -> Tensor:
        """
        Calculate the drift for the current positions.
        """
        drifts_list = []
        for s_idx in range(self.num_samples):
            drifts_list.append(ode_func(t, pos_current_S_N_3[s_idx, ...]))
        drift_S_N_3 = torch.stack(drifts_list, dim=0)
        return drift_S_N_3

    def _fk_steering_loop(self, batch: Batch) -> Tensor:
        """Feynman-Kac steered generation loop."""
        t_T = self._time_grid()
        n_T = len(t_T)
        event_steps = self._event_steps(t_T)
        # number of resampling events
        L = len(event_steps)

        # Initialize positions
        pos_current_S_N_3 = torch.zeros((self.num_samples, batch.num_nodes, 3), device=self.device)
        for i in range(self.num_samples):
            torch.manual_seed(self.seed + i)
            pos_current_S_N_3[i, ...] = self._sample_prior(batch.pos[:, self.pos_fit_idx, :], batch)

        pos_gen_traj_S_T_N_3 = torch.zeros(
            (self.num_samples, n_T, batch.num_nodes, 3), device=self.device
        )
        pos_gen_traj_S_T_N_3[:, 0, ...] = pos_current_S_N_3

        drift_traj_S_T_N_3 = torch.zeros(
            (self.num_samples, n_T, batch.num_nodes, 3), device=self.device
        )

        # Initialize FK weights
        log_weights_S_Nm = torch.zeros((self.num_samples, batch.num_graphs), device=self.device)
        log_weights_cum_S_Nm = torch.zeros((self.num_samples, batch.num_graphs), device=self.device)
        potential_prev_S_Nm = None
        potential_max_S_Nm = None

        def ode_func(t, x_t_N_3):
            t_G = torch.tensor([t] * batch.num_graphs, device=self.device)
            return self.model_output(x_t_N_3, batch, t_G)

        # Calculate initial drifts
        drift_S_N_3 = self._calculate_drift_S_N_3(pos_current_S_N_3, batch, ode_func, 0)
        # Store initial drift
        drift_traj_S_T_N_3[:, 0, ...] = drift_S_N_3

        # Lth harmonic number
        if self.fk_potential_scheduler == "harmonic_sum" and L > 0:
            H_L = torch.sum(1.0 / torch.arange(1, L + 1, device=self.device, dtype=torch.float32))
        else:
            H_L = None

        ell = 0
        t_steering = 0.0
        # diagnostics
        ess_traj = []
        ancestors_traj = []

        # --- Main Steering Loop ---
        for k, t in enumerate(t_T[:-1]):
            t_val = float(t)
            dt = float(t_T[k + 1] - t)
            is_last = k == n_T - 2

            # 1. FIRST: Do dynamics step
            pos_new_S_N_3 = self._flow_and_diffuse_with_precomputed_drift_step(
                pos_current_S_N_3, drift_S_N_3, batch, t, dt
            )

            # Calculate the drift for the new positions
            if not is_last:
                drift_S_N_3 = self._calculate_drift_S_N_3(pos_new_S_N_3, batch, ode_func, t + dt)
                drift_traj_S_T_N_3[:, k + 1, ...] = drift_S_N_3

            is_event = k in event_steps

            # 2. THEN: Calculate potential and update weights
            if is_event:
                if is_last:
                    potential_S_Nm, _ = self._calculate_potentials_S_Nm(pos_new_S_N_3, batch)
                else:
                    pos_one_shot_S_N_3 = pos_new_S_N_3 + (1 - (t + dt)) * drift_S_N_3
                    potential_S_Nm, _ = self._calculate_potentials_S_Nm(pos_one_shot_S_N_3, batch)

                if self.fk_potential_scheduler == "sum":
                    if is_last:
                        log_weights_S_Nm = (
                            -self.fk_steering_temperature * potential_S_Nm - log_weights_cum_S_Nm
                        )
                    else:
                        log_weights_S_Nm = (
                            log_weights_S_Nm
                            - self.fk_steering_temperature * potential_S_Nm * (t_val - t_steering)
                        )
                    log_weights_cum_S_Nm = log_weights_cum_S_Nm + log_weights_S_Nm

                elif self.fk_potential_scheduler == "harmonic_sum":
                    if is_last:
                        log_weights_S_Nm = (
                            -self.fk_steering_temperature * potential_S_Nm - log_weights_cum_S_Nm
                        )
                    elif H_L is not None:
                        log_weights_S_Nm = log_weights_S_Nm - (
                            self.fk_steering_temperature * (potential_S_Nm / ((L - ell) * H_L))
                        )
                    log_weights_cum_S_Nm = log_weights_cum_S_Nm + log_weights_S_Nm

                elif self.fk_potential_scheduler == "difference":
                    if potential_prev_S_Nm is None:
                        log_weights_S_Nm = -self.fk_steering_temperature * potential_S_Nm
                    else:
                        log_weights_S_Nm = self.fk_steering_temperature * (
                            potential_prev_S_Nm - potential_S_Nm
                        )
                    potential_prev_S_Nm = potential_S_Nm

                elif self.fk_potential_scheduler == "max":
                    if is_last:
                        log_weights_S_Nm = (
                            -self.fk_steering_temperature * potential_S_Nm - log_weights_cum_S_Nm
                        )
                    else:
                        potential_max_S_Nm = (
                            potential_S_Nm
                            if potential_max_S_Nm is None
                            else torch.maximum(potential_max_S_Nm, potential_S_Nm)
                        )
                        log_weights_S_Nm = -self.fk_steering_temperature * potential_max_S_Nm
                    log_weights_cum_S_Nm = log_weights_cum_S_Nm + log_weights_S_Nm

                t_steering = t_val
                ell += 1

                # 3. Resample
                stable_log_weights = log_weights_S_Nm - log_weights_S_Nm.max(dim=0, keepdim=True)[0]
                G_t_S_Nm = torch.exp(stable_log_weights)
                G_t_S_Nm = G_t_S_Nm / (G_t_S_Nm.sum(dim=0, keepdim=True) + 1e-8)

                indices_Nm_S = self._get_resampling_indices(G_t_S_Nm, batch)
                if is_last and self.save_final_particles:
                    self._final_pre = (pos_new_S_N_3.clone(), potential_S_Nm.clone())
                # ESS and number of surviving ancestors
                ess_traj.append(1.0 / (G_t_S_Nm**2).sum(dim=0).clamp_min(1e-12))
                ancestor_hits_Nm_S = torch.zeros(
                    indices_Nm_S.shape, device=self.device, dtype=torch.float32
                )
                ancestor_hits_Nm_S.scatter_(1, indices_Nm_S, 1.0)
                ancestors_traj.append(ancestor_hits_Nm_S.sum(dim=1))
                pos_new_S_N_3 = self._reindex_resampled_positions(
                    pos_new_S_N_3, indices_Nm_S, batch
                )
                drift_S_N_3 = self._reindex_resampled_positions(drift_S_N_3, indices_Nm_S, batch)
                # Reindex weights
                log_weights_S_Nm = torch.gather(log_weights_S_Nm, 0, indices_Nm_S.T)
                log_weights_cum_S_Nm = torch.gather(log_weights_cum_S_Nm, 0, indices_Nm_S.T)
                if potential_prev_S_Nm is not None:
                    potential_prev_S_Nm = torch.gather(potential_prev_S_Nm, 0, indices_Nm_S.T)
                if potential_max_S_Nm is not None:
                    potential_max_S_Nm = torch.gather(potential_max_S_Nm, 0, indices_Nm_S.T)
                assert log_weights_S_Nm.shape == (self.num_samples, batch.num_graphs), (
                    f"Expected log_weights_S_Nm shape {(self.num_samples, batch.num_graphs)}, "
                    f"got {log_weights_S_Nm.shape}"
                )

            # Update current positions
            pos_current_S_N_3 = pos_new_S_N_3
            pos_gen_traj_S_T_N_3[:, k + 1, ...] = pos_current_S_N_3

        if ess_traj:
            ess_L_Nm = torch.stack(ess_traj)
            ancestors_L_Nm = torch.stack(ancestors_traj)
            self._fk_diag_Nm = {
                "ess_mean": ess_L_Nm.mean(dim=0),
                "ess_min": ess_L_Nm.min(dim=0).values,
                "ancestors_final": ancestors_L_Nm[-1],
            }
        else:
            self._fk_diag_Nm = None

        return pos_gen_traj_S_T_N_3

    def test_step(self, batch: Batch, batch_idx: int):
        self.seed += 1
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)

        if self.inference_sampling_method == "fk_steering":
            self._final_pre = None
            pos_gen_traj_S_T_N_3 = self._fk_steering_loop(batch)
        else:
            self._fk_diag_Nm = None
            t_T = self._time_grid()

            def ode_func(t, x_t_N_3):
                t_G = torch.tensor([t] * batch.num_graphs, device=self.device)
                return self.model_output(x_t_N_3, batch, t_G)

            pos_gen_traj_S_T_N_3 = torch.zeros(
                (self.num_samples, len(t_T), batch.num_nodes, 3),
                device=self.device,
            )
            for i in range(self.num_samples):
                torch.manual_seed(self.seed + i)
                pos_init_N_3 = self._sample_prior(batch.pos[:, self.pos_fit_idx, :], batch)
                pos_gen_traj_S_T_N_3[i, ...] = odeint(ode_func, pos_init_N_3, t_T, method="euler")

        # Aggregation
        for j, data in enumerate(batch.to_data_list()):
            # Get single molecule positions from sampled trajectories
            mask = (batch.batch == j).cpu()
            pos_gen_traj_S_T_Nm_3 = pos_gen_traj_S_T_N_3[:, :, mask]
            pos_gen_traj_Sv_T_Nm_3 = self.align_and_rotate_samples(data, pos_gen_traj_S_T_Nm_3)
            # -------------------------- START: Aggregate the S samples --------------------------
            pos_aggr_Nm_3 = torch.median(pos_gen_traj_Sv_T_Nm_3[:, -1, :, :], dim=0).values
            distances_Sv = torch.linalg.vector_norm(
                pos_gen_traj_Sv_T_Nm_3[:, -1, :, :] - pos_aggr_Nm_3, dim=(1, 2)
            )
            pos_best_T_Nm_3 = pos_gen_traj_Sv_T_Nm_3[torch.argmin(distances_Sv)]
            # -------------------------- END: Aggregate the S samples --------------------------
            data.pos_gen = pos_best_T_Nm_3
            if self._final_pre is not None:
                pos_pre_S_N_3, potential_pre_S_Nm = self._final_pre
                data.particles_pre = self.align_and_rotate_samples(
                    data, pos_pre_S_N_3[:, None, mask]
                )[:, -1]
                data.particle_potential_pre = potential_pre_S_Nm[:, j]
            if self.save_final_particles:
                data.particles_post = pos_gen_traj_Sv_T_Nm_3[:, -1]
            if self._fk_diag_Nm is not None:
                for diag_key, diag_Nm in self._fk_diag_Nm.items():
                    setattr(data, diag_key, float(diag_Nm[j]))
            self.results_R.append(data.to("cpu"))

    def align_and_rotate_samples(self, data, pos_gen_traj_S_T_Nm_3):
        """
        For each molecule we sample the TS S times.
        Do substructure matching, and align (w. Kabsch) to the GT.
        """
        pos_gt_Nm_3 = data.pos[:, self.pos_fit_idx, :]

        # Substructure matching (batched for S)
        matches = get_substruct_matches(data.smiles)
        match_Sv_Nm = get_min_dmae_match_torch(
            matches, pos_gt_Nm_3, pos_gen_traj_S_T_Nm_3[:, -1, :, :]
        )
        pos_gen_traj_S_T_Nm_3[:, -1, :, :] = torch.gather(
            pos_gen_traj_S_T_Nm_3[:, -1, :, :],
            1,
            match_Sv_Nm.unsqueeze(-1).expand(-1, -1, 3),
        )

        # Kabsch rotation
        S = pos_gen_traj_S_T_Nm_3.shape[0]
        Nm = pos_gen_traj_S_T_Nm_3.shape[2]
        # This is a trick to make the batched rotation to the GT molecule easy
        data.batch = torch.arange(S, device=self.device).repeat_interleave(Nm)
        # Repeat GT pos S times (have to rotate each sample to it)
        pos_gt_SNm_3 = pos_gt_Nm_3.repeat(S, 1)
        # Reshape [S, Nm] nodes to single S*Nm dimension. Makes batched rotation possible.
        pos_gen_SNm_3 = pos_gen_traj_S_T_Nm_3[:, -1, :, :].reshape(S * Nm, 3)
        pos_gen_aligned_SNm_3 = get_shortest_path_batched_x_1(pos_gt_SNm_3, pos_gen_SNm_3, data)
        pos_gen_aligned_S_Nm_3 = pos_gen_aligned_SNm_3.reshape(S, Nm, 3)
        pos_gen_traj_S_T_Nm_3[:, -1, :, :] = pos_gen_aligned_S_Nm_3

        return pos_gen_traj_S_T_Nm_3

    def configure_optimizers(self) -> Tuple[List[torch.optim.Optimizer], List[Dict]]:
        """Configure optimizers and learning rate schedulers."""
        if self.schedule_free:
            import schedulefree

            optimizer = schedulefree.AdamWScheduleFreeClosure(
                self.trainer.model.parameters(),
                lr=self.lr,
                weight_decay=self.weight_decay,
                eps=1e-8,
                warmup_steps=self.lr_warmup_steps,
            )
            return [optimizer], []

        optimizer = torch.optim.AdamW(
            self.trainer.model.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
            eps=1e-7,
        )

        if self.scheduler and callable(self.scheduler):
            scheduler, _ = self.scheduler(optimizer=optimizer)
        else:
            scheduler = ReduceLROnPlateau(
                optimizer,
                factor=self.lr_decay,
                patience=self.lr_patience,
                min_lr=self.lr_minlr,
            )

        schedule = {
            "scheduler": scheduler,
            "monitor": self.lr_monitor,
            "interval": "epoch",
            "frequency": 1,
            "strict": True,
        }

        return [optimizer], [schedule]
