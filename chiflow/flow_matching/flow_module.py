from typing import Dict, Optional, Callable, TypeVar, List, Union, Tuple
import torch
from torch import nn, Tensor
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch_geometric.data import Data, Batch
import lightning.pytorch as pl
from torchdiffeq import odeint

from flow_matching.utils import (
    get_shortest_path_batched_x_1,
    rmsd_loss,
    no_steric_clash,
    get_substruct_matches,
    get_min_dmae_match_torch,
    compute_steric_clash_batched_fast,
    compute_steric_clash_batched,
    chiral_potential,
    steric_potential,
)
import numpy as np


from gotennet.models.components.outputs import Atomwise3DOut
from gotennet.models.representation.gotennet import GotenNet
from flow_matching.resampling import Resampler


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
    Flow Matching module with Feynman-Kac steering for chemical reaction transition state generation.
    
    This module implements continuous normalizing flows with optional FK steering for generating
    molecular conformations, particularly focused on chemical reaction transition states with
    chirality awareness.
    
    Args:
        representation (GotenNet): The neural network representation model
        lr (float): Learning rate for optimization. Defaults to 5e-4.
        lr_decay (float): Learning rate decay factor. Defaults to 0.5.
        lr_patience (int): Patience for learning rate scheduler. Defaults to 100.
        lr_minlr (float): Minimum learning rate. Defaults to 1e-6.
        lr_monitor (str): Metric to monitor for LR scheduling. Defaults to "validation/ema_val_loss".
        chain_scheduler (Optional[Callable]): Optional chained scheduler. Defaults to None.
        weight_decay (float): Weight decay for optimizer. Defaults to 0.01.
        num_steps (int): Number of integration steps for sampling. Defaults to 10.
        num_samples (int): Number of samples for FK steering. Defaults to 1.
        seed (int): Random seed. Defaults to 1.
        filter_clash (bool): Whether to filter steric clashes. Defaults to False.
        pos_fit_idx (int): Position index to fit. Defaults to 1.
        ema_decay (float): Exponential moving average decay. Defaults to 0.9.
        dataset_meta (Optional[Dict]): Dataset metadata. Defaults to None.
        output (Optional[Dict]): Output layer configuration. Defaults to None.
        scheduler (Optional[Callable]): Learning rate scheduler. Defaults to None.
        save_predictions (Optional[bool]): Whether to save predictions. Defaults to None.
        input_contribution (float): Input contribution weight. Defaults to 1.
        task_config (Optional[Dict]): Task configuration. Defaults to None.
        lr_warmup_steps (int): Learning rate warmup steps. Defaults to 0.
        schedule_free (bool): Whether to use schedule-free optimization. Defaults to False.
        use_ema (bool): Whether to use exponential moving average. Defaults to False.
        inference_sampling_method (str): Sampling method for inference ("fk_steering" or "median"). 
                                       Defaults to "fk_steering".
        fk_potential_fns (Optional[List[Callable]]): List of FK potential functions. 
                                                   Defaults to [chiral_potential].
        fk_potential_scheduler (str): How to combine multiple potentials ("sum", "difference", 
                                    "max", "harmonic_sum"). Defaults to "sum".
        steering_base_variance (float): Base variance for FK steering. Defaults to 1.0.
        fk_steering_temperature (float): Temperature parameter for FK steering. Defaults to 2e0.
        resample_method (str): Resampling method ("multinomial", "stratified", "residual", 
                             "systematic"). Defaults to "stratified".
        resample_freq (int): Frequency of resampling in FK loop. Defaults to 2.
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
        fk_potential_fns: Optional[List[Callable]] = [chiral_potential],
        fk_potential_scheduler: str = "sum",  # "difference", "sum", "max", "harmonic_sum"
        steering_base_variance: float = 1.0,
        fk_steering_temperature: float = 2e0,
        resample_method: str = "stratified",  # "multinomial", "stratified", "residual", "systematic"
        resample_freq: int = 2,  # Frequency of resampling in the Feynman-Kac loop
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
        print("FM has seed", self.seed, "-----------------------------")
        self.filter_clash = filter_clash
        self.pos_fit_idx = pos_fit_idx

        self.scheduler = scheduler

        self.save_hyperparameters()
        # save results in test_step
        self.results_R = []

        self.inference_sampling_method = inference_sampling_method

        self.fk_potential_fns = fk_potential_fns if fk_potential_fns is not None else []
        valid_schedulers = ["difference", "sum", "harmonic_sum", "max"]
        if fk_potential_scheduler not in valid_schedulers:
            raise ValueError(
                f"fk_potential_scheduler must be one of {valid_schedulers}"
            )
        self.fk_potential_scheduler = fk_potential_scheduler
        self.steering_base_variance = steering_base_variance
        self.fk_steering_temperature = fk_steering_temperature
        self.resample_freq = resample_freq

        self.resampler = Resampler(resample_method=resample_method)

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
            gradients_S_N_3 = torch.zeros_like(
                positions_S_N_3, device=positions_S_N_3.device
            )
            # Iterate through each of the S particles/samples
            for s_idx in range(S):
                pos_N_3 = positions_S_N_3[s_idx, ...].clone()

                # Make sure it requires gradients
                if not pos_N_3.requires_grad:
                    pos_N_3 = pos_N_3.requires_grad_(True)

                slice_gradients_N_3 = torch.zeros_like(
                    pos_N_3, device=positions_S_N_3.device
                )

                # Calculate all specified potentials for this single sample
                for potential_fn in self.fk_potential_fns:
                    # potential_fn expects (N, 3) and returns (Nm,)
                    potential_for_sample_Nm = potential_fn(
                        pos_N_3, batch, device=self.device
                    )
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

    def _calculate_steering_reward_normalized_S_Nm(self, positions_S_N_3, batch):
        r_S_Nm = self._potential(positions_S_N_3, batch)
        G_S_Nm = torch.exp(-1 * self.steering_temperature * r_S_Nm)
        # Normalize along sample axis:
        G_norm_S_Nm = torch.softmax(G_S_Nm, dim=0)
        return G_norm_S_Nm

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

    def _reindex_resampled_positions(
        self, positions_S_N_3, sampled_indices_Nm_S, batch
    ):
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
            s_original_indices_for_nodes_Nm = s_original_indices_for_graphs_Nm[
                batch_batch
            ]
            current_target_slice_N_3 = positions_S_N_3[
                s_original_indices_for_nodes_Nm,
                torch.arange(batch.num_nodes, device=self.device),
                :,
            ]
            output_pos_slices.append(current_target_slice_N_3)

        pos_new_S_N_3 = torch.stack(output_pos_slices, dim=0)
        return pos_new_S_N_3

    def _flow_and_diffuse_step(
        self,
        pos_current_S_N_3: Tensor,
        batch: Batch,
        t: float,
        dt: float,
        ode_func: Callable[[float, Tensor], Tensor],
    ) -> Tuple[Tensor, Tensor]:
        """
        Perform a single flow and diffusion step.
        Returns:
            pos_new_S_N_3: Updated positions after flow and diffusion (S, N, 3)
            drift_S_N_3: Drift for each sample (S, N, 3)
        """
        sigma_t_val = self.steering_base_variance * (1 - t)
        # Collect drifts from each sample
        drift_S_N_3 = self._calculate_drift_S_N_3(pos_current_S_N_3, batch, ode_func, t)

        if 1 - t < 1e-6:
            pos_new_S_N_3 = pos_current_S_N_3
        else:
            term1_factor = 1 - (sigma_t_val**2) / (2 * (1 - t)) * dt
            term2_factor = ((2 * (1 - t) + t * sigma_t_val**2) / (2 * (1 - t))) * dt
            diffusion = sigma_t_val * np.sqrt(dt) * torch.randn_like(pos_current_S_N_3)

            pos_new_S_N_3 = (
                term1_factor * pos_current_S_N_3
                + term2_factor * drift_S_N_3
                + diffusion
            )

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
        Perform a single flow and diffusion step with precomputed drift.
        Returns:
            pos_new_S_N_3: Updated positions after flow and diffusion (S, N, 3)
        """
        sigma_t_val = self.steering_base_variance * (1 - t)
        if 1 - t < 1e-6:
            pos_new_S_N_3 = pos_current_S_N_3
        else:
            term1_factor = 1 - (sigma_t_val**2) / (2 * (1 - t)) * dt
            term2_factor = ((2 * (1 - t) + t * sigma_t_val**2) / (2 * (1 - t))) * dt
            diffusion = sigma_t_val * np.sqrt(dt) * torch.randn_like(pos_current_S_N_3)

            pos_new_S_N_3 = (
                term1_factor * pos_current_S_N_3
                + term2_factor * drift_S_N_3
                + diffusion
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
        """Corrected Feynman-Kac steered generation loop."""
        t_T = torch.linspace(0, 1, steps=self.num_steps, device=self.device)
        dt = 1 / (self.num_steps - 1)

        # Initialize positions
        pos_current_S_N_3 = torch.zeros(
            (self.num_samples, batch.num_nodes, 3), device=self.device
        )
        for i in range(self.num_samples):
            torch.manual_seed(self.seed + i)
            pos_current_S_N_3[i, ...] = self._sample_prior(
                batch.pos[:, self.pos_fit_idx, :], batch
            )

        pos_gen_traj_S_T_N_3 = torch.zeros(
            (self.num_samples, self.num_steps, batch.num_nodes, 3), device=self.device
        )
        pos_gen_traj_S_T_N_3[:, 0, ...] = pos_current_S_N_3

        drift_traj_S_T_N_3 = torch.zeros(
            (self.num_samples, self.num_steps, batch.num_nodes, 3), device=self.device
        )

        # Initialize FK weights (accumulate over trajectory)
        log_weights_S_Nm = torch.zeros(
            (self.num_samples, batch.num_graphs), device=self.device
        )

        # Initialize FK trajectories
        log_weights_traj_S_T_Nm = torch.zeros(
            (self.num_samples, self.num_steps, batch.num_graphs), device=self.device
        )
        # Initialise potentials for difference calculation
        potential_prev_S_Nm = torch.zeros(
            (self.num_samples, batch.num_graphs), device=self.device
        )

        def ode_func(t, x_t_N_3):
            t_G = torch.tensor([t] * batch.num_graphs, device=self.device)
            return self.model_output(x_t_N_3, batch, t_G)

        # Calculate initial drifts
        drift_S_N_3 = self._calculate_drift_S_N_3(pos_current_S_N_3, batch, ode_func, 0)
        # Store initial drift
        drift_traj_S_T_N_3[:, 0, ...] = drift_S_N_3

        # nth harmonic number
        H_n = sum(1 / torch.arange(1, self.num_steps + 1, device=self.device))

        # --- Main Steering Loop ---
        for k, t in enumerate(t_T[:-1]):
            # 1. FIRST: Do dynamics step
            pos_new_S_N_3 = self._flow_and_diffuse_with_precomputed_drift_step(
                pos_current_S_N_3, drift_S_N_3, batch, t, dt
            )

            # Calculate the drift for the new positions
            drift_S_N_3 = self._calculate_drift_S_N_3(
                pos_new_S_N_3, batch, ode_func, t + dt
            )
            # Store the drift for this step
            drift_traj_S_T_N_3[:, k + 1, ...] = drift_S_N_3

            # Calculate one-shot estimated potential for the new positions
            # i.e. at pos_new_S_N_3 + (1 - (t+dt)) * drift_S_N_3

            pos_one_shot_S_N_3 = pos_new_S_N_3 + (1 - (t + dt)) * drift_S_N_3

            # 2. THEN: Calculate potential at one shot position and update weights
            potential_new_S_Nm, _ = self._calculate_potentials_S_Nm(
                pos_one_shot_S_N_3, batch
            )

            # Accumulate weights (integrate potential along path)
            if self.fk_potential_scheduler == "sum":
                log_weights_S_Nm -= (
                    self.fk_steering_temperature * potential_new_S_Nm * dt
                )  # (1/((self.num_steps- k +1)*H_n))
            elif self.fk_potential_scheduler == "harmonic_sum":
                log_weights_S_Nm -= (
                    self.fk_steering_temperature
                    * potential_new_S_Nm
                    / ((self.num_steps - k + 1) * H_n)
                )
            elif self.fk_potential_scheduler == "difference":
                if k == 0:
                    log_weights_S_Nm = (
                        -self.fk_steering_temperature * potential_new_S_Nm
                    )
                else:
                    log_weights_S_Nm = self.fk_steering_temperature * (
                        potential_prev_S_Nm - potential_new_S_Nm
                    )

            # Store the potential for the next step
            potential_prev_S_Nm = potential_new_S_Nm.clone()

            # Store the log weights for this step
            log_weights_traj_S_T_Nm[:, k + 1, :] = log_weights_S_Nm

            # 3. Resample periodically (not every step)
            if self.resample_freq is not None and (k + 1) % self.resample_freq == 0:
                # Convert to normalized weights for resampling
                stable_log_weights = (
                    log_weights_S_Nm - log_weights_S_Nm.max(dim=0, keepdim=True)[0]
                )
                G_t_S_Nm = torch.exp(stable_log_weights)
                G_t_S_Nm = G_t_S_Nm / (G_t_S_Nm.sum(dim=0, keepdim=True) + 1e-8)

                indices_Nm_S = self._get_resampling_indices(G_t_S_Nm, batch)
                # Resample positions based on indices
                pos_new_S_N_3 = self._reindex_resampled_positions(
                    pos_new_S_N_3, indices_Nm_S, batch
                )
                drift_S_N_3 = self._reindex_resampled_positions(
                    drift_S_N_3, indices_Nm_S, batch
                )
                # Reindex weights
                # Note: indices_Nm_S is (Nm, S), so we need to transpose it to (S, Nm)
                # Use torch.gather to select the correct indices along the sample dimension
                # indices_Nm_S: (Nm, S), need to gather along dim=0 (samples)
                log_weights_S_Nm = torch.gather(log_weights_S_Nm, 0, indices_Nm_S.T)
                # check dimensions
                assert log_weights_S_Nm.shape == (self.num_samples, batch.num_graphs), (
                    f"Expected log_weights_S_Nm shape {(self.num_samples, batch.num_graphs)}, got {log_weights_S_Nm.shape}"
                )

                # Also reindex the log weights trajectory (for all previous steps)
                log_weights_traj_S_T_Nm = torch.gather(
                    log_weights_traj_S_T_Nm,
                    0,
                    indices_Nm_S.T.unsqueeze(1).expand(-1, self.num_steps, -1),
                )

            # Update current positions
            pos_current_S_N_3 = pos_new_S_N_3
            pos_gen_traj_S_T_N_3[:, k + 1, ...] = pos_current_S_N_3

        if (
            self.fk_potential_scheduler == "max"
            or self.fk_potential_scheduler == "sum"
            or self.fk_potential_scheduler == "harmonic_sum"
        ):
            # In the end, calculate the final reweighing:
            # Calculate the final potential at the last positions
            potential_final_S_Nm, _ = self._calculate_potentials_S_Nm(
                pos_gen_traj_S_T_N_3[:, -1, :, :], batch
            )
            # The final reweighing log weight should be (- self.fk_steering_temperature * potential_final_S_Nm) - (sum of all previous log weights)
            final_log_weights_S_Nm = (
                -self.fk_steering_temperature * potential_final_S_Nm
                + torch.sum(log_weights_traj_S_T_Nm, dim=1)
            )
            # Resample based on the final log weights
            G_t_S_Nm = torch.exp(final_log_weights_S_Nm)
            G_t_S_Nm = G_t_S_Nm / (G_t_S_Nm.sum(dim=0, keepdim=True) + 1e-8)
            indices_Nm_S = self._get_resampling_indices(G_t_S_Nm, batch)
            pos_gen_traj_S_T_N_3[:, -1, :, :] = self._reindex_resampled_positions(
                pos_gen_traj_S_T_N_3[:, -1, :, :], indices_Nm_S, batch
            )
        return pos_gen_traj_S_T_N_3

    def test_step(self, batch: Batch, batch_idx: int):
        self.seed += 1
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)

        # print(chiral_potential(batch.pos[:, 0, :], batch, self.device, pt=True))

        if self.inference_sampling_method == "fk_steering":
            pos_gen_traj_S_T_N_3 = self._fk_steering_loop(batch)
        else:
            t_T = torch.linspace(0, 1, steps=self.num_steps, device=self.device)

            def ode_func(t, x_t_N_3):
                t_G = torch.tensor([t] * batch.num_graphs, device=self.device)
                return self.model_output(x_t_N_3, batch, t_G)

            pos_gen_traj_S_T_N_3 = torch.zeros(
                (self.num_samples, self.num_steps, batch.num_nodes, 3),
                device=self.device,
            )
            for i in range(self.num_samples):
                torch.manual_seed(self.seed + i)
                pos_init_N_3 = self._sample_prior(
                    batch.pos[:, self.pos_fit_idx, :], batch
                )
                pos_gen_traj_S_T_N_3[i, ...] = odeint(
                    ode_func, pos_init_N_3, t_T, method="euler"
                )

        # Aggregation
        for j, data in enumerate(batch.to_data_list()):
            # Get single molecule positions from sampled trajectories
            mask = (batch.batch == j).cpu()
            pos_gen_traj_S_T_Nm_3 = pos_gen_traj_S_T_N_3[:, :, mask]
            pos_gen_traj_Sv_T_Nm_3 = self.align_and_rotate_samples(
                data, pos_gen_traj_S_T_Nm_3
            )
            # -------------------------- START: Aggregate the S samples --------------------------
            pos_aggr_Nm_3 = torch.median(
                pos_gen_traj_Sv_T_Nm_3[:, -1, :, :], dim=0
            ).values
            distances_Sv = torch.linalg.vector_norm(
                pos_gen_traj_Sv_T_Nm_3[:, -1, :, :] - pos_aggr_Nm_3, dim=(1, 2)
            )
            pos_best_T_Nm_3 = pos_gen_traj_Sv_T_Nm_3[torch.argmin(distances_Sv)]
            # -------------------------- END: Aggregate the S samples --------------------------
            data.pos_gen = pos_best_T_Nm_3
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
        pos_gen_aligned_SNm_3 = get_shortest_path_batched_x_1(
            pos_gt_SNm_3, pos_gen_SNm_3, data
        )
        pos_gen_aligned_S_Nm_3 = pos_gen_aligned_SNm_3.reshape(S, Nm, 3)
        pos_gen_traj_S_T_Nm_3[:, -1, :, :] = pos_gen_aligned_S_Nm_3

        return pos_gen_traj_S_T_Nm_3

    def configure_optimizers(self) -> Tuple[List[torch.optim.Optimizer], List[Dict]]:
        """Configure optimizers and learning rate schedulers."""
        print("self.weight_decay", self.weight_decay)
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
