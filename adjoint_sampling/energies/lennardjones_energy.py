# Copyright (c) Meta Platforms, Inc. and affiliates.

import torch
# from fairchem.core.common.utils import conditional_grad

import matplotlib.pyplot as plt
import PIL
import numpy as np
import os
from io import BytesIO
from typing import Optional

from hydra.utils import get_original_cwd

import torch_geometric.utils as tg_utils
from torch_geometric.utils import unbatch_edge_index, unbatch, sort_edge_index

from typing import Dict, Optional

from adjoint_sampling.energies.regularizers import bond_structure_regularizer
from adjoint_sampling.utils.visualize_utils import fig2img, interatomic_dist

def remove_mean(x, n_particles, spatial_dim):
    # x [B, N, 3]
    if x.ndim == 2:
        x = x.view(-1, n_particles, spatial_dim)
    return x - torch.mean(x, dim=1, keepdim=True)

def lennard_jones_energy_torch(r, eps=1.0, rm=1.0, tau=1.0):
    # p = 0.9
    lj = eps * ((rm / r)**12 - 2 * (rm / r)**6) # * 0.5 * tau
    return lj


# https://github.com/noegroup/bgflow/blob/main/bgflow/utils/shape.py

def tile(a, dim, n_tile):
    """
    Tiles a pytorch tensor along one an arbitrary dimension.

    Parameters
    ----------
    a : PyTorch tensor
        the tensor which is to be tiled
    dim : Integer
        dimension along the tensor is tiled
    n_tile : Integer
        number of tiles

    Returns
    -------
    b : PyTorch tensor
        the tensor with dimension `dim` tiled `n_tile` times
    """
    init_dim = a.size(dim)
    repeat_idx = [1] * a.dim()
    repeat_idx[dim] = n_tile
    a = a.repeat(*(repeat_idx))
    order_index = np.concatenate(
        [init_dim * np.arange(n_tile) + i for i in range(init_dim)]
    )
    order_index = torch.LongTensor(order_index).to(a).long()
    return torch.index_select(a, dim, order_index)

# https://github.com/noegroup/bgflow/blob/main/bgflow/utils/geometry.py

def distance_vectors(x, remove_diagonal=True):
    r"""
    Computes the matrix :math:`r` of all distance vectors between
    given input points where

    .. math::
        r_{ij} = x_{i} - y_{j}

    as used in :footcite:`Khler2020EquivariantFE`

    Parameters
    ----------
    x : torch.Tensor
        Tensor of shape `[n_batch, n_particles, n_dimensions]`
        containing input points.
    remove_diagonal : boolean
        Flag indicating whether the all-zero distance vectors
        `x_i - x_i` should be included in the result

    Returns
    -------
    r : torch.Tensor
        Matrix of all distance vectors r.
        If `remove_diagonal=True` this is a tensor of shape
            `[n_batch, n_particles, n_particles, n_dimensions]`.
        Otherwise this is a tensor of shape
            `[n_batch, n_particles, n_particles - 1, n_dimensions]`.

    Examples
    --------
    TODO

    References
    ----------
    .. footbibliography::

    """
    r = tile(x.unsqueeze(2), 2, x.shape[1])
    r = r - r.permute([0, 2, 1, 3])
    if remove_diagonal:
        r = r[:, torch.eye(x.shape[1], x.shape[1]) == 0].view(
            -1, x.shape[1], x.shape[1] - 1, x.shape[2]
        )
    return r


def distance_vectors_v2(x, y, remove_diagonal=True):
    """
    Computes the matrix `r` of all distance vectors between
    given input points x and y where
    .. math::
        r_{ij} = x_{i} - y_{j}

    Parameters
    ----------
    x : torch.Tensor
        Tensor of shape `[n_batch, n_particles, n_dimensions]`
        containing input points.
    y : torch.Tensor
        Tensor of shape `[n_batch, n_particles, n_dimensions]`
        containing input points.
    remove_diagonal : boolean
        Flag indicating whether the all-zero distance vectors
        `x_i - y_i` should be included in the result

    Returns
    -------
    r : torch.Tensor
        Matrix of all distance vectors r.
        If `remove_diagonal=True` this is a tensor of shape
            `[n_batch, n_particles, n_particles - 1, n_dimensions]`.
        Otherwise this is a tensor of shape
            `[n_batch, n_particles, n_particles, n_dimensions]`.

    Examples
    --------
    TODO
    """
    r1 = tile(x.unsqueeze(2), 2, x.shape[1])
    r2 = tile(y.unsqueeze(2), 2, y.shape[1])
    r = r1 - r2.permute([0, 2, 1, 3])
    if remove_diagonal:
        r = r[:, torch.eye(x.shape[1], x.shape[1]) == 0].view(
            -1, x.shape[1], x.shape[1] - 1, x.shape[2]
        )
    return r


def distances_from_vectors(r, eps=1e-6):
    """
    Computes the all-distance matrix from given distance vectors.
    
    Parameters
    ----------
    r : torch.Tensor
        Matrix of all distance vectors r.
        Tensor of shape `[n_batch, n_particles, n_other_particles, n_dimensions]`
    eps : Small real number.
        Regularizer to avoid division by zero.
    
    Returns
    -------
    d : torch.Tensor
        All-distance matrix d.
        Tensor of shape `[n_batch, n_particles, n_other_particles]`.
    """
    return (r.pow(2).sum(dim=-1) + eps).sqrt()


def compute_distances(x, n_particles, n_dimensions, remove_duplicates=True):
    """
    Computes the all distances for a given particle configuration x.

    Parameters
    ----------
    x : torch.Tensor
        Positions of n_particles in n_dimensions.
    remove_duplicates : boolean
        Flag indicating whether to remove duplicate distances
        and distances be.
        If False the all distance matrix is returned instead.

    Returns
    -------
    distances : torch.Tensor
        All-distances between particles in a configuration
        Tensor of shape `[n_batch, n_particles * (n_particles - 1) // 2]` if remove_duplicates.
        Otherwise `[n_batch, n_particles , n_particles]`
    """
    x = x.reshape(-1, n_particles, n_dimensions)
    distances = torch.cdist(x, x)
    if remove_duplicates:
        distances = distances[:, torch.triu(torch.ones((n_particles, n_particles)), diagonal=1) == 1]
        distances = distances.reshape(-1, n_particles * (n_particles - 1) // 2)
    return distances


class LennardJonesEnergy(torch.nn.Module):
    """
    Custom energy model for non-molecular systems.
    
    This model computes energies and forces for arbitrary coordinate-based systems
    without requiring molecular connectivity or SMILES strings.
    """
    
    def __init__(
        self, 
        n_particles, 
        spatial_dim=3,
        tau=1.0, 
        alpha=0.0, 
        x0=5.0, 
        device="cuda", 
        default_regularize=False, 
        use_oscillator=True, 
        oscillator_scale=1.0,
    ):
        """
        Initialize the custom energy model.
        
        Args:
            tau (float): Temperature parameter
            alpha (float): Regularization parameter
            device (str): Device to run model on
        """
        super().__init__()
        self.n_particles = n_particles
        self.spatial_dim = spatial_dim
        self.tau = tau
        self.alpha = alpha
        self.device = device
        self.x0 = x0
        self.default_regularize = default_regularize
        # Harmonic oscillator w.r.t. center of mass
        # used in iDEM and Koehler2020b
        self.use_oscillator = use_oscillator
        self.oscillator_scale = 0.25 # weight of c=0.5 according to iDEM and Koehler2020b
        self.plotting_bounds_r = [0.1, 5] # [min, max] for relative distance
        self.plotting_bounds_energy = [-1.5, 2.0] # [min, max] for energy
        
        # from fairchem_energy
        # self.atomic_numbers = torch.arange(100)
        self.atomic_numbers = torch.ones(n_particles, dtype=torch.long)
        
        # vmapped functions
        self.energy_vmapped = torch.vmap(self._energy_vmap)
        
        # data paths
        current_dir = os.path.dirname(os.path.abspath(__file__))
        # root is parent of parent of current dir
        project_root = os.path.dirname(os.path.dirname(current_dir))
        # MCMC samples from Klein et al. (2023b)
        # Equivariant flow matching
        # https://proceedings.neurips.cc/paper_files/paper/2023/hash/bc827452450356f9f558f4e4568d553b-Abstract-Conference.html
        # copied from
        # https://github.com/jarridrb/DEM/tree/8e9531987b0bfe4e770d5009c10a56b8434c0f0a/data
        if n_particles == 13:
            self.data_path_train = os.path.join(project_root, "data", f"train_split_LJ13-1000.npy")
            self.data_path_val = os.path.join(project_root, "data", f"val_split_LJ13-1000.npy")
            self.data_path_test = os.path.join(project_root, "data", f"test_split_LJ13-1000.npy")
        elif n_particles == 55:
            self.data_path_train = os.path.join(project_root, "data", f"train_split_LJ55-1000-part1.npy")
            self.data_path_val = os.path.join(project_root, "data", f"val_split_LJ55-1000-part1.npy")
            self.data_path_test = os.path.join(project_root, "data", f"test_split_LJ55-1000-part1.npy")
        else:
            self.data_path_train = None
            self.data_path_val = None
            self.data_path_test = None
        
        if n_particles in [13, 55]:
            self.train_set = self.setup_train_set()
            self.val_set = self.setup_val_set()
            self.test_set = self.setup_test_set()
        else:
            self.train_set = None
            self.val_set = None
            self.test_set = None
        
    
    # torch does not support composing vmap with torch.jit.script
    def _energy_vmap(self, positions):
        """
        Compute energy for a single system.
        Args:
            positions (torch.Tensor): Coordinates of shape [N, 3]
            
        Returns:
            torch.Tensor: Energy value [1]
        """
        x = positions
        N = x.shape[0]
        # Expand x to compute all pairwise differences
        x1 = x.unsqueeze(1)  # [N, 1, 3]
        x2 = x.unsqueeze(0)  # [1, N, 3]
        # Compute squared differences and sum over last dimension
        squared_diffs = (x1 - x2).pow(2).sum(dim=-1)  # [N, N]
        # Add small epsilon to avoid numerical issues
        dists = (squared_diffs + 1e-6).sqrt()
        # Remove duplicates and diagonal elements
        dists = dists[torch.triu(torch.ones((N, N)), diagonal=1) == 1]
        # Reshape to get flat array of unique distances
        dists = dists.reshape(N * (N - 1) // 2)
        # compute lennard jones energy
        lj_energies = ((1.0 / dists)**12 - 2 * (1.0 / dists)**6) 
        # lj_energies = torch.clip(lj_energies, -1e4, 1e4)
        lj_energies = lj_energies.sum()
        if self.use_oscillator:
            # [B, N, 3] -> [B]
            x_centered = (x - torch.mean(x, dim=1, keepdim=True))
            osc_energies = 0.5 * x_centered.pow(2).sum(dim=(-2, -1))
            lj_energies = lj_energies + osc_energies * self.oscillator_scale
        return lj_energies * 2

    def _energy(self, x):
        """
        Compute energy for a batch of systems.
        Args:
            positions (torch.Tensor): Coordinates of shape [B, N, 3]
            
        Returns:
            torch.Tensor: Energy value [B, 1]
        """
        # [B, N, N-1]
        dists = distances_from_vectors(
            distance_vectors(x) # [B, N, N-1, 3]
        )

        # [B, N, N-1] -> [B]
        lj_energies = lennard_jones_energy_torch(dists, 1.0, 1.0)
        # lj_energies = torch.clip(lj_energies, -1e4, 1e4)
        lj_energies = lj_energies.sum(dim=(-2, -1)) # [B]

        if self.use_oscillator:
            osc_energies = 0.5 * self._remove_mean(x).pow(2).sum(dim=(-2, -1)) # [B]
            lj_energies = lj_energies + osc_energies * self.oscillator_scale

        return lj_energies
    
    def forward(self, graph_state, regularize=None):
        """
        Compute energy and forces for the given state.
        
        Args:
            graph_state (dict): batch of DataBatch objects
            - positions: [B * N, 3]
            - batch: [B * N]
            - natoms: [B]
            
        Returns:
            dict: Dictionary 
            - energy: [B, 1]
            - forces: [B, N, 3]
            - reg_energy: [B, 1]
            - reg_forces: [B, N, 3]
        """
        if regularize is None:
            regularize = self.default_regularize
        
        # rename/add required input fields for fairchem model
        # adjoint_sampling.energies.fairchem_energy
        graph_state.natoms = graph_state.ptr[1:] - graph_state.ptr[:-1]
        graph_state.atomic_numbers = graph_state["node_attrs"].argmax(dim=-1)
        
        # useful to discern between systems
        batch_size = graph_state.natoms.numel()
        num_atoms_in_batch = graph_state.natoms.sum()
        # batch_indices = graph_state.batch
        
        positions = graph_state["positions"].view(batch_size, -1, 3)
        positions.requires_grad_(True)
        
        # Compute energy
        energy = self._energy(positions)
        
        # Compute forces (negative gradient of energy)
        forces = -torch.autograd.grad(
            outputs=energy.sum(), 
            inputs=positions, 
            create_graph=True, # create_graph=self.training
            retain_graph=True
        )[0]
        # [B, N, 3] -> [B * N, 3]
        forces = forces.view(num_atoms_in_batch.item(), -1)
        
        output_dict = {
            "energy": energy.detach(),
            "forces": forces.detach(),
        }

        # Add regularization forces if needed
        if regularize:
            raise NotImplementedError("Bond regularizer not implemented for custom energy model")
            # need to define bonded/unbonded state and bond distance
            reg_energy, reg_force = self.bond_regularizer(graph_state)
            output_dict["reg_forces"] = reg_force.detach()  # / self.tau
            output_dict["reg_energy"] = reg_energy.detach()
        else:
            output_dict["reg_forces"] = torch.zeros_like(output_dict["forces"])
            output_dict["reg_energy"] = torch.zeros_like(output_dict["energy"])

        output_dict["forces"] = (output_dict["forces"].detach()) / self.tau
        return output_dict

    def _remove_mean(self, x):
        # x [B, N, 3]
        # x = x.view(-1, self._n_particles, self._n_dims)
        return x - torch.mean(x, dim=1, keepdim=True)
        
    # copy of fairchem_energy.py
    def bond_regularizer(self, batch):
        """
        Compute bond structure regularization energy and gradients.
        
        This function applies a regularization term that encourages the model to 
        maintain valid chemical bonds based on the provided bond types and limits.
        
        Args:
            batch: Dictionary containing molecular graph data with fields:
                - positions: Atom coordinates
                - edge_attrs: Bond attributes (limits and types)
                - edge_index: Graph connectivity
                - ptr: Graph batch pointers
                
        Returns:
            tuple: (regularization_energy, regularization_gradients)
                - regularization_energy: Energy penalty for invalid bond structures
                - regularization_gradients: Gradients of the regularization energy
        """
        energy_reg = bond_structure_regularizer(
            batch["positions"],
            batch["edge_attrs"][:, 0].unsqueeze(-1),  # bond limits
            batch["edge_attrs"][:, 1].unsqueeze(-1),  # bond types
            batch["edge_index"],
            batch["ptr"],
            alpha=self.alpha,
        )
        grad_outputs = [torch.ones_like(energy_reg)]
        gradient = torch.autograd.grad(
            outputs=[energy_reg],  # [n_graphs, ]
            inputs=[batch["positions"]],  # [n_nodes, 3]
            grad_outputs=grad_outputs,
            retain_graph=False,  # Make sure the graph is not destroyed during training
            create_graph=False,  # Create graph for second derivative
            allow_unused=True,  # For complete dissociation turn to true
        )[0]
        return energy_reg, -gradient
    
    def score(self, samples: torch.Tensor) -> torch.Tensor:
        grad_fxn = torch.func.grad(self._energy)
        vmapped_grad = torch.vmap(grad_fxn)
        return -vmapped_grad(samples)
    
    def setup_test_set(self):
        data_np = np.load(self.data_path_test, allow_pickle=True)
        data_torch = torch.from_numpy(data_np).to(dtype=torch.float32, device=self.device)
        data = remove_mean(data_torch, self.n_particles, self.spatial_dim)
        return data

    def setup_val_set(self):
        if self.data_path_val is None:
            raise ValueError("Data path for validation data is not provided")
        data_np = np.load(self.data_path_val, allow_pickle=True)
        data_torch = torch.from_numpy(data_np).to(dtype=torch.float32, device=self.device)
        data = remove_mean(data_torch, self.n_particles, self.spatial_dim)
        return data

    def setup_train_set(self):
        if self.data_path_train is None:
            raise ValueError("Data path for training data is not provided")

        if self.data_path_train.endswith(".pt"):
            data_np = torch.load(self.data_path_train).cpu().numpy()
        else:
            data_np = np.load(self.data_path_train, allow_pickle=True)

        data_torch = torch.from_numpy(data_np).to(dtype=torch.float32, device=self.device)
        data = remove_mean(data_torch, self.n_particles, self.spatial_dim)
        return data
    
    def sample_test_set(
        self, num_points: int, normalize: bool = False, full: bool = False
    ) -> Optional[torch.Tensor]:
        if self.test_set is None:
            return None

        if full:
            outs = self.test_set
        else:
            idxs = torch.randperm(len(self.test_set))[:num_points]
            outs = self.test_set[idxs]
        if normalize:
            outs = self.normalize(outs)

        return outs

    def sample_train_set(self, num_points: int, normalize: bool = False) -> Optional[torch.Tensor]:
        if self.train_set is None:
            self._train_set = self.setup_train_set()

        idxs = torch.randperm(len(self.train_set))[:num_points]
        outs = self.train_set[idxs]
        if normalize:
            outs = self.normalize(outs)

        return outs

    def sample_val_set(self, num_points: int, normalize: bool = False) -> Optional[torch.Tensor]:
        if self.val_set is None:
            return None

        idxs = torch.randperm(len(self.val_set))[:num_points]
        outs = self.val_set[idxs]
        if normalize:
            outs = self.normalize(outs)

        return outs
    
    def interatomic_dist(self, x):
        # x: [B, N, D]
        # x = x.view(x.shape[0], self.n_particles, self.spatial_dim)

        # Compute the pairwise interatomic distances
        # removes duplicates and diagonal
        distances = x[:, None, :, :] - x[:, :, None, :]
        distances = distances[
            :,
            torch.triu(torch.ones((self.n_particles, self.n_particles)), diagonal=1) == 1,
        ]
        dist = torch.linalg.norm(distances, dim=-1)
        return dist
    
    def get_fig_samples_in_potential(self, graph_state, energies, cfg, outputs=None):
        
        atom_to_batch_indices = graph_state.batch
        batch_size = graph_state.natoms.numel()
        # n_systems = len(graph_state["ptr"]) - 1
        # Extract dimensions from graph state
        n_particles = int(len(graph_state["batch"]) // batch_size)
        n_spatial_dim = graph_state["positions"].shape[-1]
        
        # Reshape positions into (B, N, 3)
        x = graph_state["positions"].view(batch_size, n_particles, n_spatial_dim)
        
        energies = energies.detach().cpu()
        
        # for plotting
        _bs = 100
        
        # if there are only 2d particles, it's effictevily a 1d potential of the relative distance
        if self.n_particles == 2:
            fig, ax = plt.subplots()
            # get relative distances in batch [B * N, 1]
            dist = interatomic_dist(x).detach().cpu()
            # plot relative distance vs energy
            ax.scatter(dist, energies)
            
            # compute energies on 1d grid
            # place first particle along x-axis
            grid_r = torch.linspace(self.plotting_bounds_r[0], self.plotting_bounds_r[1], _bs, device=self.device)
            grid_pos_a0 = torch.stack([grid_r, torch.zeros_like(grid_r, device=self.device), torch.zeros_like(grid_r, device=self.device)], dim=-1) # [B, 3]
            # place second particle at the origin
            grid_pos_a1 = torch.zeros_like(grid_pos_a0, device=self.device)
            # combine into systems by concatenating and alternating rows
            grid_pos = torch.stack([grid_pos_a0, grid_pos_a1], dim=1) # [B, 2, 3]
            # grid_pos = grid_pos.reshape(-1, 3) # [2*B, 3]
            # compute energies
            grid_energy = self._energy(grid_pos).detach().cpu()
            ax.plot(grid_r.cpu(), grid_energy)
            ax.set_ylim(self.plotting_bounds_energy)
            
            ax.set_xlabel("Relative distance")
            ax.set_ylabel("Energy")
        
        else:
            return None
        
        # Convert figure to PIL image
        fig.canvas.draw()
        PIL_im = fig2img(fig)
        plt.close()
        return PIL_im

    def log_samples(
        self,
        samples: torch.Tensor,
        wandb_logger,
        name: str = "",
    ) -> None:
        if wandb_logger is None:
            return

        samples_fig = self.get_dataset_fig(samples)
        wandb_logger.log_image(f"{name}", [samples_fig])

    def get_dataset_fig(self, samples):
        """Energy histogram and interatomic distance histogram.
        Args:
            samples: [B, N, 3]
        Returns:
            PIL.Image
        Usage:
            samples_fig = self.get_dataset_fig(latest_samples)
            wandb_logger.log_image(f"{prefix}generated_samples", [samples_fig])
        """
        test_data_smaller = self.sample_test_set(1000)

        fig, axs = plt.subplots(1, 2, figsize=(12, 4))

        dist_samples = self.interatomic_dist(samples).detach().cpu()
        dist_test = self.interatomic_dist(test_data_smaller).detach().cpu()

        if self.n_particles == 13:
            bins = 100
        elif self.n_particles == 55:
            bins = 50

        axs[0].hist(
            dist_samples.view(-1),
            bins=bins,
            alpha=0.5,
            density=True,
            histtype="step",
            linewidth=4,
        )
        axs[0].hist(
            dist_test.view(-1),
            bins=100,
            alpha=0.5,
            density=True,
            histtype="step",
            linewidth=4,
        )
        axs[0].set_xlabel("Interatomic distance")
        axs[0].legend(["generated data", "test data"])

        energy_samples = self._energy(samples).detach().detach().cpu()
        energy_test = self._energy(test_data_smaller).detach().detach().cpu()

        # min_energy = min(energy_test.min(), energy_samples.min()).item()
        # max_energy = max(energy_test.max(), energy_samples.max()).item()
        if self.n_particles == 13:
            min_energy = -60
            max_energy = 0

        elif self.n_particles == 55:
            min_energy = -380
            max_energy = -180

        axs[1].hist(
            energy_test.cpu(),
            bins=100,
            density=True,
            alpha=0.4,
            range=(min_energy, max_energy),
            color="g",
            histtype="step",
            linewidth=4,
            label="test data",
        )
        axs[1].hist(
            energy_samples.cpu(),
            bins=100,
            density=True,
            alpha=0.4,
            range=(min_energy, max_energy),
            color="r",
            histtype="step",
            linewidth=4,
            label="generated data",
        )
        axs[1].set_xlabel("Energy")
        axs[1].legend()

        try:
            buffer = BytesIO()
            fig.savefig(buffer, format="png", bbox_inches="tight", pad_inches=0)
            buffer.seek(0)

            return PIL.Image.open(buffer)

        except Exception as e:
            fig.canvas.draw()
            return PIL.Image.frombytes(
                "RGB", fig.canvas.get_width_height(), fig.canvas.renderer.buffer_rgba()
            )

    
if __name__ == "__main__":
    n_particles = 13
    energy_model = LennardJonesEnergy(n_particles=n_particles, use_oscillator=False)
    
    B = 2
    samples = torch.randn(B, n_particles, 3, device=energy_model.device)
    print(f"samples: {samples.shape}")
    energy_vmapped = energy_model.energy_vmapped(samples)
    print(f"energy vmap: {energy_vmapped.shape}")
    
    energy = energy_model._energy(samples)
    print(f"energy: {energy.shape}")
    
    train_data = energy_model.setup_train_set()
    val_data = energy_model.setup_val_set()
    test_data = energy_model.setup_test_set()
    
    print(f"train_data: {train_data.shape}")
    print(f"val_data: {val_data.shape}")
    print(f"test_data: {test_data.shape}")
    
    # get dataset fig
    fig = energy_model.get_dataset_fig(samples)
    print(f"fig: {type(fig)}")
    