# Copyright (c) Meta Platforms, Inc. and affiliates.

import torch
# from fairchem.core.common.utils import conditional_grad

import matplotlib.pyplot as plt
import PIL

import torch_geometric.utils as tg_utils
from torch_geometric.utils import unbatch_edge_index, unbatch, sort_edge_index

from typing import Dict, Optional

from adjoint_sampling.energies.regularizers import bond_structure_regularizer
from adjoint_sampling.utils.visualize_utils import fig2img, interatomic_dist

def lennard_jones_energy_torch(r, eps=1.0, rm=1.0, tau=1.0):
    # p = 0.9
    lj = eps * ((rm / r)**12 - 2 * (rm / r)**6) # * 0.5 * tau
    return lj

class LennardJonesEnergy(torch.nn.Module):
    """
    Custom energy model for non-molecular systems.
    
    This model computes energies and forces for arbitrary coordinate-based systems
    without requiring molecular connectivity or SMILES strings.
    """
    
    def __init__(
        self, 
        num_particles, 
        spatial_dim=3,
        tau=1.0, 
        alpha=0.0, 
        x0=5.0, 
        device="cuda", 
        default_regularize=False, 
        use_oscillator=True, 
        oscillator_scale=0.25,
    ):
        """
        Initialize the custom energy model.
        
        Args:
            tau (float): Temperature parameter
            alpha (float): Regularization parameter
            device (str): Device to run model on
        """
        super().__init__()
        self.num_particles = num_particles
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
        
        # vmapped functions
        self.energy_vmapped = torch.vmap(self._energy_vmap)
        
    
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
        return lj_energies
    
    def _energy(self, positions):
        """
        Compute energy for a batch of systems.
        Args:
            positions (torch.Tensor): Coordinates of shape [B, N, 3]
            
        Returns:
            torch.Tensor: Energy value [B, 1]
        """
        # [B, 2, 1]
        x = positions
        n_particles = x.shape[1]
        dists = torch.cdist(x, x)
        # remove_duplicates
        dists = dists[:, torch.triu(torch.ones((n_particles, n_particles)), diagonal=1) == 1]
        dists = dists.reshape(-1, n_particles * (n_particles - 1) // 2)
        # compute lennard jones energy
        lj = ((1.0 / dists)**12 - 2 * (1.0 / dists)**6) 
        # sum over relative distances
        lj = lj.sum(dim=1)
        if self.use_oscillator:
            # [B, N, 3] -> [B]
            x_centered = (x - torch.mean(x, dim=1, keepdim=True))
            osc_energies = 0.5 * x_centered.pow(2).sum(dim=(-2, -1))
            lj = lj + osc_energies * self.oscillator_scale
        return lj
    
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
        if self.num_particles == 2:
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
    


    
if __name__ == "__main__":
    num_particles = 8
    energy_model = LennardJonesEnergy(num_particles=num_particles)
    samples = torch.randn(256, num_particles, 3, device=energy_model.device)
    energy = energy_model.energy_vmapped(samples)
    print(f"energy: {energy.shape}")
    