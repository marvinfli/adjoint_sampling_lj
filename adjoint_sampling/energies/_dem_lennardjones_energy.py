from io import BytesIO
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import PIL
import torch
# from bgflow import Energy
# from bgflow.utils import distance_vectors, distances_from_vectors
# from hydra.utils import get_original_cwd

from typing import Union, Optional, Sequence
from collections.abc import Sequence as _Sequence
from abc import ABC, abstractmethod
import warnings
import os

"""
Standalone copy of DEM Lennard-Jones energy function, with dependencies removed.
Used as ground truth to check our implementation of the energy function.
"""

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



# https://github.com/noegroup/bgflow/blob/main/bgflow/distribution/energy/base.py

def _is_non_empty_sequence_of_integers(x):
    return (
        isinstance(x, _Sequence) and (len(x) > 0) and all(isinstance(y, int) for y in x)
    )


def _is_sequence_of_non_empty_sequences_of_integers(x):
    return (
        isinstance(x, _Sequence)
        and len(x) > 0
        and all(_is_non_empty_sequence_of_integers(y) for y in x)
    )


def _parse_dim(dim):
    if isinstance(dim, int):
        return [torch.Size([dim])]
    if _is_non_empty_sequence_of_integers(dim):
        return [torch.Size(dim)]
    elif _is_sequence_of_non_empty_sequences_of_integers(dim):
        return list(map(torch.Size, dim))
    else:
        raise ValueError(
            f"dim must be either:"
            f"\n\t- an integer"
            f"\n\t- a non-empty list of integers"
            f"\n\t- a list with len > 1 containing non-empty lists containing integers"
        )


class BGFlowEnergy(torch.nn.Module):
    """
    Base class for all energy models.

    It supports energies defined over:
        - simple vector states of shape [..., D]
        - tensor states of shape [..., D1, D2, ..., Dn]
        - states composed of multiple tensors (x1, x2, x3, ...)
          where each xi is of form [..., D1, D2, ...., Dn]

    Each input can have multiple batch dimensions,
    so a final state could have shape
        ([B1, B2, ..., Bn, D1, D2, ..., Dn],
         ...,
         [B1, B2, ..., Bn, D'1, ..., D'1n]).

    which would return an energy tensor with shape
        ([B1, B2, ..., Bn, 1]).

    Forces are computed for each input by default.
    Here the convention is followed, that forces will have
    the same shape as the input state.

    To define the state shape, the parameter `dim` has to
    be of the following form:
        - an integer, e.g. d = 5
            then each event is a simple vector state
            of shape [..., 5]
        - a non-empty list of integers, e.g. d = [3, 6, 7]
            then each event is a tensor state of shape [..., 3, 6, 7]
        - a list of len > 1 containing non-empty integer lists,
            e.g. d = [[1, 3], [5, 3, 6]]. Then each event is
            a tuple of tensors of shape ([..., 1, 3], [..., 5, 3, 6])

    Parameters:
    -----------
    dim: Union[int, Sequence[int], Sequence[Sequence[int]]]
        The event shape of the states for which energies/forces ar computed.

    """

    def __init__(self, dim: Union[int, Sequence[int], Sequence[Sequence[int]]], **kwargs):

        super().__init__(**kwargs)
        self._event_shapes = _parse_dim(dim)

    @property
    def dim(self):
        if len(self._event_shapes) > 1:
            raise ValueError(
                "This energy instance is defined for multiple events."
                "Therefore there exists no coherent way to define the dimension of an event."
                "Consider using Energy.event_shapes instead."
            )
        elif len(self._event_shapes[0]) > 1:
            warnings.warn(
                "This Energy instance is defined on multidimensional events. "
                "Therefore, its Energy.dim is distributed over multiple tensor dimensions. "
                "Consider using Energy.event_shape instead.",
                UserWarning,
            )
        return int(torch.prod(torch.tensor(self.event_shape, dtype=int)))

    @property
    def event_shape(self):
        if len(self._event_shapes) > 1:
            raise ValueError(
                "This energy instance is defined for multiple events."
                "Therefore therefore there exists no single event shape."
                "Consider using Energy.event_shapes instead."
            )
        return self._event_shapes[0]

    @property
    def event_shapes(self):
        return self._event_shapes

    def _energy(self, *xs, **kwargs):
        raise NotImplementedError()

    def energy(self, *xs, temperature=1.0, **kwargs):
        assert len(xs) == len(
            self._event_shapes
        ), f"Expected {len(self._event_shapes)} arguments but only received {len(xs)}"
        batch_shape = xs[0].shape[: -len(self._event_shapes[0])]
        for i, (x, s) in enumerate(zip(xs, self._event_shapes)):
            assert x.shape[: -len(s)] == batch_shape, (
                f"Inconsistent batch shapes."
                f"Input at index {i} has batch shape {x.shape[:-len(s)]}"
                f"however input at index 0 has batch shape {batch_shape}."
            )
            assert (
                x.shape[-len(s) :] == s
            ), f"Input at index {i} as wrong shape {x.shape[-len(s):]} instead of {s}"
        return self._energy(*xs, **kwargs) / temperature

    def force(
        self,
        *xs: Sequence[torch.Tensor],
        temperature: float = 1.0,
        ignore_indices: Optional[Sequence[int]] = None,
        no_grad: Union[bool, Sequence[int]] = False,
        **kwargs,
    ):
        """
        Computes forces with respect to the input tensors.

        If states are tuples of tensors, it returns a tuple of forces for each input tensor.
        If states are simple tensors / vectors it returns a single forces.

        Depending on the context it might be unnecessary to compute all input forces.
        For this case `ignore_indices` denotes those input tensors for which no forces.
        are to be computed.

        E.g. by setting `ignore_indices = [1]` the result of `energy.force(x, y, z)`
        will be `(fx, None, fz)`.

        Furthermore, the forces will allow for taking high-order gradients by default.
        If this is unwanted, e.g. to save memory it can be turned off by setting `no_grad=True`.
        If higher-order gradients should be ignored for only a subset of inputs it can
        be specified by passing a list of ignore indices to `no_grad`.

        E.g. by setting `no_grad = [1]` the result of `energy.force(x, y, z)`
        will be `(fx, fy, fz)`, where `fx` and `fz` allow for taking higher order gradients
        and `fy` will not.

        Parameters:
        -----------
        xs: *torch.Tensor
            Input tensor(s)
        temperature: float
            Temperature at which to compute forces
        ignore_indices: Sequence[int]
            Which inputs should be skipped in the force computation
        no_grad: Union[bool, Sequence[int]]
            Either specifies whether higher-order gradients should be computed at all,
            or specifies which inputs to leave out when computing higher-order gradients.
        """
        if ignore_indices is None:
            ignore_indices = []

        with torch.enable_grad():
            forces = []
            requires_grad_states = [x.requires_grad for x in xs]

            for i, x in enumerate(xs):
                if i not in ignore_indices:
                    x = x.requires_grad_(True)
                else:
                    x = x.requires_grad_(False)

            energy = self.energy(*xs, temperature=temperature, **kwargs)

            for i, x in enumerate(xs):
                if i not in ignore_indices:
                    if isinstance(no_grad, bool):
                        with_grad = not no_grad
                    else:
                        with_grad = i not in no_grad
                    force = -torch.autograd.grad(
                        energy.sum(), x, create_graph=with_grad,
                    )[0]
                    forces.append(force)
                    x.requires_grad_(requires_grad_states[i])
                else:
                    forces.append(None)

        forces = (*forces,)
        if len(self._event_shapes) == 1:
            forces = forces[0]
        return forces
    

# https://github.com/jarridrb/DEM/blob/main/dem/utils/data_utils.py

def dem_remove_mean(samples, n_particles, n_dimensions):
    """Makes a configuration of many particle system mean-free.

    Parameters
    ----------
    samples : torch.Tensor
        Positions of n_particles in n_dimensions.

    Returns
    -------
    samples : torch.Tensor
        Mean-free positions of n_particles in n_dimensions.
    """
    shape = samples.shape
    if isinstance(samples, torch.Tensor):
        samples = samples.view(-1, n_particles, n_dimensions)
        samples = samples - torch.mean(samples, dim=1, keepdim=True)
        samples = samples.view(*shape)
    else:
        samples = samples.reshape(-1, n_particles, n_dimensions)
        samples = samples - samples.mean(axis=1, keepdims=True)
        samples = samples.reshape(*shape)
    return samples

# https://github.com/jarridrb/DEM/blob/main/dem/energies/base_energy_function.py

class DEMBaseEnergyFunction(ABC):
    def __init__(
        self,
        dimensionality: int,
        is_molecule: Optional[bool] = False,
        normalization_min: Optional[float] = None,
        normalization_max: Optional[float] = None,
    ):
        self._dimensionality = dimensionality

        self._test_set = self.setup_test_set()
        self._val_set = self.setup_val_set()
        self._train_set = None

        self.normalization_min = normalization_min
        self.normalization_max = normalization_max

        self._is_molecule = is_molecule

    def setup_test_set(self) -> Optional[torch.Tensor]:
        return None

    def setup_train_set(self) -> Optional[torch.Tensor]:
        return None

    def setup_val_set(self) -> Optional[torch.Tensor]:
        return None

    @property
    def _can_normalize(self) -> bool:
        return self.normalization_min is not None and self.normalization_max is not None

    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        if x is None or not self._can_normalize:
            return x

        mins = self.normalization_min
        maxs = self.normalization_max

        # [ 0, 1 ]
        x = (x - mins) / (maxs - mins + 1e-5)
        # [ -1, 1 ]
        return x * 2 - 1

    def unnormalize(self, x: torch.Tensor) -> torch.Tensor:
        if x is None or not self._can_normalize:
            return x

        mins = self.normalization_min
        maxs = self.normalization_max

        x = (x + 1) / 2
        return x * (maxs - mins) + mins

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

    @property
    def dimensionality(self) -> int:
        return self._dimensionality

    @property
    def is_molecule(self) -> Optional[bool]:
        return self._is_molecule

    @property
    def test_set(self) -> Optional[torch.Tensor]:
        return self._test_set

    @property
    def val_set(self) -> Optional[torch.Tensor]:
        return self._val_set

    @property
    def train_set(self) -> Optional[torch.Tensor]:
        return self._train_set

    @abstractmethod
    def __call__(self, samples: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def score(self, samples: torch.Tensor) -> torch.Tensor:
        grad_fxn = torch.func.grad(self.__call__)
        vmapped_grad = torch.vmap(grad_fxn)
        return vmapped_grad(samples)


# https://github.com/jarridrb/DEM/blob/main/dem/energies/lennardjones_energy.py

def sample_from_array(array, size):
    idx = np.random.choice(array.shape[0], size=size)
    return array[idx]


def lennard_jones_energy_torch(r, eps=1.0, rm=1.0):
    p = 0.9
    lj = eps * ((rm / r) ** 12 - 2 * (rm / r) ** 6)
    return lj


class BGFlowLennardJonesPotential(BGFlowEnergy):
    def __init__(
        self,
        dim,
        n_particles,
        eps=1.0,
        rm=1.0,
        oscillator=True,
        oscillator_scale=1.0,
        two_event_dims=True,
        energy_factor=1.0,
    ):
        """Energy for a Lennard-Jones cluster.

        Parameters
        ----------
        dim : int
            Number of degrees of freedom ( = space dimension x n_particles)
        n_particles : int
            Number of Lennard-Jones particles
        eps : float
            LJ well depth epsilon
        rm : float
            LJ well radius R_min
        oscillator : bool
            Whether to use a harmonic oscillator as an external force
        oscillator_scale : float
            Force constant of the harmonic oscillator energy
        two_event_dims : bool
            If True, the energy expects inputs with two event dimensions (particle_id, coordinate).
            Else, use only one event dimension.
        """
        if two_event_dims:
            super().__init__([n_particles, dim // n_particles])
        else:
            super().__init__(dim)
        self._n_particles = n_particles
        self._n_dims = dim // n_particles

        self._eps = eps
        self._rm = rm
        self.oscillator = oscillator
        self._oscillator_scale = oscillator_scale

        # this is to match the eacf energy with the eq-fm energy
        # for lj13, to match the eacf set energy_factor=0.5
        self._energy_factor = energy_factor

    def _energy(self, x):
        """
        Compute energy for a batch of systems.
        Args:
            positions (torch.Tensor): Coordinates of shape [B, N*D]
            where N is the number of particles and D is the spatial dimension
            
        Returns:
            torch.Tensor: Energy value [B, 1]
        """
        batch_shape = x.shape[: -len(self.event_shape)] # B
        x = x.view(*batch_shape, self._n_particles, self._n_dims) # [B, N, D]

        dists = distances_from_vectors(
            distance_vectors(
                # x.view(-1, self._n_particles, self._n_dims) # [B, N, D]
                x
            ) # [B, N, N-1, 3]
        ) # [B, N, N-1]

        lj_energies = lennard_jones_energy_torch(dists, self._eps, self._rm)
        # lj_energies = torch.clip(lj_energies, -1e4, 1e4)
        # [B, N, N-1] -> [B]
        lj_energies = lj_energies.view(*batch_shape, -1).sum(dim=-1) * self._energy_factor

        if self.oscillator:
            osc_energies = 0.5 * self._remove_mean(x).pow(2).sum(dim=(-2, -1)).view(*batch_shape) # [B]
            lj_energies = lj_energies + osc_energies * self._oscillator_scale

        return lj_energies[:, None] # [B, 1]

    def _remove_mean(self, x):
        x = x.view(-1, self._n_particles, self._n_dims)
        return x - torch.mean(x, dim=1, keepdim=True)

    def _energy_numpy(self, x):
        x = torch.Tensor(x)
        return self._energy(x).cpu().numpy()

    def _log_prob(self, x):
        return -self._energy(x)

class DEMLennardJonesEnergy(DEMBaseEnergyFunction):
    def __init__(
        self,
        dimensionality,
        n_particles,
        data_path=None,
        data_path_train=None,
        data_path_val=None,
        data_path_test=None,
        device="cpu",
        plot_samples_epoch_period=5,
        plotting_buffer_sample_size=512,
        data_normalization_factor=1.0,
        energy_factor=1.0,
        is_molecule=True,
        use_oscillator=True,
        oscillator_scale=1.0,
    ):
        self.n_particles = n_particles
        self.n_spatial_dim = dimensionality // n_particles

        if self.n_particles != 13 and self.n_particles != 55:
            raise NotImplementedError

        self.curr_epoch = 0
        self.plotting_buffer_sample_size = plotting_buffer_sample_size
        self.plot_samples_epoch_period = plot_samples_epoch_period

        self.data_normalization_factor = data_normalization_factor

        # self.data_path = data_path
        # self.data_path_train = data_path_train
        # self.data_path_val = data_path_val
        # self.data_path_test = data_path_test

        # self.data_path = get_original_cwd() + "/" + data_path
        # self.data_path_train = get_original_cwd() + "/" + data_path_train
        # self.data_path_val = get_original_cwd() + "/" + data_path_val
        
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

        if self.n_particles == 13:
            self.name = "LJ13_efm"
        elif self.n_particles == 55:
            self.name = "LJ55"

        self.device = device

        self.lennard_jones = BGFlowLennardJonesPotential(
            dim=dimensionality,
            n_particles=n_particles,
            eps=1.0,
            rm=1.0,
            oscillator=use_oscillator,
            oscillator_scale=oscillator_scale,
            two_event_dims=False,
            energy_factor=energy_factor,
        )

        super().__init__(dimensionality=dimensionality, is_molecule=is_molecule)

    def __call__(self, samples: torch.Tensor) -> torch.Tensor:
        return self.lennard_jones._log_prob(samples).squeeze(-1)

    def setup_test_set(self):
        data = np.load(self.data_path_test, allow_pickle=True)
        data = dem_remove_mean(data, self.n_particles, self.n_spatial_dim)
        data = torch.tensor(data, device=self.device)
        return data

    def setup_val_set(self):
        if self.data_path_val is None:
            raise ValueError("Data path for validation data is not provided")
        data = np.load(self.data_path_val, allow_pickle=True)
        data = dem_remove_mean(data, self.n_particles, self.n_spatial_dim)
        data = torch.tensor(data, device=self.device)
        return data

    def setup_train_set(self):
        if self.data_path_train is None:
            raise ValueError("Data path for training data is not provided")

        if self.data_path_train.endswith(".pt"):
            data = torch.load(self.data_path_train).cpu().numpy()
        else:
            data = np.load(self.data_path_train, allow_pickle=True)

        data = dem_remove_mean(data, self.n_particles, self.n_spatial_dim)
        data = torch.tensor(data, device=self.device)
        return data

    def interatomic_dist(self, x):
        batch_shape = x.shape[: -len(self.lennard_jones.event_shape)]
        x = x.view(*batch_shape, self.n_particles, self.n_spatial_dim)

        # Compute the pairwise interatomic distances
        # removes duplicates and diagonal
        distances = x[:, None, :, :] - x[:, :, None, :]
        distances = distances[
            :,
            torch.triu(torch.ones((self.n_particles, self.n_particles)), diagonal=1) == 1,
        ]
        dist = torch.linalg.norm(distances, dim=-1)
        return dist

    def log_on_epoch_end(
        self,
        latest_samples: torch.Tensor,
        latest_energies: torch.Tensor,
        wandb_logger,
        unprioritized_buffer_samples=None,
        cfm_samples=None,
        replay_buffer=None,
        prefix: str = "",
    ) -> None:
        if latest_samples is None:
            return

        if wandb_logger is None:
            return

        if len(prefix) > 0 and prefix[-1] != "/":
            prefix += "/"

        if self.curr_epoch % self.plot_samples_epoch_period == 0:
            samples_fig = self.get_dataset_fig(latest_samples)

            wandb_logger.log_image(f"{prefix}generated_samples", [samples_fig])

            if cfm_samples is not None:
                cfm_samples_fig = self.get_dataset_fig(cfm_samples)

                wandb_logger.log_image(f"{prefix}cfm_generated_samples", [cfm_samples_fig])

        self.curr_epoch += 1

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

        energy_samples = -self(samples).detach().detach().cpu()
        energy_test = -self(test_data_smaller).detach().detach().cpu()

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
    # test the energy function
    # https://github.com/jarridrb/DEM/blob/main/configs/energy/lj13.yaml
    # dimensionality: 39
    # n_particles: 13
    # data_normalization_factor: 1.0
    energy = DEMLennardJonesEnergy(dimensionality=13*3, n_particles=13, use_oscillator=False)
    
    B = 2
    N = 13
    D = 3
    x = torch.randn(B, N, D)
    
    x = x.reshape(B, N*D)
    print("x", x.shape)
    print("energy(x)", energy(x).shape)
    
    # energy with vmap
    # DEM used vmap to parallelize over Monte Carlo samples 
    # x = x.reshape(B*N, D)
    x = x.reshape(B, N*D).unsqueeze(0)
    print("x", x.shape)
    energy_vmap = torch.vmap(energy)
    print("energy_vmap(x)", energy_vmap(x).shape)
    
    