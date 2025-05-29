import torch
from torch_geometric.data import Data, Dataset
from adjoint_sampling.utils.data_utils import get_atomic_graph



class IdenticalParticlesDataset(Dataset):
    """
    Dataset for arbitrary coordinate-based systems without molecular connectivity.
    """
    
    def __init__(self, energy_model, num_particles, num_samples=100, spatial_dim=3, duplicate=1):
        """
        Initialize the custom dataset.
        
        Args:
            energy_model: Model to compute energies/forces
            num_particles (int): Number of particles in each system
            duplicate (int): Number of samples in the dataset
            spatial_dim (int): Dimensionality of the space (usually 3)
            duplicate (int): Number of duplicate copies (for distributed training)
        """
        super().__init__()
        self.energy_model = energy_model
        self.num_particles = num_particles
        self.num_samples = num_samples
        self.spatial_dim = spatial_dim
        self.dim = spatial_dim * num_particles
        
    def len(self):
        """Return the number of samples in the dataset."""
        return self.num_samples 
        
    def get(self, idx):
        """
        Get a sample from the dataset.
        
        Args:
            idx (int): Index of the sample
            
        Returns:
            Data: PyTorch Geometric data object with system information
        """
        # see datasets.get_spice_dataset
        
        # Set zero initial positions
        # positions = torch.randn(self.num_particles, self.spatial_dim)
        positions = torch.zeros(self.num_particles, self.spatial_dim)
        
        # Create a PyG Data object with the necessary attributes
        # see utils.data_utils.get_atomic_graph
        # atomic_number_table = self.energy_model.atomic_numbers 
        atomic_number_table = torch.arange(100)
        atomic_index_table = {int(z): i for i, z in enumerate(atomic_number_table)}
        
        data: Data = get_atomic_graph(
            atom_list=[0] * self.num_particles,
            positions=positions,
            z_table=atomic_index_table,
        )
        
        # see datasets.process_smiles
        edge_index = data.edge_index
        
        length_matrix = torch.full((self.num_particles, self.num_particles), fill_value=0.0)
        length_attr = length_matrix[edge_index[0], edge_index[1]].unsqueeze(-1)
        type_matrix = torch.full((self.num_particles, self.num_particles), fill_value=0.0)
        type_attr = type_matrix[edge_index[0], edge_index[1]].unsqueeze(-1)
        edge_attr = torch.cat([length_attr, type_attr], dim=-1).float()
        data["edge_attrs"] = edge_attr
        
        data["smiles"] = "H" * self.num_particles
        
        return data
    
    def _create_edges(self, num_nodes):
        rows, cols = [], []
        for i in range(num_nodes):
            for j in range(i + 1, num_nodes):
                rows.append(i)
                cols.append(j)
                rows.append(j)
                cols.append(i)
        # return [torch.LongTensor(rows), torch.LongTensor(cols)]
        return torch.tensor([rows, cols], dtype=torch.long)
        
def get_identical_particles_dataset(
    energy_model, num_particles, spatial_dim=3, num_samples=100, 
    # taken from fairchem.core.models.esen.esen.py
    max_atom_types=100
):
    # Set zero initial positions
    # positions = torch.randn(self.num_particles, self.spatial_dim)
    positions = torch.zeros(num_particles, spatial_dim)
    
    # Create a PyG Data object with the necessary attributes
    # see utils.data_utils.get_atomic_graph
    # atomic_number_table = self.energy_model.atomic_numbers 
    atomic_number_table = torch.arange(max_atom_types)
    atomic_index_table = {int(z): i for i, z in enumerate(atomic_number_table)}
    
    data: Data = get_atomic_graph(
        atom_list=[0] * num_particles,
        positions=positions,
        z_table=atomic_index_table,
    )
    
    # see datasets.process_smiles
    edge_index = data.edge_index
    
    # uniform unbonded edges
    length_matrix = torch.full((num_particles, num_particles), fill_value=0.0)
    length_attr = length_matrix[edge_index[0], edge_index[1]].unsqueeze(-1)
    type_matrix = torch.full((num_particles, num_particles), fill_value=0.0)
    type_attr = type_matrix[edge_index[0], edge_index[1]].unsqueeze(-1)
    edge_attr = torch.cat([length_attr, type_attr], dim=-1).float()
    data["edge_attrs"] = edge_attr
    
    # just a placeholder
    data["smiles"] = "H" * num_particles
    
    # dataset of duplicate copies
    dataset = []
    for _ in range(num_samples):
        dataset.append(data.clone())
        
    return dataset

