import torch

from adjoint_sampling.energies._dem_lennardjones_energy import DEMLennardJonesEnergy, distances_from_vectors, distance_vectors
from adjoint_sampling.energies.lennardjones_energy import LennardJonesEnergy

"""
Test that our Lennard-Jones energy function matches the DEM implementation.
"""


if __name__ == "__main__":
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    B = 4
    N = 13
    D = 3
    x = torch.randn(B, N, D, device=device)
    use_oscillator = True
    
    # test the energy function
    # https://github.com/jarridrb/DEM/blob/main/configs/energy/lj13.yaml
    # dimensionality: 39
    # n_particles: 13
    # data_normalization_factor: 1.0
    dem_energy_model = DEMLennardJonesEnergy(dimensionality=N*D, n_particles=N, use_oscillator=use_oscillator, device=device)
    
    print(" ")
    
    x = x.reshape(B, N*D)
    energy_dem = -dem_energy_model(x) # DEM energy is log-prob
    print("energy DEM", energy_dem.shape)
    
    self = dem_energy_model.lennard_jones
    
    # energy with vmap
    # DEM used vmap to parallelize over Monte Carlo samples 
    # x = x.reshape(B*N, D)
    # x = x.reshape(B, N*D).unsqueeze(0)
    # energy_vmapped_dem = -torch.vmap(dem_energy_model)(x) # DEM energy is log-prob
    # print("energy_vmap DEM", energy_vmapped_dem.shape)
    # energy_vmapped_dem = energy_vmapped_dem[0]
    
    #######################################################################################################
    print(" ")
    
    x = x.reshape(B, N, D)
    
    # test our implementation of the energy function
    energy_model = LennardJonesEnergy(num_particles=N, use_oscillator=use_oscillator, device=device)
    
    energy = energy_model._energy(x)
    print(f"energy: {energy.shape}")
    
    energy_vmapped = energy_model.energy_vmapped(x)
    print(f"energy vmap: {energy_vmapped.shape}")
    
    

    # Compare our implementation energy with vmap
    print("\nComparing our implementation energy with vmap:")
    print(f"Our energy: {energy}")
    print(f"Our energy vmap: {energy_vmapped}")
    print(f"Max absolute difference: {torch.max(torch.abs(energy - energy_vmapped))}")

    # Compare DEM energy with our implementation
    print("\nComparing DEM energy with our implementation:")
    print(f"DEM energy: {energy_dem}")
    print(f"Our energy: {energy}")
    print(f"Max absolute difference: {torch.max(torch.abs(energy_dem - energy))}")