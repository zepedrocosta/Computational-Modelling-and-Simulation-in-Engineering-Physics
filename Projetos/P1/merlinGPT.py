# This program extends the previous code by adding a plot using Matplotlib to visualize how the total 
# energy evolves over the Monte Carlo cycles during the simulation of the Ising model on a 3D cubic grid. 
# Further development and analysis are needed to explore additional physical quantities and investigate the system behavior in more detail.

import numpy as np
import random
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


# Function to initialize a 3D grid with random spin values
def initialize_grid(size):
    return np.random.choice([-1, 1], size=(size, size, size))


# Function to calculate the energy of a specific spin site
def calculate_energy(grid, x, y, z):
    energy = 0
    # Calculate interactions with nearest neighbors in a periodic boundary condition
    for dx, dy, dz in [
        (-1, 0, 0),
        (1, 0, 0),
        (0, -1, 0),
        (0, 1, 0),
        (0, 0, -1),
        (0, 0, 1),
    ]:
        energy += (
            grid[x, y, z] * grid[(x + dx) % size, (y + dy) % size, (z + dz) % size]
        )
    return -energy


# Function to perform Monte Carlo simulation for a given temperature
def monte_carlo(grid, temperature, mc_cycles):
    energies = []
    for _ in range(mc_cycles):
        x, y, z = (
            random.randint(0, size - 1),
            random.randint(0, size - 1),
            random.randint(0, size - 1),
        )
        current_energy = calculate_energy(grid, x, y, z)

        grid[x, y, z] *= -1
        new_energy = calculate_energy(grid, x, y, z)
        delta_energy = new_energy - current_energy

        if delta_energy > 0 and random.random() > np.exp(-delta_energy / temperature):
            grid[x, y, z] *= -1

        energies.append(np.sum(grid))

    return energies


# Main simulation parameters
size = 10
temperature = 1.0
mc_cycles = 10000

# Initialize the grid
grid = initialize_grid(size)

# Perform Monte Carlo simulation and collect energies
energies = monte_carlo(grid, temperature, mc_cycles)

# Plot the energy evolution over MC cycles
plt.figure()
plt.plot(range(mc_cycles), energies)
plt.xlabel("MC Cycles")
plt.ylabel("Total Energy")
plt.title("Total Energy vs. MC Cycles")
plt.show()

# Further code implementation is necessary to analyze the results and physical quantities.
