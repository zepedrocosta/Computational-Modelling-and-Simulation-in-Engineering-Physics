import numpy as np
import random
import matplotlib.pyplot as plt
import math


# Function to initialize a 3D grid with all spins down
def initialize_grid(size):
    return np.full((size, size, size), 1, dtype="int")


# Function to calculate the energy of a specific spin site considering the external magnetic field
def calculate_energy(grid, x, y, z, h=0):
    J = 1  # Exchange interaction strength
    sum_neighbors = 0
    for dx, dy, dz in [
        (1, 0, 0),
        (-1, 0, 0),
        (0, 1, 0),
        (0, -1, 0),
        (0, 0, 1),
        (0, 0, -1),
    ]:
        sum_neighbors += grid[(x + dx) % size, (y + dy) % size, (z + dz) % size]
    return -J * grid[x, y, z] * sum_neighbors - h * grid[x, y, z]


def w(delta_epsilon, sigma_i_inicial, h, t):
    if delta_epsilon < 0:
        return 1
    else:
        return math.exp(-1 * (delta_epsilon + sigma_i_inicial * h) / t)


# Function to perform Monte Carlo simulation for a given temperature without magnetic fields for energy tracking
def monte_carlo_energy(grid, temperature, mc_cycles):
    energies = []
    for _ in range(mc_cycles):
        x, y, z = (
            random.randint(0, size - 1),
            random.randint(0, size - 1),
            random.randint(0, size - 1),
        )
        current_energy = calculate_energy(grid, x, y, z)
        grid[x, y, z] *= -1  # Flip the spin
        new_energy = calculate_energy(grid, x, y, z)
        delta_energy = new_energy - current_energy

        if delta_energy < 0 or random.random() < math.exp(-delta_energy / temperature):
            current_energy = new_energy
        else:
            grid[x, y, z] *= -1  # Flip back if not accepted

        # grid[x, y, z] *= w(delta_energy, grid[x, y, z], 0, temperature)

        energies.append(np.sum(grid))
    return energies


# Plot function for original energy plot
def plot_energy(mc_cycles, energies):
    plt.figure()
    plt.plot(range(mc_cycles), energies)
    plt.xlabel("Monte Carlo Cycles")
    plt.ylabel("Total Energy")
    plt.title("Total Energy vs. MC Cycles")
    plt.grid(True)
    plt.show()


def momento_magnetico_medio(M, L):
    """
    Calcula o momento magnético médio da rede.

    Args:
    - M: Momento magnético médio da rede
    - L: Tamanho da rede

    Returns:
    - m: Momento magnético médio da rede com o fator de ordem
    """
    m = np.linalg.norm(M) / (L**2)
    return m


def energia_media_por_ponto_de_rede(E, L):
    """
    Calcula a energia média por ponto de rede.

    Args:
    - E: Energia média por ponto de rede
    - L: Tamanho da rede

    Returns:
    - epsilon: Energia média por ponto de rede
    """
    epsilon = E / (L**2)
    return epsilon


def susceptibilidade_magnetica(sigma_M, t, L):
    """
    Calcula a susceptibilidade magnética.

    Args:
    - sigma_M: Variância do momento magnético
    - t: Temperatura
    - L: Tamanho da rede

    Returns:
    - chi: Susceptibilidade magnética
    """
    chi = ((sigma_M**2) / t) * (L**2)
    return chi


def capacidade_calorifica(sigma_epsilon, t, L):
    """
    Calcula a capacidade calorífica.

    Args:
    - sigma_epsilon: Variância da energia
    - t: Temperatura
    - L: Tamanho da rede

    Returns:
    - C: Capacidade calorífica
    """
    C = (sigma_epsilon**2) / (t**2 * L**2)
    return C


def simulacao_temp(grid, temps, size, mc_cycles, h):
    """
    Simula o modelo de Ising em uma rede 3D.

    Args:
    - temps: Lista de temperaturas
    - size: Tamanho da rede
    - mc_cycles: Número de ciclos de Monte Carlo
    - h: Campo magnético externo

    Returns:
    - m: Lista de momentos magnéticos médios
    - sus: Lista de susceptibilidades magnéticas
    - e: Lista de energias médias
    - c: Lista de capacidades caloríficas
    """
    m = []
    sus = []
    e = []
    c = []
    for t in temps:
        M = np.zeros((size, size, size))
        E = 0
        M2 = 0
        E2 = 0
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
            grid[x, y, z] *= w(delta_energy, grid[x, y, z], 0, temperature)
            M += grid / grid.size  # momento magnético médio ???
            E += np.sum(grid) / grid.size
            M2 += np.linalg.norm(M)
            E2 += E**2
        m.append(momento_magnetico_medio(M, size))
        sus.append(susceptibilidade_magnetica(M2, t, size))
        e.append(energia_media_por_ponto_de_rede(E, size))
        c.append(capacidade_calorifica(E2, t, size))
    return m, sus, e, c


def ferro_graf(m, sus, e, c, temps):
    fig, axs = plt.subplots(2, 2)
    axs[0, 0].plot(temps, m)
    axs[0, 0].set_title("m vs t")
    axs[0, 1].plot(temps, sus)
    axs[0, 1].set_title("χ vs t")
    axs[1, 0].plot(temps, e)
    axs[1, 0].set_title("e vs t")
    axs[1, 1].plot(temps, c)
    axs[1, 1].set_title("C vs t")
    fig.tight_layout()
    return fig


# Function for Monte Carlo Simulations including a varying magnetic field for hysteresis
def monte_carlo_hysteresis(grid, temperature, mc_cycles, h_values):
    magnetizations = []
    for h in h_values:
        M = 0
        for _ in range(mc_cycles):
            x, y, z = (
                random.randint(0, size - 1),
                random.randint(0, size - 1),
                random.randint(0, size - 1),
            )
            current_energy = calculate_energy(grid, x, y, z, h)
            grid[x, y, z] *= -1  # Flip the spin
            new_energy = calculate_energy(grid, x, y, z, h)
            delta_energy = new_energy - current_energy

            if delta_energy < 0 or random.random() < math.exp(
                -delta_energy / temperature
            ):
                current_energy = new_energy
            else:
                grid[x, y, z] *= -1  # Flip back if not accepted

            M += np.sum(grid)
        magnetizations.append(M / size**3)  # Normalized magnetization
    return magnetizations


# Function to plot the hysteresis loop
def plot_hysteresis(grid, temperature, size, mc_cycles):
    h_max = 4  # Maximum strength of magnetic field
    h_values = np.concatenate(
        [np.linspace(h_max, -h_max, 50)]
    )

    magnetizations = monte_carlo_hysteresis(grid, temperature, mc_cycles, h_values)

    plt.figure(figsize=(8, 6))
    plt.plot(h_values, magnetizations)
    plt.title(f"Hysteresis Loop at T = {temperature}")
    plt.xlabel("External Magnetic Field (h)")
    plt.ylabel("Magnetization (M)")
    plt.grid(True)
    plt.show()


# Main simulation parameters
size = 10
temperature = 5.5  # Near the critical temperature for visualization
mc_cycles = 10000

# Initialize the grid
grid = initialize_grid(size)

# Energy tracking
energies = monte_carlo_energy(grid, temperature, mc_cycles)
plot_energy(mc_cycles, energies)

temps = np.arange(0.5, 5.5, 0.1)
m, sus, e, c = simulacao_temp(grid, temps, 10, 1000, 0)
fig1 = ferro_graf(m, sus, e, c, temps)
plt.show()

# max_sus = max(sus)
# temps[sus == max_sus]

# Hysteresis plot
plot_hysteresis(grid, temperature, size, mc_cycles)
