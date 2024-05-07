import time
import numpy as np
import random
import matplotlib.pyplot as plt
import math


# Function to initialize a 3D grid with all spins up
def initialize_grid(size):
    return np.full((size, size, size), 1, dtype="int")


# Function to calculate the energy of a specific spin site considering the external magnetic field
def calculate_energy(grid, x, y, z, h):
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


def transitionFunctionValues(t, h):

    deltaE = [[j + i * h for i in range(-1, 3, 2)] for j in range(-4, 6, 2)]
    output = [
        [1 if elem <= 0 else np.exp(-2 * elem / t) for elem in row] for row in deltaE
    ]

    return np.array(output)


def w(sigma, sigSoma, valuesW):

    i = int(sigSoma / 2 + 2)
    j = int(sigma / 2 + 1 / 2)

    return valuesW[i, j]


# Function to perform Monte Carlo simulation for a given temperature without magnetic fields for energy tracking
def monte_carlo_energy(grid, temperature, mc_cycles):
    energies = np.array([])
    for _ in range(mc_cycles):
        x, y, z = (
            random.randint(0, size - 1),
            random.randint(0, size - 1),
            random.randint(0, size - 1),
        )
        current_energy = calculate_energy(grid, x, y, z, 0)
        grid[x, y, z] *= -1  # Flip the spin
        new_energy = calculate_energy(grid, x, y, z, 0)
        delta_energy = new_energy - current_energy

        if delta_energy < 0 or random.random() < math.exp(-delta_energy / temperature):
            current_energy = new_energy
        else:
            grid[x, y, z] *= -1  # Flip back if not accepted

        # grid[x, y, z] *= w(delta_energy, grid[x, y, z], 0, temperature)

        energies = np.append(energies, np.sum(grid))
    return energies


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


def simulacao_temp(grid, temps, size, mc_cycles):
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
    m = np.array([])
    sus = np.array([])
    e = np.array([])
    c = np.array([])
    grid = initialize_grid(size)
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
            current_energy = calculate_energy(grid, x, y, z, 0)
            grid[x, y, z] *= -1
            new_energy = calculate_energy(grid, x, y, z, 0)
            delta_energy = new_energy - current_energy

            # grid[x, y, z] *= w(delta_energy, grid[x, y, z], 0, temperature)
            if delta_energy < 0 or random.random() < math.exp(
                -delta_energy / temperature
            ):
                current_energy = new_energy
            else:
                grid[x, y, z] *= -1  # Flip back if not accepted

            M += grid / grid.size  # momento magnético médio ???
            E += np.sum(grid) / grid.size
            M2 += np.linalg.norm(M)
            E2 += E**2
        m = np.append(m, momento_magnetico_medio(M, size))
        e = np.append(e, energia_media_por_ponto_de_rede(E, size))
        sus = np.append(sus, susceptibilidade_magnetica(M2, t, size))
        c = np.append(c, capacidade_calorifica(E2, t, size))
    return m, sus, e, c


# Function for Monte Carlo Simulations including a varying magnetic field for hysteresis
def monte_carlo_hysteresis(grid, temperature, mc_cycles, h_values):
    magnetizations = np.array([])
    grid = initialize_grid(size)
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
        # magnetizations /= size**3
        magnetizations = np.append(magnetizations, M)
    return magnetizations


# Plot function for original energy plot
def plot_energy(mc_cycles, energies):
    plt.figure()
    plt.plot(range(mc_cycles), energies)
    plt.xlabel("Monte Carlo Cycles")
    plt.ylabel("Total Energy")
    plt.title("Total Energy vs. MC Cycles")
    plt.grid(True)
    plt.show()


# Function to plot the ferromagnetic properties
def plot_ferro_graf(m, sus, e, c, temps):
    fig, axs = plt.subplots(2, 2)
    axs[0, 0].plot(temps, m)
    axs[0, 0].set_title("m vs t")
    plt.ticklabel_format(style = 'plain')
    axs[0, 1].plot(temps, sus)
    axs[0, 1].set_title("χ vs t")
    plt.ticklabel_format(style = 'plain')
    axs[1, 0].plot(temps, e)
    axs[1, 0].set_title("e vs t")
    plt.ticklabel_format(style = 'plain')
    axs[1, 1].plot(temps, c)
    axs[1, 1].set_title("C vs t")
    plt.ticklabel_format(style = 'plain')
    fig.tight_layout()
    plt.show()


# Function to plot the hysteresis loop
def plot_hysteresis(temperature, h_values, magnetizations):
    plt.figure(figsize=(8, 6))
    plt.plot(h_values, magnetizations)
    plt.title(f"Hysteresis Loop at T = {temperature}")
    plt.xlabel("External Magnetic Field (h)")
    plt.ylabel("Magnetization (M)")
    plt.grid(True)
    plt.ticklabel_format(style = 'plain')
    plt.show()


def estimate_curie_temperature(temps, m, sus, e, c):
    max_sus = max(sus)
    temps[sus == max_sus]

    max_c = max(c)
    temps[c == max_c]

    output = np.array([temps, m, sus, e, c]).transpose()
    np.savetxt("results.tsv", output, delimiter="\t")

    data = np.loadtxt("results.tsv", delimiter="\t")

    print(data)
    print(f"Curie temperature estimate: {temps[c == max_c]}")
    print(f"Curie temperature estimate: {temps[sus == max_sus]}")


# Main simulation parameters
size = 10
temperature = 5.5  # Near the critical temperature for visualization
mc_cycles = 10000

# h values
h_max = 4  # Maximum strength of magnetic field
h_values = np.concatenate(
    [np.linspace(-h_max, h_max, 10), np.linspace(h_max, -h_max, 10)]
)

valuesW = transitionFunctionValues(temperature, h_max)

# Initialize the grid
grid = initialize_grid(size)

# Energy tracking
start = time.time()
energies = monte_carlo_energy(grid, temperature, mc_cycles)
end = time.time()
elapsedTime = end - start
print(f"Tempo gasto a calcular a energia: {elapsedTime}")
plot_energy(mc_cycles, energies)

# Ferromagnetic properties
start = time.time()
temps = np.arange(0.5, 5.5, 0.1)
m, sus, e, c = simulacao_temp(grid, temps, 10, 1000)
end = time.time()
elapsedTime = end - start
print(f"Tempo gasto a calcular as propriedades ferromagnéticas: {elapsedTime}")
plot_ferro_graf(m, sus, e, c, temps)

# Estimating the Curie temperature
estimate_curie_temperature(temps, m, sus, e, c)

temperatures = np.array([0.5, 2, 3.5, 4, 5.3])

# Hysteresis tracking
start = time.time()
magnetizations = monte_carlo_hysteresis(grid, temperature, mc_cycles, h_values)
end = time.time()
elapsedTime = end - start
print(f"Tempo gasto a calcular a Paramagnetic hysteresis: {elapsedTime}")
plot_hysteresis(temperature, h_values, magnetizations)
