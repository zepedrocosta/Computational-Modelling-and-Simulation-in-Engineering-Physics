import time
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def transitionFunctionValues(t, h=0):
    # Initialize the 3D list for deltaE
    deltaE = []
    # Iterate over the range for the 'k' coordinate
    for k in range(-7, 9, 2):  # Adjust the range as needed for the 3D grid
        plane = []
        # Iterate over the range for the 'j' coordinate
        for j in range(-4, 6, 2):
            line = []
            # Iterate over the range for the 'i' coordinate
            for i in range(-1, 3, 2):
                # Calculate and append new element changed by delta i, j, and k
                line.append(j + i * h + k * h)
            plane.append(line)
        deltaE.append(plane)

    # Compute the output based on deltaE values
    output = []
    for plane in deltaE:
        output_plane = []
        for row in plane:
            inner_list = []
            for elem in row:
                # Conditional computation based on the element value
                if elem <= 0:
                    inner_list.append(1)
                else:
                    inner_list.append(np.exp(-2 * elem / t))
            output_plane.append(inner_list)
        output.append(output_plane)

    # Convert output list into a 3D numpy array
    return np.array(output)


def w(sigma, sigSoma, valuesW):

    i = int((sigSoma + 8) / 2)
    j = int((sigma + 1) / 2)
    k = int((sigSoma + 8) / 2)

    return valuesW[i, j, k]


def soma(spin_states, N, x, y, z):
    left = spin_states[x, (y - 1) % N, z]
    right = spin_states[x, (y + 1) % N, z]
    top = spin_states[(x - 1) % N, y, z]
    bottom = spin_states[(x + 1) % N, y, z]
    front = spin_states[x, y, (z + 1) % N]
    back = spin_states[x, y, (z - 1) % N]

    soma = left + right + top + bottom + front + back

    return soma


def inicializacao(tamanho, valor=-1):
    if valor != 0:
        rede = np.full((tamanho, tamanho, tamanho), valor, dtype="int")
    else:
        rede = np.random.choice([-1, 1], size=(tamanho, tamanho, tamanho))
    return rede


def cicloFerro(rede, tamanho, valuesW, h):
    e = 0
    for x in range(tamanho):
        for y in range(tamanho):
            for z in range(tamanho):
                sigma = rede[x, y, z]  # Direcção do spin do ponto de rede

                # A soma das direcções dos primeiros vizinhos
                soma = soma(rede, tamanho, x, y, z)

                soma *= sigma
                etmp = -0.5 * (soma - sigma * h)
                p = np.random.random()

                if p < w(sigma, soma, valuesW):
                    rede[x, y, z] = -sigma
                    etmp = -etmp

                e += etmp

    return rede, e


def ferroSimul(tamanho, nCiclos, temp, h):
    rede = inicializacao(tamanho)

    valuesW = transitionFunctionValues(temp, h)

    order = np.zeros(nCiclos)

    e = np.zeros(nCiclos)

    for i in range(nCiclos):
        rede, eCiclo = cicloFerro(rede, tamanho, valuesW, h)

        order[i] = 2 * rede[rede == 1].shape[0] - tamanho**2

        e[i] = eCiclo

    order /= tamanho**2
    e /= tamanho**2

    return rede, order, e


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


def simulacao_temp(temps, size, n_ciclos, h):
    m = np.array([0.0] * len(temps))
    sus = np.array([0.0] * len(temps))
    e = np.array([0.0] * len(temps))
    c = np.array([0.0] * len(temps))
    for i, t in enumerate(temps):
        print("Temperatura", t)
        rede, order, e_ = ferroSimul(size, n_ciclos, t, h)
        m[i] = momento_magnetico_medio(order, size)
        sus[i] = susceptibilidade_magnetica(order.var(), t, size)
        e[i] = energia_media_por_ponto_de_rede(e_.sum(), size)
        c[i] = capacidade_calorifica(e_.var(), t, size)
    return m, sus, e, c


# 1
def hysteresis_calc_varying_temp(temperatures, size, n_ciclos, h):
    magnetizations = np.array([])
    for temperature in temperatures:
        print("Temperatura", temperature)
        rede, order, e = ferroSimul(size, n_ciclos, temperature, h)
        magnetizations = np.append(magnetizations, np.sum(order) / size)
    magnetizations /= size**2
    return magnetizations


# 2
def hysteresis_calc_varying_h(temperature, size, n_ciclos, h_values):
    magnetizations = np.array([])
    for h in h_values:
        print("Campo magnético externo", h)
        rede, order, e = ferroSimul(size, n_ciclos, temperature, h)
        magnetizations = np.append(magnetizations, np.sum(order) / size)
    magnetizations /= size**2
    return magnetizations


def calculate_curie_temperature(temps, m, sus, e, c):
    max_sus = max(sus)
    temps[sus == max_sus]

    max_c = max(c)
    temps[c == max_c]

    output = np.array([temps, m, sus, e, c]).transpose()
    np.savetxt("results3D.tsv", output, delimiter="\t")

    print("Temperaturas salvas no arquivo results3D.tsv")
    # print(f"Temperatura de Curie estimada: {temps[sus == max_sus][0]}")
    print(f"Temperatura de Curie estimada: {temps[c == max_c][0]}")


# 1
def calc_magnetism_for_mult_temps_varying_temp(temperatures, mc_cycles, h_values, size):
    magnetizations = np.array([])
    for h in h_values:
        print("Campo magnético externo", h)
        magnetizations = np.append(
            magnetizations,
            hysteresis_calc_varying_temp(temperatures, size, mc_cycles, h),
        )
    return magnetizations


# 2
def calc_magnetism_for_mult_temps_varying_h(temperatures, mc_cycles, h_values, size):
    magnetizations = np.array([])
    for temperature in temperatures:
        print("Temperatura = ", temperature)
        magnetizations = np.append(
            magnetizations,
            hysteresis_calc_varying_h(temperature, size, mc_cycles, h_values),
        )
    return magnetizations


# Functions to plot the final state of the grid
def plot_grid(rede):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    x, y, z = np.where(rede == 1)
    ax.scatter(x, y, z, c="r", marker="o")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    plt.show()


# Functions to plot the order and energy graphs
def plot_graphs(order, e):
    plt.plot(order)
    plt.show()

    plt.plot(e)
    plt.show()


# Functions to plot the graphs of m, χ, e and C
def plot_ferro_graph(m, sus, e, c, temps):
    fig, axs = plt.subplots(2, 2)
    axs[0, 0].plot(temps, m)
    axs[0, 0].set_title("m vs t")
    # plt.ticklabel_format(style="plain")
    axs[0, 1].plot(temps, sus)
    axs[0, 1].set_title("χ vs t")
    # plt.ticklabel_format(style="plain")
    axs[1, 0].plot(temps, e)
    axs[1, 0].set_title("e vs t")
    # plt.ticklabel_format(style="plain")
    axs[1, 1].plot(temps, c)
    axs[1, 1].set_title("C vs t")
    # plt.ticklabel_format(style="plain")
    fig.tight_layout()
    plt.show()


def plot_hysteresis(temperature, h_values, magnetizations):
    plt.figure(figsize=(8, 6))
    plt.plot(h_values, magnetizations)
    plt.title(f"Hysteresis Loop at T = {temperature}")
    plt.xlabel("External Magnetic Field (h)")
    plt.ylabel("Magnetization (M)")
    plt.grid(True)
    plt.ticklabel_format(style="plain")
    plt.show()


def plot_magnetism_for_mult_temps(temperatures, h_values, magnetizations):
    fig, ax = plt.subplots(figsize=(12, 12))
    for i, temperature in enumerate(temperatures):
        ax.plot(
            h_values,
            magnetizations[i * len(h_values) : (i + 1) * len(h_values)],
            "-o",
            label=f"T = {temperature}",
        )
    ax.set_title("Magnetization vs. External Magnetic Field for Multiple Temperatures")
    ax.set_xlabel("External Magnetic Field (h)")
    ax.set_ylabel("Magnetization (M)")
    ax.legend()
    ax.grid(True)
    plt.show()


temperature = 5.5
h = 0.0
size = 10
mc_cicles = 10000

start = time.time()
rede, order, e = ferroSimul(size, mc_cicles, temperature, h)
end = time.time()
elapsed_time = end - start
print("Tempo de execução a fazer a simulação:", elapsed_time, "segundos")
plot_graphs(order, e)

temperatures = np.arange(0.5, 5.5, 0.1)

start = time.time()
m, sus, e, c = simulacao_temp(temperatures, size, (int)(mc_cicles * 0.1), h)
end = time.time()
elapsed_time = end - start
print("Tempo de execução a fazer a calcular as propriedades:", elapsed_time, "segundos")
plot_ferro_graph(m, sus, e, c, temperatures)

calculate_curie_temperature(temperatures, m, sus, e, c)

# h values
h_max = 4  # Maximum strength of magnetic field
h_values = np.linspace(-h_max, h_max, (h_max * 2) + 1)

magnetizationsTemperatures = np.array([0.5, 2.4, 2.5, 2.6, 4.5])

start = time.time()
magnetizations = hysteresis_calc_varying_h(
    temperature, size, (int)(mc_cicles * 0.1), h_values
)
end = time.time()
elapsed_time = end - start
print("Tempo de execução a fazer a calcular a histerese:", elapsed_time, "segundos")
plot_hysteresis(temperature, h_values, magnetizations)

start = time.time()
magnetism_for_mult_temps_varying_temp = calc_magnetism_for_mult_temps_varying_temp(
    magnetizationsTemperatures, (int)(mc_cicles * 0.1), h_values, size
)
end = time.time()
elapsed_time = end - start
print(
    "Tempo de execução a fazer a calcular o magnetismo para múltiplas temperaturas variando a temperatura:",
    elapsed_time,
    "segundos",
)
plot_magnetism_for_mult_temps(
    magnetizationsTemperatures, h_values, magnetism_for_mult_temps_varying_temp
)

start = time.time()
magnetism_for_mult_temps_varying_h = calc_magnetism_for_mult_temps_varying_h(
    magnetizationsTemperatures, (int)(mc_cicles * 0.1), h_values, size
)
end = time.time()
elapsed_time = end - start
print(
    "Tempo de execução a fazer a calcular o magnetismo para múltiplas temperaturas variando o campo magnético externo:",
    elapsed_time,
    "segundos",
)
plot_magnetism_for_mult_temps(
    magnetizationsTemperatures, h_values, magnetism_for_mult_temps_varying_h
)
