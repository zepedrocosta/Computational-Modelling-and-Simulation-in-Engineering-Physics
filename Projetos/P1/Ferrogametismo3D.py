import time
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def inicializacao(tamanho, valor=-1):
    # Adjusted for 3D
    if valor != 0:
        rede = np.full((tamanho, tamanho, tamanho), valor, dtype="int")
    else:
        rede = np.random.choice([-1, 1], size=(tamanho, tamanho, tamanho))
    return rede


def transitionFunctionValues(t, h=0):
    deltaE = [
        [[k + j * h + i * h * h for k in range(-1, 3, 2)] for j in range(-4, 6, 2)]
        for i in range(-4, 6, 2)
    ]
    output = [
        [[1 if elem <= 0 else np.exp(-2 * elem / t) for elem in row] for row in plane]
        for plane in deltaE
    ]
    return np.array(output)


def w(sigma, sigSoma, valuesW):
    i = int(sigSoma / 2 + 2)
    j = int(sigma / 2 + 1 / 2)
    k = int((sigSoma - sigma) / 2 + 2)
    i = min(max(i, 0), valuesW.shape[0] - 1)
    j = min(max(j, 0), valuesW.shape[1] - 1)
    k = min(max(k, 0), valuesW.shape[2] - 1)
    return valuesW[i, j, k]


def vizinhosTabela(tamanho):
    vizMais = [i + 1 for i in range(tamanho)]
    vizMais[-1] = 0
    vizMenos = [i - 1 for i in range(tamanho)]
    return np.array([vizMais, vizMenos])


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


def cicloFerro(rede, tamanho, vizinhos, valuesW, h):
    e = 0
    for i in range(tamanho):
        for j in range(tamanho):
            for k in range(tamanho):
                sigma = rede[i, j, k]
                soma = (
                    rede[vizinhos[0, i], j, k]
                    + rede[vizinhos[1, i], j, k]
                    + rede[i, vizinhos[0, j], k]
                    + rede[i, vizinhos[1, j], k]
                    + rede[i, j, vizinhos[0, k]]
                    + rede[i, j, vizinhos[1, k]]
                )
                soma *= sigma
                etmp = -0.5 * (soma - sigma * h)
                p = np.random.random()
                if p < w(sigma, soma, valuesW):
                    rede[i, j, k] = -sigma
                    etmp = -etmp
                e += etmp
    return rede, e


def ferromagnetic_simulation(tamanho, mc_cicles, temp, h):
    rede = inicializacao(tamanho)
    valuesW = transitionFunctionValues(temp, h)
    vizinhos = vizinhosTabela(tamanho)
    order = np.zeros(mc_cicles)
    e = np.zeros(mc_cicles)
    for i in range(mc_cicles):
        rede, eCiclo = cicloFerro(rede, tamanho, vizinhos, valuesW, h)
        order[i] = 2 * rede[rede == 1].shape[0] - tamanho**3
        e[i] = eCiclo
    order /= tamanho**3
    e /= tamanho**3
    return rede, order, e


def simulacao_temp(temps, size, n_ciclos, h):
    m = np.array([0.0] * len(temps))
    sus = np.array([0.0] * len(temps))
    e = np.array([0.0] * len(temps))
    c = np.array([0.0] * len(temps))
    for i, t in enumerate(temps):
        print("Temperatura", t)
        rede, order, e_ = ferromagnetic_simulation(size, n_ciclos, t, h)
        m[i] = momento_magnetico_medio(order, size)
        sus[i] = susceptibilidade_magnetica(order.var(), t, size)
        e[i] = energia_media_por_ponto_de_rede(e_.sum(), size)
        c[i] = capacidade_calorifica(e_.var(), t, size)
    return m, sus, e, c


def calculate_curie_temperature(temps, m, sus, e, c):
    max_sus = max(sus)
    temps[sus == max_sus]

    max_c = max(c)
    temps[c == max_c]

    output = np.array([temps, m, sus, e, c]).transpose()
    np.savetxt("results.tsv", output, delimiter="\t")

    # print(data)
    print("Temperaturas salvas no arquivo results.tsv")
    print(f"Temperatura de Curie estimada: {temps[sus == max_sus][0]}")
    print(f"Temperatura de Curie estimada: {temps[c == max_c][0]}")

def hysteresis_loop(temperature, size, n_ciclos, h_values):
    magnetizations = np.array([])
    for h in h_values:
        print("Campo magnético externo", h)
        rede, order, e = ferromagnetic_simulation(size, n_ciclos, temperature, h)
        m = momento_magnetico_medio(order, size)
        magnetizations = np.append(magnetizations, np.sum(order) / size)
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


# Function to plot the hysteresis loop
def plot_hysteresis(temperature, h_values, magnetizations):
    plt.figure(figsize=(8, 6))
    plt.plot(h_values, magnetizations)
    plt.title(f"Hysteresis Loop at T = {temperature}")
    plt.xlabel("External Magnetic Field (h)")
    plt.ylabel("Magnetization (M)")
    plt.grid(True)
    plt.ticklabel_format(style="plain")
    plt.show()


temperature = 5.5
h = 0.0
size = 10
mc_cicles = 10000

# h values
h_max = 4  # Maximum strength of magnetic field
h_values = np.linspace(-h_max, h_max, (h_max * 2) + 1)

start = time.time()
rede, order, e = ferromagnetic_simulation(size, mc_cicles, temperature, h)
end = time.time()
elapsedTime = end - start
print(
    "Elapsed time executing the ferromagnetic simulation:", elapsedTime / 60, " minutes"
)

plot_grid(rede)

plot_graphs(order, e)

sus = order.var() * size**2 / temperature
print(sus)

C = e.var() / (size**2 * temperature**2)
print(C)

temps = np.arange(0.5, 5.5, 0.1)

start = time.time()
m, sus, e, c = simulacao_temp(temps, size, (int)(mc_cicles * 0.1), h)
end = time.time()
elapsedTime = end - start
print(
    "Elapsed time calculating physical quantities (m, χ, e, C): ",
    elapsedTime / 60,
    " minutes",
)

plot_ferro_graph(m, sus, e, c, temps)

calculate_curie_temperature(temps, m, sus, e, c)

temperatures = np.array([0.5, 2.4, 2.5, 2.6, 4.5])

start = time.time()
magnetizations = hysteresis_loop(temperature, size, (int)(mc_cicles * 0.1), h_values)
end = time.time()
elapsedTime = end - start
print("Elapsed time calculating hysteresis loop: ", elapsedTime / 60, " minutes")
plot_hysteresis(temperature, h_values, magnetizations)
