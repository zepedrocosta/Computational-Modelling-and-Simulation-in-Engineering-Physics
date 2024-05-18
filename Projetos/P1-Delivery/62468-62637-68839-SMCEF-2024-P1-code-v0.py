import numpy as np
import matplotlib.pyplot as plt


def transitionFunctionValues(t, h):
    deltaE = [[j + i * h for i in range(-1, 3, 2)] for j in range(-6, 8, 2)]
    output = [
        [1 if elem <= 0 else np.exp(-2 * elem / t) for elem in row] for row in deltaE
    ]
    return np.array(output)


def w(sigma, sigSoma, valuesW):
    i = int(sigSoma / 2 + 3)
    j = int(sigma / 2 + 1 / 2)
    return valuesW[i, j]


def vizinhosTabela(tamanho):
    vizMais = [i + 1 for i in range(tamanho)]
    vizMais[-1] = 0
    vizMenos = [i - 1 for i in range(tamanho)]

    vizinhos = np.array([vizMais, vizMenos])
    return vizinhos


def inicializacao(tamanho, spin):
    if spin == -1 or spin == 1:
        rede = np.full((tamanho, tamanho, tamanho), spin, dtype="int")
    else:
        rede = np.random.choice([-1, 1], size=(tamanho, tamanho, tamanho))
    return rede


def cicloFerro(rede, vizinhos, tamanho, valuesW, h):
    e = 0
    for x in range(tamanho):
        for y in range(tamanho):
            for z in range(tamanho):
                sigma = rede[x, y, z]
                soma = (
                    rede[vizinhos[0, x], y, z]
                    + rede[vizinhos[1, x], y, z]
                    + rede[x, vizinhos[0, y], z]
                    + rede[x, vizinhos[1, y], z]
                    + rede[x, y, vizinhos[0, z]]
                    + rede[x, y, vizinhos[1, z]]
                )
                soma *= sigma
                etmp = -0.5 * (soma - sigma * h)
                p = np.random.random()

                if p < w(sigma, soma, valuesW):
                    rede[x, y, z] = -sigma
                    etmp = -etmp

                e += etmp

    return rede, e


def ferroSimul(tamanho, nCiclos, temp, h, spin):
    rede = inicializacao(tamanho, spin)
    vizinhos = vizinhosTabela(tamanho)
    valuesW = transitionFunctionValues(temp, h)
    order = np.zeros(nCiclos)
    e = np.zeros(nCiclos)

    for i in range(nCiclos):
        rede, eCiclo = cicloFerro(rede, vizinhos, tamanho, valuesW, h)
        order[i] = 2 * rede[rede == 1].shape[0] - tamanho**3
        e[i] = eCiclo

    order /= tamanho**3
    e /= tamanho**3
    return rede, order, e


def momento_magnetico_medio(M, L):
    m = np.linalg.norm(M) / (L**2)
    return m


def energia_media_por_ponto_de_rede(E, L):
    epsilon = E / (L**2)
    return epsilon


def susceptibilidade_magnetica(sigma_M, t, L):
    chi = ((sigma_M**2) / t) * (L**2)
    return chi


def capacidade_calorifica(sigma_epsilon, t, L):
    C = (sigma_epsilon**2) / (t**2 * L**2)
    return C


def simulacao_temp(temps, size, n_ciclos, h, spin):
    m = np.array([0.0] * len(temps))
    sus = np.array([0.0] * len(temps))
    e = np.array([0.0] * len(temps))
    c = np.array([0.0] * len(temps))
    for i, t in enumerate(temps):
        rede, order, e_ = ferroSimul(size, n_ciclos, t, h, spin)
        m[i] = momento_magnetico_medio(order, size)
        sus[i] = susceptibilidade_magnetica(order.var(), t, size)
        e[i] = energia_media_por_ponto_de_rede(e_.sum(), size)
        c[i] = capacidade_calorifica(e_.var(), t, size)
    return m, sus, e, c


def hysteresis_calc_varying_h(temperature, size, n_ciclos, h_values, spin):
    magnetizations = np.array([])
    for h in h_values:
        rede, order, e = ferroSimul(size, n_ciclos, temperature, h, spin)
        magnetizations = np.append(magnetizations, np.sum(order) / size)
    magnetizations /= size**2
    return magnetizations


def calculate_curie_temperature(temps, m, sus, e, c):
    max_sus = max(sus)
    temps[sus == max_sus]

    max_c = max(c)
    temps[c == max_c]

    output = np.array([temps, m, sus, e, c]).transpose()
    print("Curie Temperature:")
    print(output)


def calc_magnetism_for_mult_temps(temperatures, mc_cycles, h_values, size, spin):
    magnetizations = np.array([])
    for temperature in temperatures:
        magnetizations = np.append(
            magnetizations,
            hysteresis_calc_varying_h(temperature, size, mc_cycles, h_values, spin),
        )
    return magnetizations


def plot_grid(rede):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    x, y, z = np.where(rede == 1)
    ax.scatter(x, y, z, c="r", marker="o")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    plt.show()


def plot_graphs(order, e):
    plt.plot(order)
    plt.show()

    plt.plot(e)
    plt.show()


def plot_ferro_graph(m, sus, e, c, temps):
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


spin = -1
temperature = 5.5
h = 0.0
size = 10
mc_cicles = 10000

rede, order, e = ferroSimul(size, mc_cicles, temperature, h, spin)
plot_grid(rede)
plot_graphs(order, e)

temperatures = np.arange(0.5, 5.5, 0.1)

m, sus, e, c = simulacao_temp(temperatures, size, (int)(mc_cicles * 0.1), h, spin)
plot_ferro_graph(m, sus, e, c, temperatures)

calculate_curie_temperature(temperatures, m, sus, e, c)

h_max = 5
h_values = np.linspace(-h_max, h_max, (h_max * 2) + 1)

magnetizationsTemperatures = np.array([0.5, 2.4, 2.5, 2.6, 4.5])

magnetizations = hysteresis_calc_varying_h(
    temperature, size, (int)(mc_cicles * 0.1), h_values, spin
)
plot_hysteresis(temperature, h_values, magnetizations)


magnetism_for_mult_temps = calc_magnetism_for_mult_temps(
    magnetizationsTemperatures, (int)(mc_cicles * 0.1), h_values, size, spin
)
plot_magnetism_for_mult_temps(
    magnetizationsTemperatures, h_values, magnetism_for_mult_temps
)
