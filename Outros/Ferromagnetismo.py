import numpy as np
import matplotlib.pylab as plt

# BOMM, "DADO PELO STOR???"


def transitionFunctionValues(t, h=0):

    deltaE = [[j + i * h for i in range(-1, 3, 2)] for j in range(-4, 6, 2)]
    output = [
        [1 if elem <= 0 else np.exp(-2 * elem / t) for elem in row] for row in deltaE
    ]

    return np.array(output)


def w(sigma, sigSoma, valuesW):

    i = int(sigSoma / 2 + 2)
    j = int(sigma / 2 + 1 / 2)

    return valuesW[i, j]


def vizinhosTabela(tamanho):

    vizMais = [i + 1 for i in range(tamanho)]
    vizMais[-1] = 0
    vizMenos = [i - 1 for i in range(tamanho)]

    return np.array([vizMais, vizMenos])


def inicializacao(tamanho, valor=-1):

    if valor != 0:
        rede = np.full((tamanho, tamanho), valor, dtype="int")
    else:
        rede = np.random.choice([-1, 1], size=(tamanho, tamanho))
    return rede


def cicloFerro(rede, tamanho, vizinhos, valuesW, h):

    e = 0
    for i in range(tamanho):
        for j in range(tamanho):
            sigma = rede[i, j]  # Direcção do spin do ponto de rede
            # A soma das direcções dos primeiros vizinhos
            soma = (
                rede[vizinhos[0, i], j]
                + rede[vizinhos[1, i], j]
                + rede[i, vizinhos[0, j]]
                + rede[i, vizinhos[1, j]]
            )
            soma *= sigma
            etmp = -0.5 * (soma - sigma * h)
            p = (
                np.random.random()
            )  # Número aleatório entre 0 e 1, distribuição uniforme

            if p < w(sigma, soma, valuesW):
                rede[i, j] = -sigma
                etmp = -etmp

            e += etmp

    return rede, e


def ferroSimul(tamanho, nCiclos, temp, h):

    rede = inicializacao(tamanho)

    valuesW = transitionFunctionValues(temp, h)

    vizinhos = vizinhosTabela(tamanho)

    order = np.zeros(nCiclos)

    e = np.zeros(nCiclos)

    for i in range(nCiclos):
        print("Ciclo", i)
        rede, eCiclo = cicloFerro(rede, tamanho, vizinhos, valuesW, h)

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


def hysteresis_loop(temperature, size, n_ciclos, h_values):
    magnetizations = np.array([])
    for h in h_values:
        print("Campo magnético externo", h)
        rede, order, e = ferroSimul(size, n_ciclos, temperature, h)
        # m = momento_magnetico_medio(order, size)
        magnetizations = np.append(magnetizations, np.sum(order) / size)
    return magnetizations


def plot_hysteresis(temperature, h_values, magnetizations):
    plt.figure(figsize=(8, 6))
    plt.plot(h_values, magnetizations)
    plt.title(f"Hysteresis Loop at T = {temperature}")
    plt.xlabel("External Magnetic Field (h)")
    plt.ylabel("Magnetization (M)")
    plt.grid(True)
    plt.ticklabel_format(style="plain")
    plt.show()


t = 5.5
h = 0.0
tamanho = 10
mc_cicles = 10000
rede, order, e = ferroSimul(tamanho, mc_cicles, t, h)
plt.plot(order)
plt.show()

temps = np.arange(0.5, 5.5, 0.1)

m, sus, e, c = simulacao_temp(temps, tamanho, (int)(mc_cicles * 0.1), h)
plot_ferro_graph(m, sus, e, c, temps)

# # h values
# h_max = 4  # Maximum strength of magnetic field
# h_values = np.linspace(-h_max, h_max, (h_max * 2) + 1)


# magnetizations = hysteresis_loop(t, tamanho, (int)(mc_cicles * 0.1), h_values)
# plot_hysteresis(t, h_values, magnetizations)
