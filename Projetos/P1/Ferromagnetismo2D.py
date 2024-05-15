import time
import numpy as np
import matplotlib.pylab as plt
import threading


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
        print("Temperatura", t, end="\r")
        rede, order, e_ = ferroSimul(size, n_ciclos, t, h)
        m[i] = momento_magnetico_medio(order, size)
        sus[i] = susceptibilidade_magnetica(order.var(), t, size)
        e[i] = energia_media_por_ponto_de_rede(e_.sum(), size)
        c[i] = capacidade_calorifica(e_.var(), t, size)
    return m, sus, e, c


# # 1
# def hysteresis_calc_varying_temp(temperatures, size, n_ciclos, h):
#     magnetizations = np.array([])
#     for temperature in temperatures:
#         print("Temperatura:", temperature, "Campo magnético externo:", h, end="\r")
#         rede, order, e = ferroSimul(size, n_ciclos, temperature, h)
#         magnetizations = np.append(magnetizations, np.sum(order) / size)
#     magnetizations /= size**2
#     return magnetizations


# 2
def hysteresis_calc_varying_h(temperature, size, n_ciclos, h_values):
    magnetizations = np.array([])
    for h in h_values:
        print("Temperatura:", temperature, "Campo magnético externo:", h, end="\r")
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
    np.savetxt("results2D.tsv", output, delimiter="\t")

    print("Temperaturas salvas no arquivo results2D.tsv")
    # print(f"Temperatura de Curie estimada: {temps[sus == max_sus][0]}")
    print(f"Temperatura de Curie estimada: {temps[c == max_c][0]}")


# # 1
# def calc_magnetism_for_mult_temps_varying_temp(temperatures, mc_cycles, h_values, size):
#     magnetizations = np.array([])
#     for h in h_values:
#         print("Campo magnético externo", h, end="\r")
#         magnetizations = np.append(
#             magnetizations,
#             hysteresis_calc_varying_temp(temperatures, size, mc_cycles, h),
#         )
#     return magnetizations


# 2
def calc_magnetism_for_mult_temps_varying_h(temperatures, mc_cycles, h_values, size):
    magnetizations = np.array([])
    for temperature in temperatures:
        magnetizations = np.append(
            magnetizations,
            hysteresis_calc_varying_h(temperature, size, mc_cycles, h_values),
        )
    return magnetizations


# Functions to plot the order and energy graphs
def plot_graphs(order, e):
    plt.plot(order)
    plt.title("Order vs MC Cycles")
    plt.show()

    plt.plot(e)
    plt.title("Energy vs MC Cycles")
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


# # 1
# def plot_magnetism_for_mult_temps_1(temperatures, h_values, magnetizations):
#     fig, ax = plt.subplots(figsize=(12, 12))
#     for i, h in enumerate(h_values):
#         ax.plot(
#             temperatures,
#             magnetizations[i * len(temperatures) : (i + 1) * len(temperatures)],
#             "-o",
#             label=f"h = {h}",
#         )
#     ax.set_title("Magnetization vs. Temperature for Multiple External Magnetic Fields")
#     ax.set_xlabel("Temperature (T)")
#     ax.set_ylabel("Magnetization (M)")
#     ax.legend()
#     ax.grid(True)
#     plt.show()


# 2
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


def timer(signal):
    seconds = 0
    while not signal.is_set():
        print(
            "                                               ",
            "Temporizador : {} segundos".format(seconds),
            end="\r",
        )
        seconds += 1
        time.sleep(1)


t = 5.5
h = 0.0
size = 10
mc_cicles = 10000
times = np.array([])

signal = threading.Event()
thread_timer = threading.Thread(target=timer, args=(signal,))
thread_timer.start()
start = time.time()
rede, order, e = ferroSimul(size, mc_cicles, t, h)
end = time.time()
signal.set()
thread_timer.join()
elapsed_time = end - start
times = np.append(times, elapsed_time)
print("Tempo de execução a fazer a simulação inicial: ", elapsed_time, "segundos")
plot_graphs(order, e)

temperatures = np.arange(0.5, 5.5, 0.1)

signal = threading.Event()
thread_timer = threading.Thread(target=timer, args=(signal,))
thread_timer.start()
start = time.time()
m, sus, e, c = simulacao_temp(temperatures, size, (int)(mc_cicles * 0.1), h)
end = time.time()
signal.set()
thread_timer.join()
elapsed_time = end - start
times = np.append(times, elapsed_time)
print("Tempo de execução a fazer a calcular as propriedades:", elapsed_time, "segundos")
plot_ferro_graph(m, sus, e, c, temperatures)

calculate_curie_temperature(temperatures, m, sus, e, c)

# h values
h_max = 4  # Maximum strength of magnetic field
h_values = np.linspace(-h_max, h_max, (h_max * 2) + 1)

magnetizationsTemperatures = np.array([0.5, 2.4, 2.5, 2.6, 4.5])

start = time.time()
magnetizations = hysteresis_calc_varying_h(t, size, (int)(mc_cicles * 0.1), h_values)
end = time.time()
elapsed_time = end - start
times = np.append(times, elapsed_time)
print("Tempo de execução a fazer a calcular a histerese:", elapsed_time, "segundos")
plot_hysteresis(t, h_values, magnetizations)

# start = time.time()
# magnetism_for_mult_temps_varying_temp = calc_magnetism_for_mult_temps_varying_temp(
#     magnetizationsTemperatures, (int)(mc_cicles * 0.1), h_values, size
# )
# end = time.time()
# elapsed_time = end - start
# times = np.append(times, elapsed_time)
# print(
#     "Tempo de execução a fazer a calcular o magnetismo para múltiplas temperaturas variando a temperatura:",
#     elapsed_time,
#     "segundos",
# )
# plot_magnetism_for_mult_temps_1(
#     magnetizationsTemperatures, h_values, magnetism_for_mult_temps_varying_temp
# )

signal = threading.Event()
thread_timer = threading.Thread(target=timer, args=(signal,))
thread_timer.start()
start = time.time()
magnetism_for_mult_temps_varying_h = calc_magnetism_for_mult_temps_varying_h(
    magnetizationsTemperatures, (int)(mc_cicles * 0.1), h_values, size
)
end = time.time()
signal.set()
thread_timer.join()
elapsed_time = end - start
times = np.append(times, elapsed_time)
print(
    "Tempo de execução a fazer a calcular o magnetismo para múltiplas temperaturas variando o campo magnético externo:",
    elapsed_time,
    "segundos",
)
plot_magnetism_for_mult_temps(
    magnetizationsTemperatures, h_values, magnetism_for_mult_temps_varying_h
)

print("Tempo total de execução:", np.sum(times), "segundos")
