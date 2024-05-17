import time
import numpy as np
import matplotlib.pyplot as plt
import os
import shutil


def transitionFunctionValues(t, h):
    """
    Calcula os valores da função de transição.

    Args:
    - t: Temperatura
    - h: Campo magnético externo

    Returns:
    - output: Valores da função de transição
    """
    deltaE = [[j + i * h for i in range(-1, 3, 2)] for j in range(-6, 8, 2)]
    output = [
        [1 if elem <= 0 else np.exp(-2 * elem / t) for elem in row] for row in deltaE
    ]
    return np.array(output)


def w(sigma, sigSoma, valuesW):
    """
    Calcula a função de transição.

    Args:
    - sigma: Spin do ponto
    - sigSoma: Soma dos spins dos vizinhos
    - valuesW: Valores da função de transição

    Returns:
    - valuesW: Valores da função de transição
    """
    i = int(sigSoma / 2 + 3)
    j = int(sigma / 2 + 1 / 2)
    return valuesW[i, j]


def vizinhosTabela(tamanho):
    """
    Cria uma tabela com os vizinhos.

    Args:
    - tamanho: Tamanho da rede

    Returns:
    - vizinhos: Tabela com os vizinhos
    """
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
    """
    Faz um ciclo de Monte Carlo para o ferromagnetismo em 3D.

    Args:
    - rede: Rede com os spins
    - tamanho: Tamanho da rede
    - valuesW: Valores da função de transição
    - h: Campo magnético externo

    Returns:
    - rede: Rede com os spins
    - e: Energia no ponto
    """
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
    """
    Simula o ferromagnetismo em 3D.

    Args:
    - tamanho: Tamanho da rede
    - nCiclos: Número de ciclos
    - temp: Temperatura
    - h: Campo magnético externo

    Returns:
    - rede: Rede com os spins
    - order: Fator de ordem
    - e: Energia
    """
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


def simulacao_temp(temps, size, n_ciclos, h, spin):
    """
    Simula o ferromagnetismo em 3D calculando as propriedades ferromagneticas para várias temperaturas.

    Args:
    - temps: Lista de temperaturas
    - size: Tamanho da rede
    - n_ciclos: Número de ciclos de Monte Carlo
    - h: Campo magnético externo

    Returns:
    - m: Momento magnético médio
    - sus: Susceptibilidade magnética
    - e: Energia média por ponto de rede
    - c: Capacidade calorífica
    """
    m = np.array([0.0] * len(temps))
    sus = np.array([0.0] * len(temps))
    e = np.array([0.0] * len(temps))
    c = np.array([0.0] * len(temps))
    for i, t in enumerate(temps):
        print("Temperatura", t)
        rede, order, e_ = ferroSimul(size, n_ciclos, t, h, spin)
        m[i] = momento_magnetico_medio(order, size)
        sus[i] = susceptibilidade_magnetica(order.var(), t, size)
        e[i] = energia_media_por_ponto_de_rede(e_.sum(), size)
        c[i] = capacidade_calorifica(e_.var(), t, size)
    return m, sus, e, c


def hysteresis_calc_varying_h(temperature, size, n_ciclos, h_values, spin):
    """
    Calcula a histerese variando o campo magnético externo.

    Args:
    - temperature: Temperatura
    - size: Tamanho da rede
    - n_ciclos: Número de ciclos de Monte Carlo
    - h_values: Valores do campo magnético externo

    Returns:
    - magnetizations: Magnetizações
    """
    magnetizations = np.array([])
    for h in h_values:
        print("Campo magnético externo", h)
        rede, order, e = ferroSimul(size, n_ciclos, temperature, h, spin)
        magnetizations = np.append(magnetizations, np.sum(order) / size)
    magnetizations /= size**2
    return magnetizations


def calculate_curie_temperature(temps, m, sus, e, c):
    """
    Calcula a temperatura de Curie e salva as temperaturas e as propriedades em um ficheiro.

    Args:
    - temps: Lista de temperaturas
    - m: Momento magnético médio
    - sus: Susceptibilidade magnética
    - e: Energia média por ponto de rede
    - c: Capacidade calorífica
    """
    file_name = "results3D.tsv"

    max_sus = max(sus)
    temps[sus == max_sus]

    max_c = max(c)
    temps[c == max_c]

    output = np.array([temps, m, sus, e, c]).transpose()
    np.savetxt(file_name, output, delimiter="\t")

    moveFile(file_name)

    print("Temperaturas salvas no arquivo", file_name)
    print(f"Temperatura de Curie estimada: {temps[c == max_c][0]}")


def calc_magnetism_for_mult_temps(temperatures, mc_cycles, h_values, size, spin):
    """
    Calcula o magnetismo para múltiplas temperaturas variando o campo magnético externo.

    Args:
    - temperatures: Lista de temperaturas
    - mc_cycles: Número de ciclos de Monte Carlo
    - h_values: Valores do campo magnético externo
    - size: Tamanho da rede

    Returns:
    - magnetizations: Magnetizações
    """
    magnetizations = np.array([])
    for temperature in temperatures:
        print("Temperatura = ", temperature)
        magnetizations = np.append(
            magnetizations,
            hysteresis_calc_varying_h(temperature, size, mc_cycles, h_values, spin),
        )
    return magnetizations


def plot_grid(rede):
    """
    Plota a rede em 3D.

    Args:
    - rede: Rede com os spins
    """
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
    """
    Plota os gráficos da ordem e da energia.

    Args:
    - order: ordem
    - e: Energia
    """
    plt.plot(order)
    plt.show()

    plt.plot(e)
    plt.show()


def plot_ferro_graph(m, sus, e, c, temps):
    """
    Plota os gráficos das propriedades do ferromagnetismo.

    Args:
    - m: Momento magnético médio
    - sus: Susceptibilidade magnética
    - e: Energia média por ponto de rede
    - c: Capacidade calorífica
    - temps: Lista de temperaturas
    """
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
    """
    Plota a histerese.

    Args:
    - temperature: Temperatura
    - h_values: Valores do campo magnético externo
    - magnetizations: Magnetizações
    """
    plt.figure(figsize=(8, 6))
    plt.plot(h_values, magnetizations)
    plt.title(f"Hysteresis Loop at T = {temperature}")
    plt.xlabel("External Magnetic Field (h)")
    plt.ylabel("Magnetization (M)")
    plt.grid(True)
    plt.ticklabel_format(style="plain")
    plt.show()


def plot_magnetism_for_mult_temps(temperatures, h_values, magnetizations):
    """
    Plota o magnetismo para múltiplas temperaturas variando o campo magnético externo.

    Args:
    - temperatures: Lista de temperaturas
    - h_values: Valores do campo magnético externo
    - magnetizations: Magnetizações
    """
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


def moveFile(file_name):
    file_location = os.getcwd()

    script_location = os.path.abspath(__file__)
    current_dir = os.path.dirname(script_location)
    destination_dir = os.path.join(current_dir, "Plots and Results")

    if not os.path.exists(destination_dir):
        os.makedirs(destination_dir)

    current_file = os.path.join(file_location, file_name)
    destination_file = os.path.join(destination_dir, file_name)

    shutil.move(current_file, destination_file)


spin = -1
temperature = 1
h = 0.0
size = 10
mc_cicles = 10000

# start = time.time()
# rede, order, e = ferroSimul(size, mc_cicles, temperature, h, spin)
# end = time.time()
# elapsed_time = end - start
# print("Tempo de execução a fazer a simulação:", elapsed_time, "segundos")
# plot_grid(rede)
# plot_graphs(order, e)

temperatures = np.arange(0.5, 1.0, 0.1)

start = time.time()
m, sus, e, c = simulacao_temp(temperatures, size, (int)(mc_cicles * 0.1), h, spin)
end = time.time()
elapsed_time = end - start
print("Tempo de execução a fazer a calcular as propriedades:", elapsed_time, "segundos")
plot_ferro_graph(m, sus, e, c, temperatures)

calculate_curie_temperature(temperatures, m, sus, e, c)

# h values
h_max = 5  # Maximum strength of magnetic field
h_values = np.linspace(-h_max, h_max, (h_max * 2) + 1)

magnetizationsTemperatures = np.array([0.5, 2.4, 2.5, 2.6, 4.5])

start = time.time()
magnetizations = hysteresis_calc_varying_h(
    temperature, size, (int)(mc_cicles * 0.1), h_values, spin
)
end = time.time()
elapsed_time = end - start
print("Tempo de execução a fazer a calcular a histerese:", elapsed_time, "segundos")
plot_hysteresis(temperature, h_values, magnetizations)


start = time.time()
magnetism_for_mult_temps = calc_magnetism_for_mult_temps(
    magnetizationsTemperatures, (int)(mc_cicles * 0.1), h_values, size, spin
)
end = time.time()
elapsed_time = end - start
print(
    "Tempo de execução a fazer a calcular o magnetismo para múltiplas temperaturas variando o campo magnético externo:",
    elapsed_time,
    "segundos",
)
plot_magnetism_for_mult_temps(
    magnetizationsTemperatures, h_values, magnetism_for_mult_temps
)
