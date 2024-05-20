import threading
import time
import numpy as np
import matplotlib.pyplot as plt
import os
import shutil
from colorama import Fore


def transition_function_values(t, h):
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


def neighbours_table(size):
    """
    Cria uma tabela com os vizinhos.

    Args:
    - tamanho: Tamanho da rede

    Returns:
    - vizinhos: Tabela com os vizinhos
    """
    vizPlus = [i + 1 for i in range(size)]
    vizPlus[-1] = 0
    vizMinus = [i - 1 for i in range(size)]

    neighbours = np.array([vizPlus, vizMinus])
    return neighbours


def initialize_grid(size, spin):
    """
    Inicializa a rede.

    Args:
    - tamanho: Tamanho da rede
    - spin: Spin inicial

    Returns:
    - rede: Rede com os spins
    """
    if spin == -1 or spin == 1:
        rede = np.full((size, size, size), spin, dtype="int")
    else:
        rede = np.random.choice([-1, 1], size=(size, size, size))
    return rede


def ferromagnetic_cycle(grid, neighbours, size, valuesW, h):
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
    for x in range(size):
        for y in range(size):
            for z in range(size):
                sigma = grid[x, y, z]
                soma = (
                    grid[neighbours[0, x], y, z]
                    + grid[neighbours[1, x], y, z]
                    + grid[x, neighbours[0, y], z]
                    + grid[x, neighbours[1, y], z]
                    + grid[x, y, neighbours[0, z]]
                    + grid[x, y, neighbours[1, z]]
                )
                soma *= sigma
                etmp = -0.5 * (soma - sigma * h)
                p = np.random.random()

                if p < w(sigma, soma, valuesW):
                    grid[x, y, z] = -sigma
                    etmp = -etmp

                e += etmp

    return grid, e


def ferromagnetic_simulation(size, cycles, temperature, h, spin):
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
    rede = initialize_grid(size, spin)
    vizinhos = neighbours_table(size)
    valuesW = transition_function_values(temperature, h)
    order = np.zeros(cycles)
    e = np.zeros(cycles)

    for i in range(cycles):
        rede, eCiclo = ferromagnetic_cycle(rede, vizinhos, size, valuesW, h)
        order[i] = 2 * rede[rede == 1].shape[0] - size**3
        e[i] = eCiclo

    order /= size**3
    e /= size**3
    return rede, order, e


def calculate_average_magnetic_moment(M, L):
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


def calculate_average_energy_per_network_point(E, L):
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


def calculate_magnetic_susceptibility(sigma_M, t, L):
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


def calculate_heat_capacity(sigma_epsilon, t, L):
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


def temperatureSimulation(temps, size, n_ciclos, h, spin):
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
        print(
            "                                               ",
            "Temperatura",
            t,
            end="\r",
        )
        rede, order, e_ = ferromagnetic_simulation(size, n_ciclos, t, h, spin)
        m[i] = calculate_average_magnetic_moment(order, size)
        sus[i] = calculate_magnetic_susceptibility(order.var(), t, size)
        e[i] = calculate_average_energy_per_network_point(e_.sum(), size)
        c[i] = calculate_heat_capacity(e_.var(), t, size)
    return m, sus, e, c


def hysteresis_calc_varying_h(temperature, size, cycles, h_values, spin):
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
        print(
            "                                               ",
            "Temperatura:",
            temperature,
            "Campo magnético externo:",
            h,
            end="\r",
        )
        rede, order, e = ferromagnetic_simulation(size, cycles, temperature, h, spin)
        magnetizations = np.append(magnetizations, np.sum(order) / size)
    magnetizations /= size**2
    return magnetizations


def calculate_curie_temperature(temperatures, m, sus, e, c):
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
    temperatures[sus == max_sus]

    max_c = max(c)
    temperatures[c == max_c]

    output = np.array([temperatures, m, sus, e, c]).transpose()
    np.savetxt(file_name, output, delimiter="\t")

    print(
        Fore.BLUE
        + f"Temperatura de Curie estimada a partir da suscetibilidade magnética: {temperatures[sus == max_sus][0]}"
    )
    print(
        f"Temperatura de Curie estimada a partir da capacidade calorífica: {temperatures[c == max_c][0]}"
        + Fore.RESET
    )
    moveFile(file_name)


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
        magnetizations = np.append(
            magnetizations,
            hysteresis_calc_varying_h(temperature, size, mc_cycles, h_values, spin),
        )
    return magnetizations


def print_parameters(size, mc_cycles, temperature, h, spin):
    """
    Imprime os parâmetros da simulação na consola.

    Args:
    size: Tamanho da rede
    mc_cycles: Número de ciclos de Monte Carlo
    temperature: Temperatura
    h: Campo magnético externo
    spin: Spin da rede
    """
    print(Fore.CYAN + f"Tamanho da rede: {size}" + Fore.RESET)
    print(Fore.GREEN + f"Ciclos de Monte Carlo: {mc_cycles}" + Fore.RESET)
    print(Fore.CYAN + f"Temperatura: {temperature}" + Fore.RESET)
    print(Fore.GREEN + f"Campo mangnético externo: {h}" + Fore.RESET)
    print(Fore.CYAN + f"Spin da rede: {spin}" + Fore.RESET)


def plot_grid(grid):
    """
    Plota a rede em 3D.

    Args:
    - rede: Rede com os spins
    """
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    x, y, z = np.where(grid == 1)
    ax.scatter(x, y, z, c="r", marker="o")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    plt.show()
    file_name = "rede.svg"
    fig.savefig(file_name, dpi=1200)
    moveFile(file_name)


def plot_graphs(order, e):
    """
    Plota os gráficos da ordem e da energia.

    Args:
    - order: ordem
    - e: Energia
    """
    plt.plot(order)
    file_name = "ordem.svg"
    plt.savefig(file_name, dpi=1200)
    moveFile(file_name)
    plt.show()

    plt.plot(e)
    file_name = "energia.svg"
    plt.savefig(file_name, dpi=1200)
    moveFile(file_name)
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
    axs[0, 1].plot(temps, sus)
    axs[0, 1].set_title("χ vs t")
    axs[1, 0].plot(temps, e)
    axs[1, 0].set_title("e vs t")
    axs[1, 1].plot(temps, c)
    axs[1, 1].set_title("C vs t")
    fig.tight_layout()
    file_name = "propriedades.svg"
    plt.savefig(file_name, dpi=1200)
    moveFile(file_name)
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
    file_name = "histerese.svg"
    plt.savefig(file_name, dpi=1200)
    moveFile(file_name)
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
    file_name = "histerese para varias temperaturas.svg"
    plt.savefig(file_name, dpi=1200)
    moveFile(file_name)
    plt.show()


def moveFile(file_name):
    """
    Move um ficheiro para a pasta "Plots e Resultados".

    Args:
    - file_name: Nome do ficheiro
    """
    file_location = os.getcwd()

    script_location = os.path.abspath(__file__)
    current_dir = os.path.dirname(script_location)
    destination_dir = os.path.join(current_dir, "Plots e Resultados")

    if not os.path.exists(destination_dir):
        os.makedirs(destination_dir)

    current_file = os.path.join(file_location, file_name)
    destination_file = os.path.join(destination_dir, file_name)

    shutil.move(current_file, destination_file)

    print(Fore.GREEN + f"Ficheiro {file_name} salvo em {destination_file}" + Fore.RESET)


def timer(signal):
    """
    Função temporizador.

    Args:
    - signal: Sinal de paragem
    """
    start_time = time.time()
    while not signal.is_set():
        seconds = int(time.time() - start_time)
        print(
            "Temporizador: {} segundos".format(seconds),
            end="\r",
        )
        time.sleep(1)


# Parâmetros
spin = -1
temperature = 5.5
h = 0.0
size = 10
mc_cycles = 10000
times = np.array([])

print_parameters(size, mc_cycles, temperature, h, spin)

signal = threading.Event()
thread_timer = threading.Thread(target=timer, args=(signal,))
thread_timer.start()
start = time.time()
grid, order, e = ferromagnetic_simulation(size, mc_cycles, temperature, h, spin)
end = time.time()
signal.set()
thread_timer.join()
elapsed_time = end - start
times = np.append(times, elapsed_time)
print("Tempo de execução a fazer a simulação inicial: ", elapsed_time / 60, "minutos")
plot_grid(grid)
plot_graphs(order, e)

# Temperaturas para calcular as propriedades
temperatures = np.arange(0.5, 5.5, 0.1)

signal = threading.Event()
thread_timer = threading.Thread(target=timer, args=(signal,))
thread_timer.start()
start = time.time()
m, sus, e, c = temperatureSimulation(
    temperatures, size, (int)(mc_cycles * 0.1), h, spin
)
end = time.time()
signal.set()
thread_timer.join()
elapsed_time = end - start
times = np.append(times, elapsed_time)
print(
    "Tempo de execução a fazer a calcular as propriedades:",
    elapsed_time / 60,
    "minutos",
)
plot_ferro_graph(m, sus, e, c, temperatures)

calculate_curie_temperature(temperatures, m, sus, e, c)

# Campo magnético externo para calcular a histerese
h_max = 5
h_values = np.linspace(-h_max, h_max, (h_max * 2) + 1)

# Temperaturas para calcular o magnetismo para múltiplas temperaturas variando o campo magnético externo
magnetizationsTemperatures = np.array([0.5, 2.4, 2.5, 2.6, 4.5])

signal = threading.Event()
thread_timer = threading.Thread(target=timer, args=(signal,))
thread_timer.start()
start = time.time()
magnetizations = hysteresis_calc_varying_h(
    temperature, size, (int)(mc_cycles * 0.1), h_values, spin
)
end = time.time()
signal.set()
thread_timer.join()
elapsed_time = end - start
times = np.append(times, elapsed_time)
print("Tempo de execução a fazer a calcular a histerese:", elapsed_time / 60, "minutos")
plot_hysteresis(temperature, h_values, magnetizations)

signal = threading.Event()
thread_timer = threading.Thread(target=timer, args=(signal,))
thread_timer.start()
start = time.time()
magnetism_for_mult_temps = calc_magnetism_for_mult_temps(
    magnetizationsTemperatures, (int)(mc_cycles * 0.1), h_values, size, spin
)
end = time.time()
signal.set()
thread_timer.join()
elapsed_time = end - start
times = np.append(times, elapsed_time)
print(
    "Tempo de execução a fazer a calcular o magnetismo para múltiplas temperaturas variando o campo magnético externo:",
    elapsed_time / 60,
    "minutos",
)
plot_magnetism_for_mult_temps(
    magnetizationsTemperatures, h_values, magnetism_for_mult_temps
)

print(
    Fore.MAGENTA + "Tempo total de execução:",
    np.sum(times) / 60,
    "minutos" + Fore.RESET,
)
