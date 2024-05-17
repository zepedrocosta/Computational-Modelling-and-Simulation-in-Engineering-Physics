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
            "                                               ",
            "Temporizador : {} segundos".format(seconds),
            end="\r",
        )
        time.sleep(1)


temperature = 5.5
h = 0.0
size = 10
mc_cicles = 10000
times = np.array([])

signal = threading.Event()
thread_timer = threading.Thread(target=timer, args=(signal,))
thread_timer.start()
start = time.time()
rede, order, e = ferroSimul(size, mc_cicles, temperature, h)
end = time.time()
signal.set()
thread_timer.join()
elapsed_time = end - start
times = np.append(times, elapsed_time)
print("Tempo de execução a fazer a simulação inicial: ", elapsed_time, "segundos")
plot_grid(rede)
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

h_max = 5.0 
h_values = np.linspace(-h_max, h_max, (h_max * 2) + 1)

magnetizationsTemperatures = np.array([0.5, 2.4, 2.5, 2.6, 4.5])

start = time.time()
magnetizations = hysteresis_calc_varying_h(temperature, size, (int)(mc_cicles * 0.1), h_values)
end = time.time()
elapsed_time = end - start
times = np.append(times, elapsed_time)
print("Tempo de execução a fazer a calcular a histerese:", elapsed_time, "segundos")
plot_hysteresis(temperature, h_values, magnetizations)

signal = threading.Event()
thread_timer = threading.Thread(target=timer, args=(signal,))
thread_timer.start()
start = time.time()
magnetism_for_mult_temps = calc_magnetism_for_mult_temps(
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
    magnetizationsTemperatures, h_values, magnetism_for_mult_temps
)

print("Tempo total de execução:", np.sum(times), "segundos")
