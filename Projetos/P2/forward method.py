import math
from colorama import Fore
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

"""
SMCEF - 2024 - P2 - Reentry of a Space Capsule
@author: Alexandre Tashchuk | 62568
@author: José Pedro Pires Costa | 62637
@author: Carolina dos Santos Saraiva | 68839
"""

# Constants
g = 10  # gravitational acceleration (m/s^2)
Cd = 1.2  # drag coefficient
Cl = 1.0  # lift coefficient
A = 4 * np.pi  # cross-sectional area (m^2)
m = 12000  # mass of the module (kg)
dt = 0.1  # time step (s)
x = 0.0  # horizontal position (m)
y = 130000.0  # altitude (m)
Cdp = 1.0  # drag coefficient of the parachute
Ap = 301.0  # cross-sectional area of the parachute (m^2)


def calc_v0_components(v0, alpha):
    """
    Calculate the initial horizontal and vertical velocities

    Parameters
    ----------
    v0 : float
        The initial velocity
    alpha : float
        The downward angle with the horizontal

    Returns
    -------
    vx : float
        The initial horizontal velocity
    vy : float
        The initial vertical velocity
    """
    alphaRad = math.radians(alpha)

    vx = v0 * math.cos(alphaRad)  # Initial horizontal velocity
    vy = -(v0 * math.sin(alphaRad))  # Initial vertical velocity

    return vx, vy


def get_info_from_file(filename):
    """
    Get density and altitude from a file

    Parameters
    ----------
    filename : str
        The name of the file to be read

    Returns
    -------
    altitude : numpy array
        The altitude values
    density : numpy array
        The density values
    """
    altitude = np.array([])
    density = np.array([])
    with open(filename, "r") as file:
        for line in file:
            values = line.strip().split("\t")
            altitude = np.append(altitude, float(values[0]))
            density = np.append(density, float(values[1]))
    return altitude, density


def exponential_model(x, A, B):
    """
    Exponential model

    Parameters
    ----------
    x : float
        The independent variable
    A : float
        The first parameter of the model
    B : float
        The second parameter of the model

    Returns
    -------
    float
        The value of the model at x
    """
    return A * np.exp(B * x)


def get_exponential_fit(altitude, density):
    """
    Get the exponential fit

    Parameters
    ----------
    altitude : numpy array
        The altitude values
    density : numpy array
        The density values

    Returns
    -------
    fit_altitude : numpy array
        The altitude values of the fit
    fit_density : numpy array
        The density values of the fit
    """
    altitude_norm = altitude / np.max(altitude)

    initial_guess = (1e-3, -1)

    params, _ = curve_fit(exponential_model, altitude_norm, density, p0=initial_guess)
    A, B = params

    # Extend the fit_altitude range to 150000 meters
    fit_altitude_range = 150000
    fit_altitude_norm = np.linspace(
        min(altitude_norm), fit_altitude_range / np.max(altitude), 500
    )
    fit_density = exponential_model(fit_altitude_norm, A, B)

    fit_altitude = fit_altitude_norm * np.max(altitude)

    return fit_altitude, fit_density


def air_density(altitude, filename):
    """
    Get the air density at a given altitude

    Parameters
    ----------
    altitude : float
        The altitude
    filename : str
        The name of the file to be read

    Returns
    -------
    float
        The air density at the given altitude
    """
    altitudeArray = np.array(
        [
            -1000,
            0,
            1000,
            2000,
            3000,
            4000,
            5000,
            6000,
            7000,
            8000,
            9000,
            10000,
            15000,
            20000,
            25000,
            30000,
            40000,
            50000,
            60000,
            70000,
            80000,
        ]
    )
    densityArray = np.array(
        [
            1.347,
            1.225,
            1.112,
            1.007,
            0.9093,
            0.8194,
            0.7364,
            0.6601,
            0.59,
            0.5258,
            0.4671,
            0.4135,
            0.1948,
            0.08891,
            0.04008,
            0.01841,
            0.003996,
            0.001027,
            0.0003097,
            0.00008283,
            0.00001846,
        ]
    )
    fit_altitude, fit_density = get_exponential_fit(altitudeArray, densityArray)
    density = np.interp(altitude, fit_altitude, fit_density)
    return density


def drag_force(v, rho, Cd, A):
    return -0.5 * Cd * A * rho * v**2


def lift_force(v, rho, Cl, A):
    return 0.5 * Cl * A * rho * v**2


def print_parameters(v0, alpha):
    """
    Print the initial parameters

    Parameters
    ----------
    v0 : float
        The initial velocity
    alpha : float
        The downward angle with the horizontal
    """
    print(Fore.CYAN + f"Velocidade inicial: {v0} m/s" + Fore.RESET)
    print(Fore.GREEN + f"Ângulo de entrada: {alpha} graus" + Fore.RESET + "\n")


def simulation(vx, vy, x, y, Cd, Cl, A, Cdp, Ap, filename, mode):
    time = 0
    positions = [(x, y)]
    velocities = [(vx, vy)]
    drag_coefficient = Cd
    area = A
    parachute = False
    deploy_position = None
    for i in range(int(5000 / dt)):
        v = np.sqrt(vx**2 + vy**2)
        rho = air_density(y, filename)

        if y <= 1000 and v <= 100 and not parachute:
            if mode == "manual" or mode == "fast":
                print(Fore.CYAN + "Abrindo paraquedas!" + Fore.RESET + "\n")
            deploy_position = (x, y)
            drag_coefficient += Cdp
            area += Ap
            parachute = True

        Fd = drag_force(v, rho, drag_coefficient, area)
        Fl = lift_force(v, rho, Cl, area)

        ax = Fd * np.cos(math.atan2(vy, vx)) / m
        if parachute:
            ay = (Fd * np.sin(math.atan2(vy, vx))) / m - g
        else:
            ay = (Fd * np.sin(math.atan2(vy, vx)) + Fl) / m - g

        vx += ax * dt
        vy += ay * dt
        x += vx * dt
        y += vy * dt

        positions.append((x, y))
        velocities.append((vx, vy))

        time += dt

        if y <= 0:
            break

    if mode == "manual" or mode == "fast":
        print(
            Fore.CYAN + f"Tempo de reentrada: {time / 60} minutos" + "\n" + Fore.RESET
        )

    return positions, velocities, time, deploy_position


def calculate_horizontal_distance(x, y, mode):
    """
    Calculate the horizontal distance projected on the Earth's surface.

    Parameters
    ----------
    x : list
        The x values of the trajectory
    y : list
        The y values of the trajectory
    R_earth : float
        The radius of the Earth in kilometers

    Returns
    -------
    distance : float
        The horizontal distance projected on the Earth's surface
    """
    R_earth = 6371
    theta = 0
    n = len(x)

    for i in range(1, n):
        theta_i = (x[i] - x[i - 1]) / (R_earth + y[i])
        theta += theta_i

    distance = R_earth * theta

    if mode == "manual" or mode == "fast":
        if distance <= 2500 or distance >= 4500:
            print(Fore.RED + f"Distância horizontal: {distance} km" + Fore.RESET)
        else:
            print(Fore.GREEN + f"Distância horizontal: {distance} km" + Fore.RESET)

    return distance


def calculate_g_value(velocities, time, mode):
    """
    Calculate the total acceleration and the g value.

    Parameters
    ----------
    velocities : list
        The velocities of the object
    time : float
        The time of the simulation
    mode : str
        The mode of the simulation

    Returns
    -------
    total_acceleration : float
        The total acceleration of the object
    g_value : float
        The g value of the object
    """
    n = len(velocities)
    total_acceleration = 0
    g_value = 0

    for i in range(1, n):
        vx_i, vy_i = velocities[i]
        vx_i_1, vy_i_1 = velocities[i - 1]
        ax_i = (vx_i - vx_i_1) / dt
        ay_i = (vy_i - vy_i_1) / dt
        total_acceleration += math.sqrt(ax_i**2 + ay_i**2)

    g_value = total_acceleration / time / g

    if mode == "manual" or mode == "fast":
        if g_value >= 15 or g_value <= 1:
            print(Fore.RED + f"Valor de g: {g_value}" + Fore.RESET)
        else:
            print(Fore.GREEN + f"Valor de g: {g_value}" + Fore.RESET)

    return total_acceleration, g_value


def calculate_final_velocity(vx, vy, mode):
    """
    Calculate the final velocity of the object.

    Parameters
    ----------
    vx : float
        The horizontal velocity
    vy : float
        The vertical velocity

    Returns
    -------
    final_velocity : float
        The final velocity of the object
    """
    final_velocity = math.sqrt(vx**2 + vy**2)

    if mode == "manual" or mode == "fast":
        if final_velocity >= 25:
            print(Fore.RED + f"Velocidade final: {final_velocity} m/s" + Fore.RESET)
        else:
            print(Fore.GREEN + f"Velocidade final: {final_velocity} m/s" + Fore.RESET)

    return final_velocity


def plot_trajectory(x_forward, y_forward, x_backward, y_backward, deploy_position):
    # Plot trajectories
    plt.figure()
    plt.plot(x_forward, y_forward, label="Forward Method")
    # plt.plot(x_backward, y_backward_km, label="Backward Method")
    if deploy_position:
        plt.plot(
            deploy_position[0] / 1000,
            deploy_position[1] / 1000,
            "ro",
            label="Deploy Position",
        )
    plt.xlabel("Horizontal Distance (km)")
    plt.ylabel("Altitude (km)")
    plt.legend()
    plt.title("Reentry Trajectory")
    plt.show()


def run_simulation(v0, alpha, mode):
    filename = "Projetos/P2/airdensity - students.txt"
    success = True

    v0x, v0y = calc_v0_components(v0, alpha)

    positions, velocities, time, deploy_position = simulation(
        v0x, v0y, x, y, Cd, Cl, A, Cdp, Ap, filename, mode
    )

    x_forward, y_forward = zip(*positions)
    # x_backward, y_backward = zip(*positions_backward)

    # Convert altitude from meters to kilometers
    x_forward_km = [distance / 1000 for distance in x_forward]
    y_forward_km = [altitude / 1000 for altitude in y_forward]
    # x_backward_km = [distance / 1000 for distance in x_backward]
    # y_backward_km = [altitude / 1000 for altitude in y_backward]

    # Calculate the horizontal distance
    h_distance = calculate_horizontal_distance(x_forward_km, y_forward_km, mode)

    # Calculate the final velocity
    final_velocity = calculate_final_velocity(
        velocities[-1][0], velocities[-1][1], mode
    )

    # Calculate the total acceleration and the g value
    total_acceleration, g_value = calculate_g_value(velocities, time, mode)

    if (
        (h_distance <= 2500 or h_distance >= 4500)
        or g_value >= 15
        or g_value <= 1
        or final_velocity >= 25
        or final_velocity <= 0
    ):
        success = False

    if mode == "manual" or mode == "fast":
        plot_trajectory(x_forward_km, y_forward_km, None, None, deploy_position)
        if not success:
            print("\n" + Fore.RED + "Simulação falhou!!" + Fore.RESET)
        else:
            print("\n" + Fore.GREEN + "Simulação aceite!!" + Fore.RESET)
    return success


def run_automatic(mode):
    """
    Run the simulation with a range of values for v0 and alpha

    Parameters
    ----------
    mode : str
        The mode of the simulation
    """
    v0_range = list(range(0, 15001, 100))  # should start in 0
    alpha_range = list(range(16))

    success_count = 0
    simulation_count = 0
    valid_parameters = np.empty((0, 2), dtype=int)

    for v0 in v0_range:
        for alpha in alpha_range:
            print(
                "Sucessos: {} | Simulações: {}".format(success_count, simulation_count),
                end="\r",
            )

            v0 = int(v0)
            alpha = int(alpha)
            success = run_simulation(v0, alpha, mode)
            simulation_count += 1
            if success:
                success_count += 1
                valid_parameters = np.append(valid_parameters, [[v0, alpha]], axis=0)

    print(Fore.GREEN + f"Sucessos: {success_count}" + Fore.RESET)


def check_parameters(v0, alpha, mode):
    if mode == "manual":
        if not v0.isdigit() or not alpha.isdigit():
            print(Fore.RED + "Input inválido!!" + Fore.RESET + "\n")
            exit(1)
    if int(alpha) > 15 or int(alpha) < 0 and int(v0) < 0 or int(v0) > 15000:
        print(Fore.RED + "Input inválido!!" + Fore.RESET + "\n")
        exit(1)


def handle_simulation_mode(user_input):
    if user_input == "1":
        mode = "automatic"
        print(Fore.MAGENTA + "Modo automático selecionado.")
        print("Correndo a simulação..." + Fore.RESET + "\n")
        run_automatic(mode)
    elif user_input == "2":
        mode = "manual"
        print(Fore.MAGENTA + "Modo manual selecionado.")
        v0 = input(
            Fore.CYAN + "Por favor insira a velocidade inicial (m/s): " + Fore.RESET
        )
        alpha = input(
            Fore.GREEN + "Por favor insira ângulo de entrada (graus): " + Fore.RESET
        )
        check_parameters(v0, alpha, mode)
        print(Fore.MAGENTA + "Correndo a simulação..." + Fore.RESET + "\n")
        print_parameters(int(v0), int(alpha))
        run_simulation(int(v0), int(alpha), mode)
    elif user_input == "3":
        mode = "fast"
        print(Fore.MAGENTA + "Modo rápido selecionado.")
        v0 = 11000
        alpha = 6
        check_parameters(v0, alpha, mode)
        print("Correndo a simulação..." + Fore.RESET + "\n")
        print_parameters(int(v0), int(alpha))
        run_simulation(v0, alpha, mode)
    else:
        print(Fore.RED + "Input inválido!!" + Fore.RESET + "\n")
        exit(1)


def main():
    print(
        Fore.MAGENTA
        + "Simulação de reentrada do modulo espacial | SMCEF 23/24 | P2 | FCT-UNL"
    )
    print("======================================================================")
    print("Insira o modo de simulação:")
    print(Fore.BLUE + "1 - Automático:")
    print(
        "Este modo executará todos os valores para v0 entre 0 e 15000 m/s e todos os valores para alpha entre 0 e 15 graus."
    )
    print(Fore.GREEN + "2 - Manual")
    print("Este modo solicitará os valores de v0 e alpha.")
    print(Fore.BLUE + "3 - Rápido")
    print(
        "Este modo executará a simulação com os valores de v0 e alpha fornecidos no código."
    )
    user_input = input(
        "\n" + Fore.YELLOW + "Por favor insira o modo (1/2/3): " + Fore.RESET
    )

    handle_simulation_mode(user_input)


if __name__ == "__main__":
    main()
