import math
import os
import time
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from multiprocessing import Pool
from colorama import Fore

"""
SMCEF - 2024 - P2 - Reentry of a Space Capsule
@author: Alexandre Tashchuk | 62568
@author: José Pedro Pires Costa | 62637
@author: Carolina dos Santos Saraiva | 68839
"""

# Constants
g0 = 10  # gravitational acceleration (m/s^2)
Cd = 1.2  # drag coefficient
Cl = 1.0  # lift coefficient
A = 4 * np.pi  # cross-sectional area (m^2)
m = 12000  # mass of the module (kg)
dt = 0.1  # time step (s)
x = 0.0  # horizontal position (m)
y = 130000.0  # altitude (m)
Cdp = 1.0  # drag coefficient of the parachute
Ap = 301.0  # cross-sectional area of the parachute (m^2)
R_earth = 6371  # Earth's radius (km)
M_earth_kg = 5.97e24  # Earth's mass (kg)
G = 6.67430e-11  # Gravitational constant (m^3 kg^-1 s^-2)

results_file_forward = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "accepted_simulations_forward_method.tsv",
)

results_file_backward = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "accepted_simulations_backward_method.tsv",
)

results_file_forward_and_backward = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "accepted_simulations_forward_and_backward_method.tsv",
)


def calc_v0_components(v0, alpha):
    """
    Calculate the initial horizontal and vertical velocities

    Parameters
    ----------
    v0 : float
        The initial velocity
    alpha : float
        The downward angle with the horizon

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

    fit_altitude_range = 150000
    fit_altitude_norm = np.linspace(
        min(altitude_norm), fit_altitude_range / np.max(altitude), 500
    )
    fit_density = exponential_model(fit_altitude_norm, A, B)

    fit_altitude = fit_altitude_norm * np.max(altitude)

    return fit_altitude, fit_density


def air_density(altitude):
    """
    Get the air density at a given altitude

    Parameters
    ----------
    altitude : float
        The altitude

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
    """
    Calculate the drag force

    Parameters
    ----------
    v : float
        The velocity
    rho : float
        The air density
    Cd : float
        The drag coefficient
    A : float
        The cross-sectional area

    Returns
    -------
    float
        The drag force
    """
    return -0.5 * Cd * A * rho * v


def lift_force(v, rho, Cl, A):
    """
    Calculate the lift force

    Parameters
    ----------
    v : float
        The velocity
    rho : float
        The air density
    Cl : float
        The lift coefficient
    A : float
        The cross-sectional area

    Returns
    -------
    float
        The lift force
    """
    return 0.5 * Cl * A * rho * v**2


def calculate_acceleration_due_to_gravity(y):
    """
    Calculate the acceleration due to gravity

    Parameters
    ----------
    y : float
        The altitude

    Returns
    -------
    float
        The acceleration due to gravity
    """
    R_earth_m = 6371 * 1000
    g = G * M_earth_kg / (R_earth_m + y) ** 2
    return g


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


def forward_simulation(vx, vy, x, y, mode):
    """
    Forward method simulation

    Parameters
    ----------
    vx : float
        The initial horizontal velocity
    vy : float
        The initial vertical velocity
    x : float
        The initial horizontal position
    y : float
        The initial altitude
    Cd : float
        The drag coefficient
    Cl : float
        The lift coefficient
    A : float
        The cross-sectional area
    Cdp : float
        The drag coefficient of the parachute
    Ap : float
        The cross-sectional area of the parachute
    mode : str
        The mode of the simulation

    Returns
    -------
    positions : list
        The positions of the object
    velocities : list
        The velocities of the object
    time : float
        The time of the simulation
    deploy_position : tuple
        The deploy position of the parachute
    """
    time = 0
    positions = [(x, y)]
    velocities = [(vx, vy)]
    accelerations = [(0, 0)]
    drag_coefficient = Cd
    area = A
    parachute = False
    deploy_position = None
    for _ in range(int(5000 / dt)):
        v = np.sqrt(vx**2 + vy**2)
        rho = air_density(y)
        g = calculate_acceleration_due_to_gravity(y)

        if y <= 1000 and v <= 100 and not parachute:
            if mode == "manual" or mode == "fast":
                print(
                    Fore.CYAN
                    + "Abrindo paraquedas! (Forward Method)"
                    + Fore.RESET
                    + "\n"
                )  # indicar que é a forward
            deploy_position = (x, y)
            drag_coefficient += Cdp
            area = Ap
            parachute = True

        Fd = drag_force(v, rho, drag_coefficient, area)
        Fl = lift_force(v, rho, Cl, area)

        ax = (Fd * vx) / (m)
        if parachute:
            ay = ((Fd * vy) / (m)) - g
        else:
            ay = ((Fd * vy) / (m)) + (Fl / m) - g

        vx += ax * dt
        vy += ay * dt
        x += vx * dt
        y += vy * dt

        positions.append((x, y))
        velocities.append((vx, vy))
        accelerations.append((ax, ay))

        time += dt

        if y <= 0:
            break

    if mode == "manual" or mode == "fast":
        print(
            Fore.CYAN + f"Tempo de reentrada: {time / 60} minutos" + "\n" + Fore.RESET
        )

    return positions, velocities, accelerations, time, deploy_position, parachute


def backward_simulation(vx, vy, x, y, mode):
    time = 0
    positions = [(x, y)]
    velocities = [(vx, vy)]
    accelerations = [(0, 0)]
    drag_coefficient = Cd
    area = A
    parachute = False
    deploy_position = None

    def f(vx, vy, x, y, t, parachute):
        v = math.sqrt(vx**2 + vy**2)
        rho = air_density(y)
        Fd = drag_force(v, rho, drag_coefficient, area)
        Fl = lift_force(v, rho, Cl, area)
        g = calculate_acceleration_due_to_gravity(y)

        ax = (Fd * vx) / (m)
        if parachute:
            ay = ((Fd * vy) / (m)) - g
        else:
            ay = ((Fd * vy) / (m)) + (Fl / m) - g

        return ax, ay

    while y > 0:
        time += dt

        v = math.sqrt(vx**2 + vy**2)
        if y <= 1000 and v <= 100 and not parachute:
            if mode == "manual" or mode == "fast":
                print(
                    Fore.CYAN
                    + "Abrindo paraquedas! (Backward Method)"
                    + Fore.RESET
                    + "\n"
                )  # indicar que é a forward
            deploy_position = (x, y)
            drag_coefficient += Cdp
            area = Ap
            parachute = True

        x_prev, y_prev = x, y
        vx_prev, vy_prev = vx, vy

        ax_prev, ay_prev = f(vx_prev, vy_prev, x_prev, y_prev, time, parachute)

        J = np.array([[1, 0, -dt, 0], [0, 1, 0, -dt], [0, 0, 1, 0], [0, 0, 0, 1]])

        b = np.array(
            [
                x_prev + vx_prev * dt,
                y_prev + vy_prev * dt,
                vx_prev + ax_prev * dt,
                vy_prev + ay_prev * dt,
            ]
        )

        # Solve the system
        result = np.linalg.solve(J, b)
        x_next, y_next, vx_next, vy_next = result

        x, y = x_next, y_next
        vx, vy = vx_next, vy_next

        ax, ay = f(vx, vy, x, y, time, parachute)

        positions.append((x, y))
        velocities.append((vx, vy))
        accelerations.append((ax, ay))

        if y <= 0:
            break

        if mode == "manual" or mode == "fast":
            print(
                Fore.CYAN
                + f"Tempo de reentrada: {time / 60} minutos"
                + "\n"
                + Fore.RESET
            )

    return positions, velocities, accelerations, time, deploy_position, parachute


def calculate_horizontal_distance(x, y, mode):
    """
    Calculate the horizontal distance

    Parameters
    ----------
    x : list
        The horizontal positions
    y : list
        The altitudes
    mode : str
        The mode of the simulation

    Returns
    -------
    distance : float
        The horizontal distance
    """
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


def calculate_g_value(velocities, time, positions, mode):
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
        altitude = positions[i][1]
        g = calculate_acceleration_due_to_gravity(altitude)
        ay_i += g
        total_acceleration += math.sqrt(ax_i**2 + ay_i**2)

    g_value = total_acceleration / time / g0

    if mode == "manual" or mode == "fast":
        if g_value >= 15 or g_value <= 1:
            print(Fore.RED + f"Valor de g: {g_value}" + Fore.RESET)
        else:
            print(Fore.GREEN + f"Valor de g: {g_value}" + Fore.RESET)

    return total_acceleration, g_value


def calculate_final_velocity(vx, vy, mode):
    """
    Calculate the final velocity

    Parameters
    ----------
    vx : float
        The final horizontal velocity
    vy : float
        The final vertical velocity
    mode : str
        The mode of the simulation

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


def plot_trajectory_x_y(x_forward, y_forward, deploy_position, v0, alpha):
    """
    Plot the trajectory of the object

    Parameters
    ----------
    x_forward : list
        The horizontal positions
    y_forward : list
        The altitudes
    deploy_position : tuple
        The deploy position of the parachute
    v0 : float
        The initial velocity
    alpha : float
        The downward angle with the horizon
    """
    plt.figure()
    plt.plot(x_forward, y_forward, label="Forward Method")
    if deploy_position:
        plt.plot(
            deploy_position[0] / 1000,
            deploy_position[1] / 1000,
            "ro",
            label="Deploy Position",
        )
    plt.xlabel("Distância horizontal (km)")
    plt.ylabel("Altitude (km)")
    plt.legend()
    plt.title(f"Trajetória de reentrada | v0 = {v0} m/s | alpha = {alpha} graus")
    plt.show()


def plot_trajectory_time_y(
    x_forward, y_forward, time, deploy_position
):  # plot_trajectory_forward_time_y
    """
    Plot the trajectory of the object

    Parameters
    ----------
    x_forward : list
        The horizontal positions
    y_forward : list
        The altitudes
    time : float
        The time of the simulation
    deploy_position : tuple
        The deploy position of the parachute
    """
    n = len(x_forward)
    time_range = np.linspace(0, time, n)

    plt.figure()
    plt.plot(time_range, y_forward, label="Forward Method")
    if deploy_position:
        plt.plot(
            time_range[-1],
            deploy_position[1] / 1000,
            "ro",
            label="Deploy Position",
        )
    plt.xlabel("Tempo (s)")
    plt.ylabel("Altitude (km)")
    plt.legend()
    plt.title("Trajetória de reentrada")
    plt.show()


def plot_velocities(vx, vy, time):  # plot_velocities_forward
    """
    Plot the velocities of the object

    Parameters
    ----------
    vx : list
        The horizontal velocities
    vy : list
        The vertical velocities
    time : float
        The time of the simulation
    """
    n = len(vx)
    time_range = np.linspace(0, time, n)

    plt.figure()
    plt.plot(time_range, vx, label="Vx")
    plt.plot(time_range, vy, label="Vy")
    plt.xlabel("Tempo (s)")
    plt.ylabel("Velocidade (m/s)")
    plt.legend()
    plt.title("Velocidades do objeto")
    plt.show()


# def plot_trajectory_backward_time_y(x_backward, y_backward, deploy_position, v0, alpha):
#   """
#   Plot the trajectory of the object

#   Parameters
#   ----------
#   x_backward : list
#       The horizontal positions
#   y_backward : list
#       The altitudes
#   deploy_position : tuple
#       The deploy position of the parachute
#   v0 : float

#   alpha : float
#       The downward angle with the horizon
#   """
#   plt.figure()
#   plt.plot(x_backward, y_backward, label="Backward Method")
#   if deploy_position:
#       plt.plot(
#           deploy_position[0] / 1000,
#           deploy_position[1] / 1000,
#           "ro",
#           label="Deploy Position",
#       )
#   plt.xlabel("Distância horizontal (km)")
#   plt.ylabel("Altitude (km)")
#   plt.legend()
#   plt.title(f"Trajetória de reentrada")
#   plt.show()

# def plot_velocities_backward(vx, vy, time_backward):
#   """
#   Plot the velocities of the object

#   Parameters
#   ----------
#   vx : list
#       The horizontal velocities
#   vy : list
#       The vertical velocities
#   time_backward : float
#       The time of the simulation
#   """
#   n = len(vx)
#   time_range = np.linspace(0, time_backward, n)

#   plt.figure()
#   plt.plot(time_range, vx, label="Vx")
#   plt.plot(time_range, vy, label="Vy")
#   plt.xlabel("Tempo (s)")
#   plt.ylabel("Velocidade (m/s)")
#   plt.legend()
#   plt.title("Velocidades do objeto")
#   plt.show()


def simulation_handler(v0, alpha, mode):
    """
    Run the simulation

    Parameters
    ----------
    v0 : float
        The initial velocity
    alpha : float
        The downward angle with the horizon
    mode : str
        The mode of the simulation

    Returns
    -------
    success : bool
        True if the simulation is valid, False otherwise
    """
    success_forward = True
    success_backward = True

    v0x, v0y = calc_v0_components(v0, alpha)

    (
        positions_forward,
        velocities_forward,
        accelerations_forward,
        time_forward,
        deploy_position_forward,
        parachute_forward,
    ) = forward_simulation(v0x, v0y, x, y, mode)

    (
        positions_backward,
        velocities_backward,
        accelerations_backward,
        time_backward,
        deploy_position_backward,
        parachute_backward,
    ) = backward_simulation(v0x, v0y, x, y, mode)

    x_forward, y_forward = zip(*positions_forward)
    x_backward, y_backward = zip(*positions_backward)

    x_forward_km = [distance / 1000 for distance in x_forward]
    y_forward_km = [altitude / 1000 for altitude in y_forward]
    x_backward_km = [distance / 1000 for distance in x_backward]
    y_backward_km = [altitude / 1000 for altitude in y_backward]

    h_distance_forward = calculate_horizontal_distance(x_forward_km, y_forward_km, mode)

    final_velocity_forward = calculate_final_velocity(
        velocities_forward[-1][0], velocities_forward[-1][1], mode
    )

    total_acceleration_forward, g_value_forward = calculate_g_value(
        velocities_forward, time_forward, positions_forward, mode
    )

    if (
        h_distance_forward <= 2500
        or h_distance_forward >= 4500
        or g_value_forward >= 15
        or g_value_forward <= 1
        or final_velocity_forward >= 25
        or final_velocity_forward <= 0
        or not parachute_forward
    ):
        success_forward = False

    # Backward simulation
    (
        positions_backward,
        velocities_backward,
        accelerations_backward,
        time_backward,
        deploy_position_backward,
        parachute_backward,
    ) = backward_simulation(v0x, v0y, x, y, mode)

    x_backward, y_backward = zip(*positions_backward)

    x_backward_km = [distance / 1000 for distance in x_backward]
    y_backward_km = [altitude / 1000 for altitude in y_backward]

    h_distance_backward = calculate_horizontal_distance(
        x_backward_km, y_backward_km, mode
    )

    final_velocity_backward = calculate_final_velocity(
        velocities_backward[-1][0], velocities_backward[-1][1], mode
    )

    total_acceleration_backward, g_value_backward = calculate_g_value(
        velocities_backward, time_backward, positions_backward, mode
    )

    if (
        h_distance_backward <= 2500
        or h_distance_backward >= 4500
        or g_value_backward >= 15
        or g_value_backward <= 1
        or final_velocity_backward >= 25
        or final_velocity_backward <= 0
        or not parachute_backward
    ):
        success_backward = False

    # Save the information to files
    if success_forward:
        save_info_to_file(
            v0,
            alpha,
            h_distance_forward,
            final_velocity_forward,
            g_value_forward,
            results_file_forward,
        )

    if success_backward:
        save_info_to_file(
            v0,
            alpha,
            h_distance_backward,
            final_velocity_backward,
            g_value_backward,
            results_file_backward,
        )

    if success_forward and success_backward:
        save_info_to_file(
            v0,
            alpha,
            h_distance_forward,
            final_velocity_forward,
            g_value_forward,
            results_file_forward_and_backward,
        )

    if mode == "manual" or mode == "fast":
        # plot_trajectory_x_y(x_forward_km, y_forward_km, deploy_position, v0, alpha)
        plot_trajectory_time_y(  # plot_trajectory_forward_time_y
            x_forward_km, y_forward_km, time_forward, deploy_position_forward
        )
        # plot_trajectory_backward_x_y(x_backward_km, y_backward_km, deploy_position, v0, alpha)
        plot_velocities(  # plot_velocities_forward
            [velocity[0] for velocity in velocities_forward],
            [velocity[1] for velocity in velocities_forward],
            time_forward,
        )
        # plot_velocities_backward(
        #   [velocity[0] for velocity in velocities_backward],
        #   [velocity[1] for velocity in velocities_backward],
        #   time_backward,
        # )
        if not success_forward or not success_backward:
            print("\n" + Fore.RED + "Simulação não aceite!!" + Fore.RESET)
        else:
            print("\n" + Fore.GREEN + "Simulação aceite!!" + Fore.RESET)

    return success_forward and success_backward


def delete_old_results_files():
    """
    Delete the old results files
    """
    if os.path.exists(results_file_forward):
        os.remove(results_file_forward)
    if os.path.exists(results_file_backward):
        os.remove(results_file_backward)
    if os.path.exists(results_file_forward_and_backward):
        os.remove(results_file_forward_and_backward)
    print(
        Fore.RED
        + "A apagar ficheiros de resultados já existentes!!"
        + Fore.RESET
        + "\n"
    )


def save_info_to_file(v0, alpha, distance, final_velocity, g_value, file):
    """
    Save the information to a file

    Parameters
    ----------
    v0 : float
        The initial velocity
    alpha : float
        The downward angle with the horizon
    distance : float
        The horizontal distance
    final_velocity : float
        The final velocity
    g_value : float
        The g value
    """
    with open(file, "a") as file:
        file.write(f"{v0}\t{alpha}\t{distance}\t{final_velocity}\t{g_value}\n")


def run_simulation_wrapper(params):
    """
    Wrapper for the simulation function

    Parameters
    ----------
    params : tuple
        The parameters of the simulation

    Returns
    -------
    v0 : float
        The initial velocity
    alpha : float
        The downward angle with the horizon
    success : bool
        True if the simulation is accepted, False otherwise
    """
    v0, alpha, mode = params
    return v0, alpha, simulation_handler(v0, alpha, mode)


def run_automatic(mode, n_processes, spacing):
    """
    Run the simulation in automatic mode

    Parameters
    ----------
    mode : str
        The mode of the simulation
    n_processes : int
        The number of processes to run concurrently
    spacing : int
        The spacing between the values of v0

    Returns
    -------
    success_count : int
        The number of successful simulations
    simulation_count : int
        The number of simulations
    valid_parameters : numpy array
        The valid parameters
    """
    delete_old_results_files()
    start = time.time()
    v0_range = list(range(0, 15001, spacing))
    alpha_range = list(range(16))

    success_count = 0
    simulation_count = 0
    valid_parameters = np.empty((0, 2), dtype=int)

    parameters = [(v0, alpha, mode) for v0 in v0_range for alpha in alpha_range]

    with Pool(n_processes) as pool:
        for v0, alpha, success in pool.imap(run_simulation_wrapper, parameters):
            simulation_count += 1
            if success:
                success_count += 1
                valid_parameters = np.append(valid_parameters, [[v0, alpha]], axis=0)

            print(
                "Sucessos: {} | Simulações: {}".format(success_count, simulation_count),
                end="\r",
            )
    end = time.time()
    elapsed_time = end - start
    print(
        Fore.GREEN
        + f"Simulação concluida com {success_count} simulações aceites!"
        + Fore.RESET
    )
    print(
        Fore.YELLOW
        + f"Tempo total de execução: {elapsed_time / 60} minutos"
        + Fore.RESET
    )


def check_parameters(v0, alpha, mode):
    """
    Check if the parameters are valid

    Parameters
    ----------
    v0 : str
        The initial velocity
    alpha : str
        The downward angle with the horizon
    mode : str
        The mode of the simulation
    """
    if mode == "manual":
        if not v0.isdigit() or not alpha.isdigit():
            print(Fore.RED + "Input inválido!!" + Fore.RESET + "\n")
            exit(1)
    if int(alpha) > 15 or int(alpha) < 0 and int(v0) < 0 or int(v0) > 15000:
        print(Fore.RED + "Input inválido!!" + Fore.RESET + "\n")
        exit(1)


def calculate_number_of_simulations(n):
    """
    Calculate the number of simulations

    Parameters
    ----------
    n : int
        The spacing between the values of v0

    Returns
    -------
    simulation_count : int
        The number of simulations

    """
    v0_range_length = 15000 // n + 1
    alpha_range_length = 16
    simulation_count = v0_range_length * alpha_range_length
    return simulation_count


def handle_simulation_mode(user_input):
    """
    Handle the simulation mode

    Parameters
    ----------
    user_input : str
        The mode selected by the user
    """
    if user_input == "1":
        mode = "automatic"
        print(Fore.MAGENTA + "Modo automático selecionado." + "\n" + Fore.RESET)
        print(
            Fore.RED
            + "Este modo irá apagar os ficheiros de resultados existentes quando começar!!"
            + "\n"
            + Fore.RESET
        )

        spacing = input(
            Fore.CYAN
            + "Por favor insira o espaçamento entre os valores de v0 (default = 100): "
            + Fore.RESET
        )
        if spacing == "":
            spacing = 100
        elif not spacing.isdigit() or int(spacing) <= 0 or int(spacing) > 15000:
            print(Fore.RED + "Input inválido!!" + Fore.RESET + "\n")
            exit(1)
        print("Espaçamento entre os valores de v0: " + str(spacing) + "\n")

        n_simulations = calculate_number_of_simulations(int(spacing))
        print(
            Fore.YELLOW
            + "Número de simulações a correr: "
            + str(n_simulations)
            + Fore.RESET
            + "\n"
        )

        n_processes = input(
            Fore.GREEN
            + "Por favor insira o número de processos menor que o número de simulações a correr (default = 5): "
            + Fore.RESET
        )
        if n_processes == "":
            n_processes = 5
        elif (
            not n_processes.isdigit()
            or int(n_processes) <= 0
            or int(n_processes) > n_simulations
        ):
            print(Fore.RED + "Input inválido!!" + Fore.RESET + "\n")
            exit(1)

        print("Número de processos concurrentes: " + str(n_processes) + "\n")

        print(Fore.MAGENTA + "Correndo a simulação..." + Fore.RESET + "\n")
        run_automatic(mode, int(n_processes), int(spacing))
    elif user_input == "2":
        mode = "manual"
        print(Fore.MAGENTA + "Modo manual selecionado." + "\n" + Fore.RESET)
        v0 = input(
            Fore.CYAN + "Por favor insira a velocidade inicial (m/s): " + Fore.RESET
        )
        alpha = input(
            Fore.GREEN + "Por favor insira ângulo de entrada (graus): " + Fore.RESET
        )
        check_parameters(v0, alpha, mode)
        print(Fore.MAGENTA + "Correndo a simulação..." + Fore.RESET + "\n")
        print_parameters(int(v0), int(alpha))
        simulation_handler(int(v0), int(alpha), mode)
    elif user_input == "3":
        mode = "fast"
        print(Fore.MAGENTA + "Modo rápido selecionado." + "\n" + Fore.RESET)
        v0 = 8000
        alpha = 1
        check_parameters(v0, alpha, mode)
        print(Fore.MAGENTA + "Correndo a simulação..." + Fore.RESET + "\n")
        print_parameters(int(v0), int(alpha))
        simulation_handler(v0, alpha, mode)
    else:
        print(Fore.RED + "Input inválido!!" + Fore.RESET + "\n")
        exit(1)


def main():
    """
    Main function
    """
    print(
        Fore.MAGENTA
        + "Simulação de reentrada do modulo espacial | SMCEF 23/24 | P2 | FCT-UNL"
    )
    print("======================================================================")
    print("Insira o modo de simulação:" + "\n" + Fore.RESET)
    print(Fore.BLUE + "1 - Automático:")
    print(
        "Este modo executará os valores para v0 entre 0 e 15000 m/s com espaçamento e todos os valores para alpha entre 0 e 15 graus."
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
    """
    Run the main function
    """
    main()
