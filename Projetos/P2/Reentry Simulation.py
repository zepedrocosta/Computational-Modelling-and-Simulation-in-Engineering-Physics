import math
import os
import time
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from multiprocessing import Pool
from colorama import Fore
from scipy.optimize import fsolve

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

# Paths to the results files
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
    mode : str
        The mode of the simulation

    Returns
    -------
    positions : list
        The positions of the object
    velocities : list
        The velocities of the object
    accelerations : list
        The accelerations of the object
    time : float
        The time of the simulation
    deploy_position : tuple
        The deploy position of the parachute
    parachute : bool
        True if the parachute is deployed, False otherwise
    """
    time = 0
    positions = [(x, y)]
    velocities = [(vx, vy)]
    accelerations = []
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
                )
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
            Fore.CYAN
            + f"Tempo de reentrada: {time / 60} minutos (Forward Method)."
            + "\n"
            + Fore.RESET
        )

    return positions, velocities, accelerations, time, deploy_position, parachute


def residual(v_next, vx, vy, dt, m, Cd, Cl, A, g, rho, parachute):
    v_mag = np.sqrt(v_next[0] ** 2 + v_next[1] ** 2)
    Fd = drag_force(v_mag, rho, Cd, A)
    Fl = lift_force(v_mag, rho, Cl, A) if not parachute else 0

    ax_next = (Fd * v_next[0]) / m
    ay_next = ((Fd * v_next[1]) / m - g) + (Fl / m if not parachute else 0)

    r0 = (v_next[0] - vx) / dt - ax_next
    r1 = (v_next[1] - vy) / dt - ay_next

    return np.array([r0, r1])


def jacobian(v_next, dt, m, Cd, Cl, A, g, rho, parachute):
    v_mag = np.sqrt(v_next[0] ** 2 + v_next[1] ** 2)
    Fd = drag_force(v_mag, rho, Cd, A)
    Fl = lift_force(v_mag, rho, Cl, A) if not parachute else 0

    dFd_dvx = Fd * v_next[0] / v_mag
    dFd_dvy = Fd * v_next[1] / v_mag
    dFl_dvx = Fl * v_next[0] / v_mag if not parachute else 0
    dFl_dvy = Fl * v_next[1] / v_mag if not parachute else 0

    j00 = 1 / dt - dFd_dvx / m
    j01 = -dFd_dvy / m
    j10 = -dFd_dvx / m
    j11 = 1 / dt - (dFd_dvy / m - g / m + dFl_dvy / m)

    return np.array([[j00, j01], [j10, j11]])


def backward_simulation(vx, vy, x, y, mode):
    time = 0
    positions = [(x, y)]
    velocities = [(vx, vy)]
    accelerations = []
    drag_coefficient = Cd
    area = A
    parachute = False
    deploy_position = None

    for _ in range(int(5000 / dt)):
        vx_next, vy_next = vx, vy
        x_next, y_next = x, y

        rho = air_density(y_next)
        g = calculate_acceleration_due_to_gravity(y_next)

        if y_next <= 1000 and np.sqrt(vx_next**2 + vy_next**2) <= 100 and not parachute:
            if mode == "manual" or mode == "fast":
                print(
                    Fore.CYAN
                    + "Abrindo paraquedas! (Backward Method)"
                    + Fore.RESET
                    + "\n"
                )
            deploy_position = (x_next, y_next)
            drag_coefficient = Cd + Cdp
            area = Ap
            parachute = True

        v_next = np.array([vx_next, vy_next])
        for _ in range(10):  # Newton-Raphson iterations
            r = residual(
                v_next, vx, vy, dt, m, drag_coefficient, Cl, area, g, rho, parachute
            )
            J = jacobian(v_next, dt, m, drag_coefficient, Cl, area, g, rho, parachute)
            delta_v = np.linalg.solve(J, -r)
            v_next += delta_v
            if np.linalg.norm(delta_v) < 1e-6:
                break

        vx_next, vy_next = v_next

        ax_next = (vx_next - vx) / dt
        ay_next = (vy_next - vy) / dt

        x_next = x + vx_next * dt
        y_next = y + vy_next * dt

        accelerations.append((ax_next, ay_next))

        vx, vy = vx_next, vy_next
        x, y = x_next, y_next

        positions.append((x, y))
        velocities.append((vx, vy))

        time += dt

        if y <= 0:
            break

    if mode == "manual" or mode == "fast":
        print(
            Fore.CYAN
            + f"Tempo de reentrada: {time / 60} minutos (Backward Method)."
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

        print("\n")

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


# Plots
def plot_trajectory(
    x_forward_km,
    y_forward_km,
    deploy_position_forward,
    x_backward_km,
    y_backward_km,
    deploy_position_backward,
    v0,
    alpha,
):
    """
    Plot the trajectory

    Parameters
    ----------
    x_forward_km : list
        The horizontal positions of the forward method
    y_forward_km : list
        The vertical positions of the forward method
    deploy_position_forward : tuple
        The deploy position of the parachute of the forward method
    x_backward_km : list
        The horizontal positions of the backward method
    y_backward_km : list
        The vertical positions of the backward method
    deploy_position_backward : tuple
        The deploy position of the parachute of the backward method
    v0 : float
        The initial velocity
    alpha : float
        The downward angle with the horizon
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(x_forward_km, y_forward_km, label="Forward Method")
    if deploy_position_forward:
        ax.plot(
            deploy_position_forward[0] / 1000,
            deploy_position_forward[1] / 1000,
            "ro",
            label="Posição de abertura do paraquedas (Forward)",
        )

    ax.plot(x_backward_km, y_backward_km, label="Backward Method")
    if deploy_position_backward:
        ax.plot(
            deploy_position_backward[0] / 1000,
            deploy_position_backward[1] / 1000,
            "go",
            label="Posição de abertura do paraquedas (Backward)",
        )

    ax.set_xlabel("Distância (km)")
    ax.set_ylabel("Altitude (km)")
    ax.legend()
    ax.set_title(f"Trajetória de reentrada\nv0 = {v0} m/s | alpha = {alpha} graus")

    plt.tight_layout()
    plt.show()


def plot_velocities(
    vx_forward,
    vy_forward,
    time_forward,
    vx_backward,
    vy_backward,
    time_backward,
    v0,
    alpha,
):
    """
    Plot the velocities

    Parameters
    ----------
    vx_forward : list
        The horizontal velocities of the forward method
    vy_forward : list
        The vertical velocities of the forward method
    time_forward : float
        The time of the forward simulation
    vx_backward : list
        The horizontal velocities of the backward method
    vy_backward : list
        The vertical velocities of the backward method
    time_backward : float
        The time of the backward simulation
    v0 : float
        The initial velocity
    alpha : float
        The downward angle with the horizon
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    n_forward = len(vx_forward)
    time_range_forward = np.linspace(0, time_forward, n_forward)

    ax.plot(time_range_forward, vx_forward, label="Vx Forward Method")
    ax.plot(time_range_forward, vy_forward, label="Vy Forward Method")

    n_backward = len(vx_backward)
    time_range_backward = np.linspace(0, time_backward, n_backward)

    ax.plot(time_range_backward, vx_backward, label="Vx Backward Method")
    ax.plot(time_range_backward, vy_backward, label="Vy Backward Method")

    ax.set_xlabel("Tempo (s)")
    ax.set_ylabel("Velocidade (m/s)")
    ax.legend()
    ax.set_title(
        f"Velocidades - Forward and Backward Methods\nv0 = {v0} m/s | alpha = {alpha} graus"
    )

    plt.tight_layout()
    plt.show()


def plot_accelerations(
    ax_forward,
    ay_forward,
    time_forward,
    ax_backward,
    ay_backward,
    time_backward,
    v0,
    alpha,
):
    """
    Plot the accelerations

    Parameters
    ----------
    ax_forward : list
        The horizontal accelerations of the forward method
    ay_forward : list
        The vertical accelerations of the forward method
    time_forward : float
        The time of the forward simulation
    ax_backward : list
        The horizontal accelerations of the backward method
    ay_backward : list
        The vertical accelerations of the backward method
    time_backward : float
        The time of the backward simulation
    v0 : float
        The initial velocity
    alpha : float
        The downward angle with the horizon
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    n_forward = len(ax_forward)
    time_range_forward = np.linspace(0, time_forward, n_forward)

    ax.plot(time_range_forward, ax_forward, label="Ax Forward Method")
    ax.plot(time_range_forward, ay_forward, label="Ay Forward Method")

    n_backward = len(ax_backward)
    time_range_backward = np.linspace(0, time_backward, n_backward)

    ax.plot(time_range_backward, ax_backward, label="Ax Backward Method")
    ax.plot(time_range_backward, ay_backward, label="Ay Backward Method")

    ax.set_xlabel("Tempo (s)")
    ax.set_ylabel("Aceleração (m/s^2)")
    ax.legend()
    ax.set_title(
        f"Acelerações - Forward and Backward Methods\nv0 = {v0} m/s | alpha = {alpha} graus"
    )

    plt.tight_layout()
    plt.show()


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

    x_forward, y_forward = zip(*positions_forward)

    x_forward_km = [distance / 1000 for distance in x_forward]
    y_forward_km = [altitude / 1000 for altitude in y_forward]

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
            h_distance_backward,
            final_velocity_backward,
            g_value_backward,
            results_file_forward_and_backward,
        )

    if mode == "manual" or mode == "fast":
        plot_trajectory(
            x_forward_km,
            y_forward_km,
            deploy_position_forward,
            x_backward_km,
            y_backward_km,
            deploy_position_backward,
            v0,
            alpha,
        )

        plot_velocities(
            [velocity[0] for velocity in velocities_forward],
            [velocity[1] for velocity in velocities_forward],
            time_forward,
            [velocity[0] for velocity in velocities_backward],
            [velocity[1] for velocity in velocities_backward],
            time_backward,
            v0,
            alpha,
        )

        plot_accelerations(
            [acceleration[0] for acceleration in accelerations_forward],
            [acceleration[1] for acceleration in accelerations_forward],
            time_forward,
            [acceleration[0] for acceleration in accelerations_backward],
            [acceleration[1] for acceleration in accelerations_backward],
            time_backward,
            v0,
            alpha,
        )

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
