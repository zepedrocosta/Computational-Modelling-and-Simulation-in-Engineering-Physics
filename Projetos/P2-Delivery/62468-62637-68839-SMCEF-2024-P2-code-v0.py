import math
from colorama import Fore
from matplotlib import pyplot as plt
from scipy.optimize import curve_fit
import numpy as np

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


def calc_v0_components(v0, alpha):
    alphaRad = math.radians(alpha)

    vx = v0 * math.cos(alphaRad)
    vy = -(v0 * math.sin(alphaRad))

    return vx, vy


def exponential_model(x, A, B):
    return A * np.exp(B * x)


def get_exponential_fit(altitude, density):
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


def calculate_accelaration_due_to_gravity(y):
    R_earth_m = 6371 * 1000
    return g0 * (R_earth_m / (R_earth_m + y)) ** 2


def forward_simulation(vx, vy, x, y):
    time = 0
    positions = [(x, y)]
    velocities = [(vx, vy)]
    drag_coefficient = Cd
    area = A
    parachute = False
    deploy_position = None
    for _ in range(int(5000 / dt)):
        v = np.sqrt(vx**2 + vy**2)
        rho = air_density(y)
        g = calculate_accelaration_due_to_gravity(y)

        if y <= 1000 and v <= 100 and not parachute:
            deploy_position = (x, y)
            drag_coefficient += Cdp
            area = Ap
            parachute = True

        Fd = drag_force(v, rho, drag_coefficient, area)
        Fl = lift_force(v, rho, Cl, area)

        ax = Fd * np.cos(math.atan2(vy, vx)) / m
        if parachute:
            ay = ((Fd * np.sin(math.atan2(vy, vx))) / m) - g
        else:
            ay = ((Fd * np.sin(math.atan2(vy, vx)) + Fl) / m) - g

        vx += ax * dt
        vy += ay * dt
        x += vx * dt
        y += vy * dt

        positions.append((x, y))
        velocities.append((vx, vy))

        time += dt

        if y <= 0:
            break

    return positions, velocities, time, deploy_position, parachute


def backward_simulation(vx, vy, x, y):
    time = 0
    positions = [(x, y)]
    velocities = [(vx, vy)]
    drag_coefficient = Cd
    area = A
    parachute = False
    deploy_position = None

    for _ in range(int(5000 / dt)):
        vx_next, vy_next = vx, vy
        x_next, y_next = x, y

        for _ in range(10):
            v_next = np.sqrt(vx_next**2 + vy_next**2)
            rho = air_density(y_next)
            g = calculate_accelaration_due_to_gravity(y)

            if y_next <= 1000 and v_next <= 100 and not parachute:
                deploy_position = (x_next, y_next)
                drag_coefficient = Cd + Cdp
                area = Ap
                parachute = True

            Fd = drag_force(v_next, rho, drag_coefficient, area)
            Fl = lift_force(v_next, rho, Cl, area)

            ax_next = Fd * np.cos(math.atan2(vy_next, vx_next)) / m
            if parachute:
                ay_next = ((Fd * np.sin(math.atan2(vy_next, vx_next))) / m) - g
            else:
                ay_next = ((Fd * np.sin(math.atan2(vy_next, vx_next)) + Fl) / m) - g

            vx_next = vx + ax_next * dt
            vy_next = vy + ay_next * dt
            x_next = x + vx_next * dt
            y_next = y + vy_next * dt

        vx, vy = vx_next, vy_next
        x, y = x_next, y_next

        positions.append((x, y))
        velocities.append((vx, vy))

        time += dt

        if y <= 0:
            break

    return positions, velocities, time, deploy_position, parachute


def calculate_horizontal_distance(x, y):
    theta = 0
    n = len(x)

    for i in range(1, n):
        theta_i = (x[i] - x[i - 1]) / (R_earth + y[i])
        theta += theta_i

    distance = R_earth * theta
    return distance


def calculate_g_value(velocities, time):
    n = len(velocities)
    total_acceleration = 0
    g_value = 0

    for i in range(1, n):
        vx_i, vy_i = velocities[i]
        vx_i_1, vy_i_1 = velocities[i - 1]
        ax_i = (vx_i - vx_i_1) / dt
        ay_i = (vy_i - vy_i_1) / dt
        total_acceleration += math.sqrt(ax_i**2 + ay_i**2)

    g_value = total_acceleration / time / g0

    return total_acceleration, g_value


def calculate_final_velocity(vx, vy):
    final_velocity = math.sqrt(vx**2 + vy**2)
    return final_velocity


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
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

    ax1.plot(x_forward_km, y_forward_km, label="Forward Method")
    if deploy_position_forward:
        ax1.plot(
            deploy_position_forward[0] / 1000,
            deploy_position_forward[1] / 1000,
            "ro",
            label="Deploy Position",
        )
    ax1.set_xlabel("Distância (km)")
    ax1.set_ylabel("Altitude (km)")
    ax1.legend()
    ax1.set_title(
        f"Trajetória de reentrada - Forward Method\nv0 = {v0} m/s | alpha = {alpha} graus"
    )

    ax2.plot(x_backward_km, y_backward_km, label="Backward Method")
    if deploy_position_backward:
        ax2.plot(
            deploy_position_backward[0] / 1000,
            deploy_position_backward[1] / 1000,
            "ro",
            label="Deploy Position",
        )
    ax2.set_xlabel("Distância (km)")
    ax2.set_ylabel("Altitude (km)")
    ax2.legend()
    ax2.set_title(
        f"Trajetória de reentrada - Backward Method\nv0 = {v0} m/s | alpha = {alpha} graus"
    )

    plt.tight_layout()
    plt.show()


def plot_velocities(
    vx_forward, vy_forward, time_forward, vx_backward, vy_backward, time_backward
):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

    n = len(vx_forward)
    time_range_forward = np.linspace(0, time_forward, n)

    ax1.plot(time_range_forward, vx_forward, label="Vx Forward Method")
    ax1.plot(time_range_forward, vy_forward, label="Vy Forward Method")
    ax1.set_xlabel("Tempo (s)")
    ax1.set_ylabel("Velocidade (m/s)")
    ax1.legend()
    ax1.set_title("Velocidades - Forward Method")

    n = len(vx_backward)
    time_range_backward = np.linspace(0, time_backward, n)

    ax2.plot(time_range_backward, vx_backward, label="Vx Backward Method")
    ax2.plot(time_range_backward, vy_backward, label="Vy Backward Method")
    ax2.set_xlabel("Tempo (s)")
    ax2.set_ylabel("Velocidade (m/s)")
    ax2.legend()
    ax2.set_title("Velocidades - Backward Method")

    plt.tight_layout()
    plt.show()


def simulation_handler(v0, alpha):
    success_forward = True
    success_backward = True

    v0x, v0y = calc_v0_components(v0, alpha)

    # Forward simulation
    (
        positions_forward,
        velocities_forward,
        time_forward,
        deploy_position_forward,
        parachute_forward,
    ) = forward_simulation(v0x, v0y, x, y)

    x_forward, y_forward = zip(*positions_forward)

    x_forward_km = [distance / 1000 for distance in x_forward]
    y_forward_km = [altitude / 1000 for altitude in y_forward]

    h_distance_forward = calculate_horizontal_distance(
        x_forward_km,
        y_forward_km,
    )

    final_velocity_forward = calculate_final_velocity(
        velocities_forward[-1][0], velocities_forward[-1][1]
    )

    total_acceleration_forward, g_value_forward = calculate_g_value(
        velocities_forward, time_forward
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
        time_backward,
        deploy_position_backward,
        parachute_backward,
    ) = backward_simulation(v0x, v0y, x, y)

    x_backward, y_backward = zip(*positions_backward)

    x_backward_km = [distance / 1000 for distance in x_backward]
    y_backward_km = [altitude / 1000 for altitude in y_backward]

    h_distance_backward = calculate_horizontal_distance(x_backward_km, y_backward_km)

    final_velocity_backward = calculate_final_velocity(
        velocities_backward[-1][0], velocities_backward[-1][1]
    )
    total_acceleration_backward, g_value_backward = calculate_g_value(
        velocities_backward, time_backward
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
    )

    if not success_forward:
        print("\n" + "Simulação com \x1B[3mforward method\x1B[0m não aceite!!")
    else:
        print("\n" + "Simulação com \x1B[3mforward method\x1B[0m aceite!!")
        print(
            f"Distância horizontal percorrida: {h_distance_forward} km\n"
            f"Velocidade final: {final_velocity_forward} m/s\n"
            f"Valor de g: {g_value_forward}"
        )

    if not success_backward:
        print("\n" + "Simulação com \x1B[3mbackward method\x1B[0m não aceite!!")
    else:
        print("\n" + "Simulação com \x1B[3mbackward method\x1B[0m aceite!!")
        print(
            f"Distância horizontal percorrida: {h_distance_backward} km\n"
            f"Velocidade final: {final_velocity_backward} m/s\n"
            f"Valor de g: {g_value_backward}"
        )


v0 = 8000
alpha = 1
simulation_handler(v0, alpha)
