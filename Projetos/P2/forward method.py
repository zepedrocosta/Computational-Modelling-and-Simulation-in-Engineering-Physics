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


def print_parameters(v0, alpha):
    print(Fore.CYAN + f"Velocidade inicial: {v0} m/s" + Fore.RESET)
    print(Fore.GREEN + f"Ângulo de entrada: {alpha} graus" + Fore.RESET + "\n")


def calc_v0_components(v0, alpha):
    # Convert angle to radians
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
    altitudeArray, densityArray = get_info_from_file(filename)
    fit_altitude, fit_density = get_exponential_fit(altitudeArray, densityArray)
    return np.interp(altitude, fit_altitude, fit_density)


def drag_force(v, altitude, filename):
    """
    Calculate the drag force

    Parameters
    ----------
    v : float
        The velocity of the object
    altitude : float
        The altitude of the object
    filename : str
        The name of the file to be read

    Returns
    -------
    float
        The drag force acting on the object
    """
    rho = air_density(altitude, filename)
    return -0.5 * Cd * A * rho * v**2


def simulation_without_parachute(vx, vy, x, y, filename):
    for _ in range(int(5000 / dt)):
        v = np.sqrt(vx**2 + vy**2)
        Fd = drag_force(v, y, filename)
        ax = Fd * vx / (m * v)
        ay = -g + (Fd * vy / (m * v))

        vx += ax * dt
        vy += ay * dt
        x += vx * dt
        y += vy * dt

        positions.append((x, y))
        velocities.append((vx, vy))

        if y <= 0:
            break
    # Output results
    print("Final position:", positions[-1])
    print("Final velocity:", velocities[-1])


def calculate_horizontal_distance(x, y):
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

    print(Fore.YELLOW + f"Horizontal distance: {distance} km" + Fore.RESET)

    return distance


def plot_trajectory(x_forward, y_forward, x_backward, y_backward):
    # Plot trajectories
    plt.figure()
    plt.plot(x_forward, y_forward, label="Forward Method")
    # plt.plot(x_backward, y_backward_km, label="Backward Method")
    plt.xlabel("Horizontal Distance (km)")
    plt.ylabel("Altitude (km)")
    plt.legend()
    plt.title("Reentry Trajectory Comparison")
    plt.show()


# Constants
g = 10  # gravitational acceleration (m/s^2)
Cd = 1.2  # drag coefficient
Cl = 1.0  # lift coefficient
A = 4 * np.pi  # cross-sectional area (m^2)
m = 12000  # mass of the module (kg)
dt = 0.1  # time step (s)
x = 0.0  # horizontal position (m)
y = 130000.0  # altitude (m)


filename = "Projetos/P2/airdensity - students.txt"

# Initial conditions
v0 = 15000  # Initial velocity in m/s
alpha = 15  # downward angle with the horizontal in degrees

print_parameters(v0, alpha)

vx, vy = calc_v0_components(v0, alpha)

# Lists to store the results
positions = [(x, y)]
velocities = [(vx, vy)]

simulation_without_parachute(vx, vy, x, y, filename)

x_forward, y_forward = zip(*positions)
# x_backward, y_backward = zip(*positions_backward)

# Convert altitude from meters to kilometers
x_forward_km = [distance / 1000 for distance in x_forward]
y_forward_km = [altitude / 1000 for altitude in y_forward]
# y_backward_km = [altitude / 1000 for altitude in y_backward]

# Calculate the horizontal distance
h_distance = calculate_horizontal_distance(x_forward_km, y_forward_km)

# Plot the trajectory
plot_trajectory(x_forward_km, y_forward_km, None, None)
