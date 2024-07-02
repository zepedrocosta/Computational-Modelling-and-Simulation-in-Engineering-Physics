import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.interpolate import CubicSpline

"""
SMCEF - 2024 - P2 - Reentry of a Space Capsule
@author: Alexandre Tashchuk | 62568
@author: José Pedro Pires Costa | 62637
@author: Carolina dos Santos Saraiva | 68839
"""

# File name
filename = "Projetos/P2/airdensity - students.txt"


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


def plot_data(altitude, density):
    """
    Plot the data

    Parameters
    ----------
    altitude : numpy array
        The altitude values
    density : numpy array
        The density values
    """
    plt.figure(figsize=(10, 8))
    plt.scatter(altitude, density)
    plt.xlabel("Altitude (m)")
    plt.ylabel("Density (kg m$^{-3}$)")
    plt.title("Air Density vs Altitude above sea level")
    plt.show()


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


def get_cubic_spline_fit(altitude, density):
    """
    Get the cubic spline fit

    Parameters
    ----------
    altitude : numpy array
        The altitude values
    density : numpy array
        The density values

    Returns
    -------
    altitude_fine : numpy array
        The altitude values of the fit
    density_fine : numpy array
        The density values of the fit
    """
    cs = CubicSpline(altitude, density, bc_type="natural")

    altitude_fine = np.linspace(-1000, 150000, 500)
    density_fine = cs(altitude_fine)

    return altitude_fine, density_fine


def plot_data_and_exponential_fit(altitude, density, fit_altitude, fit_density):
    """
    Plot the data and the exponential fit

    Parameters
    ----------
    fit_altitude : numpy array
        The altitude values of the fit
    fit_density : numpy array
        The density values of the fit
    """
    plt.figure(figsize=(10, 8))
    plt.scatter(altitude, density, label="data")
    plt.plot(fit_altitude, fit_density, color="red", label="exp fit")
    plt.xlabel("Altitude (m)")
    plt.ylabel("Density (kg m$^{-3}$)")
    plt.title("Air Density vs Altitude above sea level")
    plt.legend()
    plt.show()


def plot_data_and_cubic_spline_fit(altitude, density, altitude_fine, density_fine):
    """
    Plot the data and the cubic spline fit

    Parameters
    ----------
    altitude : numpy array
        The altitude values
    density : numpy array
        The density values
    """
    plt.figure(figsize=(10, 8))
    plt.plot(altitude, density, "bo", label="data")
    plt.plot(altitude_fine, density_fine, "g-", label="cubic spline")
    plt.xlabel("Altitude (m)")
    plt.ylabel("Density (kg m$^{-3}$)")
    plt.legend()
    plt.title("Air density vs altitude above sea level - data and cubic spline")
    plt.show()


def plot_data_expontential_fit_cubic_spline_fit(
    altitude, density, fit_altitude, fit_density, altitude_fine, density_fine
):
    """
    Plot the data, the exponential fit and the cubic spline fit

    Parameters
    ----------
    altitude : numpy array
        The altitude values
    density : numpy array
        The density values
    fit_altitude : numpy array
        The altitude values of the exponential fit
    fit_density : numpy array
        The density values of the exponential fit
    altitude_fine : numpy array
        The altitude values of the cubic spline fit
    density_fine : numpy array
        The density values of the cubic spline fit
    """ 
    plt.figure(figsize=(10, 8))
    plt.plot(altitude, density, "bo", label="data")
    plt.plot(fit_altitude, fit_density, "r-", label="exp fit")
    plt.plot(altitude_fine, density_fine, "g-", label="cubic spline")
    plt.xlabel("Altitude (m)")
    plt.ylabel("Density (kg m$^{-3}$)")
    plt.legend()
    plt.title(
        "Air density vs altitude above sea level - data, exp fit and cubic spline"
    )
    plt.show()


altitude, density = get_info_from_file(filename)
plot_data(altitude, density)

fit_altitude, fit_density = get_exponential_fit(altitude, density)
plot_data_and_exponential_fit(altitude, density, fit_altitude, fit_density)

altitude_fine, density_fine = get_cubic_spline_fit(altitude, density)
plot_data_and_cubic_spline_fit(altitude, density, altitude_fine, density_fine)

plot_data_expontential_fit_cubic_spline_fit(
    altitude, density, fit_altitude, fit_density, altitude_fine, density_fine
)
