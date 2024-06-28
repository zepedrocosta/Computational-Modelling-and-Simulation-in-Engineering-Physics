import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.interpolate import CubicSpline
import math

# Given data from the Engineering Toolbox
altitudes = np.array([-1000, 0, 1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000, 15000, 20000, 25000, 30000, 40000, 50000, 60000, 70000, 80000])
densities = np.array([1.347, 1.225, 1.112, 1.007, 0.9093, 0.8194, 0.7364, 0.6601, 0.59, 0.5258, 0.4671, 0.4135, 0.1948, 0.08891, 0.04008, 0.01841, 0.003996, 0.001027, 0.0003097, 0.00008283, 0.00001846])

# Exponential Fit
def exponential_func(y, A, B):
    return A * np.exp(B * y)

params, covariance = curve_fit(exponential_func, altitudes, densities)
A, B = params

# Cubic Spline
cubic_spline = CubicSpline(altitudes, densities, extrapolate=True)

# Plotting both fits
altitude_range = np.linspace(-1000, 80000, 1000)
density_exp_fit = exponential_func(altitude_range, A, B)
density_spline = cubic_spline(altitude_range)

plt.figure(figsize=(10, 6))
plt.plot(altitudes, densities, 'o', label='Data')
plt.plot(altitude_range, density_exp_fit, '-', label='Exponential Fit')
plt.plot(altitude_range, density_spline, '--', label='Cubic Spline')
plt.xlabel('Altitude (m)')
plt.ylabel('Density (kg/m³)')
plt.legend()
plt.title('Air Density vs Altitude')
plt.grid(True)
plt.show()

# Function to get air density based on method
def get_air_density(altitude, method='exponential'):
    if method == 'exponential':
        return exponential_func(altitude, A, B)
    elif method == 'cubic':
        return cubic_spline(altitude)
    else:
        raise ValueError("Method must be either 'exponential' or 'cubic'")

# Example usage
print(get_air_density(5000, method='exponential'))
print(get_air_density(5000, method='cubic'))


# Constants
REarth = 6371000  # Earth radius in meters
g0 = 9.81  # Gravity acceleration at sea level in m/s^2

# Space module parameters
mass = 12000  # kg
area = 4 * math.pi  # m^2
Cd = 1.2
Cl = 1.0

# Parachute parameters
parachute_area = 301  # m^2
Cd_parachute = 1.0

# Initial conditions
reentry_altitude = 130000  # m
v0_range = np.linspace(0, 15000, 100)  # Initial velocity range in m/s
alpha_range = np.linspace(0, 15, 100)  # Initial angle range in degrees

# Function to calculate drag force
def drag_force(rho, A, Cd, v):
    return 0.5 * rho * A * Cd * v**2

# Function to calculate gravitational acceleration at a given altitude
def gravity_acceleration(altitude):
    return g0 * (REarth / (REarth + altitude))**2

# Simulation function
def forward_simulation(v0, alpha, method='exponential'):
    dt = 0.1  # Time step in seconds
    time = 0
    altitude = reentry_altitude
    velocity = v0
    angle = math.radians(alpha)
    x = 0  # Horizontal distance
    g_max = 0  # Maximum g-force experienced
    parachute_deployed = False
    
    while altitude > 0:
        print(altitude)
        rho = get_air_density(altitude, method)
        g = gravity_acceleration(altitude)
        
        drag = drag_force(rho, area, Cd, velocity)
        lift = drag * Cl / Cd if not parachute_deployed else 0
        drag_parachute = drag_force(rho, parachute_area, Cd_parachute, velocity) if parachute_deployed else 0
        
        total_drag = drag + drag_parachute
        total_acceleration = (total_drag / mass) - g
        g_force = total_acceleration / g0
        
        if g_force > g_max:
            g_max = g_force
        
        if not parachute_deployed and altitude <= 1000 and velocity <= 100:
            parachute_deployed = True
        
        vertical_acceleration = total_acceleration * math.sin(angle)
        horizontal_acceleration = total_acceleration * math.cos(angle)
        
        velocity -= vertical_acceleration * dt
        x += horizontal_acceleration * dt
        
        altitude -= velocity * dt * math.sin(angle)
        time += dt
    
    touchdown_velocity = velocity
    horizontal_distance = x * REarth / (REarth + altitude)  # Adjusting for Earth's curvature
    
    return g_max, touchdown_velocity, horizontal_distance

# Finding valid reentry parameters
valid_params = []

for v0 in v0_range:
    for alpha in alpha_range:
        g_max, touchdown_velocity, horizontal_distance = forward_simulation(v0, alpha, method='exponential')
        if g_max <= 15 and touchdown_velocity <= 25 and 2500 <= horizontal_distance <= 4500:
            valid_params.append((v0, alpha, g_max, touchdown_velocity, horizontal_distance))

# Display valid parameters
for params in valid_params:
    print(f"v0: {params[0]} m/s, alpha: {params[1]} degrees, g_max: {params[2]}, touchdown_velocity: {params[3]} m/s, horizontal_distance: {params[4]} km")

