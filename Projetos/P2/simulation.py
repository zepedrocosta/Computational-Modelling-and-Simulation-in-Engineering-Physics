import numpy as np

# Simulation parameters
v0 = 10000  # initial velocity in m/s | 0 and 15 000 m.s−1 for v0
alpha = 7  # 0º and 15º for α

# Space module parameters
m = 12000  # mass of the object in kg
A = 4 * np.pi  # area in m^2
Cd = 1.2  # drag coefficient
Cl = 1.0  # lift coefficient

# Parachutes total model parameters
Ap = 301.00  # parachute area in m^2
Cd_p = 1.0  # parachute drag coefficient
