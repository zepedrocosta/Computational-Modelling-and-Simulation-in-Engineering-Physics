# 1

### Overview
The project involves modeling the reentry of a space module into the Earth's atmosphere, simulating various reentry parameters, and determining the optimal parameters to ensure a safe reentry. The project should include both forward and backward simulation methods, with a comparison of results and code optimization.

### Tasks and Objectives
1. **Model Implementation:**
   - Use the given empirical law for air resistance: 
     
     $$
     F_d = -\frac{1}{2} \rho A C_d v^2
     $$

   - Implement the model considering both pre and post-parachute deployment phases.
   - Air density variation with altitude needs to be accounted for using either an exponential fit or cubic spline.

2. **Simulation:**
   - Implement a forward simulation method to explore reentry parameters (initial velocity \( v_0 \) and angle \( \alpha \)) that meet the defined constraints.
   - Use an implicit method (Backward Euler or Trapezoid) for further simulations and compare results with the forward method.

3. **Constraints for Successful Reentry:**
   - Maximum sustained acceleration: 15g (150 m/s²).
   - Touchdown velocity: ≤ 25 m/s.
   - Horizontal distance: between 2500 km and 4500 km.

4. **Deliverables:**
   - **Code Files:** Python scripts implementing the forward and backward methods.
   - **Report:** A document detailing the project objectives, model, implementation, simulation results, and conclusions.

### Steps to Proceed

#### Step 1: Implement the Air Density Models
You need to implement two methods to estimate air density:
1. **Exponential Fit:**
   - Fit an exponential function to the given data points.
   - Use `scipy.optimize.curve_fit` to obtain the parameters for the exponential function.
   ```python
   import numpy as np
   from scipy.optimize import curve_fit
   import matplotlib.pyplot as plt

   # Given data points
   altitude = np.array([-1000, 0, 1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000, 15000, 20000, 25000, 30000, 40000, 50000, 60000, 70000, 80000])
   density = np.array([1.347, 1.225, 1.112, 1.007, 0.9093, 0.8194, 0.7364, 0.6601, 0.5900, 0.5258, 0.4671, 0.4135, 0.1948, 0.08891, 0.04008, 0.01841, 0.003996, 0.001027, 0.0003097, 0.00008283, 0.00001846])

   # Exponential function
   def exp_func(y, A, B):
       return A * np.exp(B * y)

   # Curve fitting
   popt, _ = curve_fit(exp_func, altitude, density)
   A, B = popt

   # Plot the data points and the exponential fit
   plt.scatter(altitude, density, label='Data points')
   plt.plot(altitude, exp_func(altitude, A, B), color='red', label='Exponential fit')
   plt.xlabel('Altitude (m)')
   plt.ylabel('Air Density (kg/m^3)')
   plt.legend()
   plt.show()
   ```

2. **Cubic Spline:**
   - Use `scipy.interpolate.CubicSpline` to create a cubic spline interpolation of the data.
   ```python
   from scipy.interpolate import CubicSpline

   # Cubic spline interpolation
   cs = CubicSpline(altitude, density, extrapolate=True)

   # Plot the data points and the cubic spline
   plt.scatter(altitude, density, label='Data points')
   plt.plot(altitude, cs(altitude), color='green', label='Cubic Spline')
   plt.xlabel('Altitude (m)')
   plt.ylabel('Air Density (kg/m^3)')
   plt.legend()
   plt.show()
   ```

#### Step 2: Implement Forward Simulation
Create a Python script to perform the forward simulation. You will need to:
- Integrate the equations of motion using the air resistance model and gravitational force.
- Track the velocity, position, and g-forces.
- Ensure the parachute deployment logic is implemented.

#### Step 3: Implement Backward Simulation
Use an implicit method like the Backward Euler or Trapezoid method:
- Implement the backward method to solve the equations of motion.
- Compare the results with the forward simulation.

#### Step 4: Generate Results and Create Report
- Plot the reentry trajectories and compare the results of forward and backward methods.
- Document the methodology, results, and analysis in a report.

### Deliverables Format
- **Code:** Provide clean and well-documented Python scripts for both forward and backward simulations.
- **Report:** Include sections for introduction, model description, implementation details, simulation results, and conclusions.

Would you like help with any specific part of the implementation or further details on any section?

# 2 Usar o ficheiro txt

Great, we can use the data provided in your file for the air density values at different altitudes. Let's proceed with the following steps:

1. **Load the Air Density Data**
2. **Fit an Exponential Function to the Data**
3. **Fit a Cubic Spline to the Data**
4. **Plot and Compare Both Fits**

### Step 1: Load the Air Density Data
We will read the provided data from the file.

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.interpolate import CubicSpline

# Load the data from the file
data = np.loadtxt('/mnt/data/airdensity - students.txt')

altitude = data[:, 0]
density = data[:, 1]

# Plot the raw data
plt.scatter(altitude, density, label='Data points')
plt.xlabel('Altitude (m)')
plt.ylabel('Air Density (kg/m^3)')
plt.title('Air Density vs Altitude')
plt.legend()
plt.show()
```

### Step 2: Fit an Exponential Function to the Data

```python
# Define the exponential function
def exp_func(y, A, B):
    return A * np.exp(B * y)

# Fit the exponential function to the data
popt, _ = curve_fit(exp_func, altitude, density)
A, B = popt

# Plot the data points and the exponential fit
plt.scatter(altitude, density, label='Data points')
plt.plot(altitude, exp_func(altitude, A, B), color='red', label='Exponential fit')
plt.xlabel('Altitude (m)')
plt.ylabel('Air Density (kg/m^3)')
plt.title('Exponential Fit to Air Density Data')
plt.legend()
plt.show()
```

### Step 3: Fit a Cubic Spline to the Data

```python
# Fit a cubic spline to the data
cs = CubicSpline(altitude, density, extrapolate=True)

# Plot the data points and the cubic spline
plt.scatter(altitude, density, label='Data points')
plt.plot(altitude, cs(altitude), color='green', label='Cubic Spline')
plt.xlabel('Altitude (m)')
plt.ylabel('Air Density (kg/m^3)')
plt.title('Cubic Spline Fit to Air Density Data')
plt.legend()
plt.show()
```

### Step 4: Compare Both Fits

```python
# Plot the raw data, exponential fit, and cubic spline fit
plt.scatter(altitude, density, label='Data points')
plt.plot(altitude, exp_func(altitude, A, B), color='red', label='Exponential fit')
plt.plot(altitude, cs(altitude), color='green', label='Cubic Spline')
plt.xlabel('Altitude (m)')
plt.ylabel('Air Density (kg/m^3)')
plt.title('Comparison of Exponential Fit and Cubic Spline to Air Density Data')
plt.legend()
plt.show()
```

### Full Code
Here's the full code in one place for convenience:

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.interpolate import CubicSpline

# Load the data from the file
data = np.loadtxt('/mnt/data/airdensity - students.txt')

altitude = data[:, 0]
density = data[:, 1]

# Plot the raw data
plt.scatter(altitude, density, label='Data points')
plt.xlabel('Altitude (m)')
plt.ylabel('Air Density (kg/m^3)')
plt.title('Air Density vs Altitude')
plt.legend()
plt.show()

# Define the exponential function
def exp_func(y, A, B):
    return A * np.exp(B * y)

# Fit the exponential function to the data
popt, _ = curve_fit(exp_func, altitude, density)
A, B = popt

# Plot the data points and the exponential fit
plt.scatter(altitude, density, label='Data points')
plt.plot(altitude, exp_func(altitude, A, B), color='red', label='Exponential fit')
plt.xlabel('Altitude (m)')
plt.ylabel('Air Density (kg/m^3)')
plt.title('Exponential Fit to Air Density Data')
plt.legend()
plt.show()

# Fit a cubic spline to the data
cs = CubicSpline(altitude, density, extrapolate=True)

# Plot the data points and the cubic spline
plt.scatter(altitude, density, label='Data points')
plt.plot(altitude, cs(altitude), color='green', label='Cubic Spline')
plt.xlabel('Altitude (m)')
plt.ylabel('Air Density (kg/m^3)')
plt.title('Cubic Spline Fit to Air Density Data')
plt.legend()
plt.show()

# Compare both fits
plt.scatter(altitude, density, label='Data points')
plt.plot(altitude, exp_func(altitude, A, B), color='red', label='Exponential fit')
plt.plot(altitude, cs(altitude), color='green', label='Cubic Spline')
plt.xlabel('Altitude (m)')
plt.ylabel('Air Density (kg/m^3)')
plt.title('Comparison of Exponential Fit and Cubic Spline to Air Density Data')
plt.legend()
plt.show()
```

This code will load the air density data, fit both an exponential function and a cubic spline to the data, and plot the results for comparison. Let me know if you need any further assistance or if you have any specific questions about this assignment.