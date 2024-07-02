# P2

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

1. **Exponential Fit: DONE**
   - Fit an exponential function to the given data points.
   - Use `scipy.optimize.curve_fit` to obtain the parameters for the exponential function.

2. **Cubic Spline:**
   - Use `scipy.interpolate.CubicSpline` to create a cubic spline interpolation of the data.

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
