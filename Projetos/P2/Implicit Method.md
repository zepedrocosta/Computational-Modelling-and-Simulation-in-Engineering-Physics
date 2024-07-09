# Implicit Methods
So far we have seen explicit methods, where the slope is computed from the current state of the system.

Even when we perform several intermediate calculations as in the case of the Runge-Kuta methods of 2nd order or higher. For the Euler method the explicit, or forward method, is

$$u^{k+1} = u^k + f\left(u^k, t^k\right) \Delta t$$

The explicit methods are easy to implement and probably to understand too, but many times have unstable outputs for higher $\Delta t$ values. What is higher $\Delta t$ values is problem and model dependent.

Stiff problems are more prone to unstable results, in some cases a small change in the $\Delta t$ can strongly change the output from stable to unstable. What is a stiff model is somewhat a fuzzy concept.

The implicit methods are more robust in matters of stability. The Euler implicit method, or backward Euler method is:

$$u^{k+1} = u^k + f\left(u^{k+1}, t^{k+1}\right) \Delta t$$

This is, the computation of $u^{k+1}$ depends on it self.

If the model is a linear model it is possible to write:

$$ u^{k+1} = u^k + \left(Au^{k+1} + b\left(t^{k+1}\right)\right) \Delta t$$

$$ \left(I-A\Delta t\right)u^{k+1} = u^k + b\left(t^{k+1}\right)\Delta t$$

With the right side of the equation being constants it is possible to solve using a linear equation solver method.

For non-linear models is more computational demanding.

$$ u^{k+1} = u^k + f\left(u^{k+1}, t^{k+1}\right) \Delta t$$

$$\frac{1}{\Delta t} \left(u^{k+1} - u^k\right) - f\left(u^{k+1}, t^{k+1}\right) = 0$$

And now it is possible to use a root find method considering $u^{k+1}$ as the independent variables vector.

The function we want to find the root, let’s call it the residual function, is:

$$r(v) = \frac{1}{\Delta t} \left(v - u^k\right) - f\left(v, t^{k+1}\right)$$

The $v$ that is the root of $r(v)$ will be the next $u^{k+1}$.

Careful that while applying the Newton method, or other root numerical method, the value of $u^{k}$ is fixed and it is the last point on the Backward Euler method.

For the Newton method each iteration approaching the root is obtained with

$$J(r^l) \Delta \textbf{v} = -r^l$$
$$v^{l+1} = v^{l} + \Delta v$$

If using the Newton-Raphson, or similar methods, we would need to compute the jacobian of this equation.

$$J = \nabla\left(\frac{1}{\Delta t} \left(v - u^k\right) - f\left(v, t^{k+1}\right)\right)$$

$$ J = \frac{1}{\Delta t} I - \nabla f\left(v, t^{k+1}\right)$$