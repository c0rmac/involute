# Derivation of the ROU Stationary Distribution

**Mirrors De Bortoli et al. (2022) — "Riemannian Score-Based Generative Modelling"**

This document derives the stationary distribution for the Riemannian Ornstein-Uhlenbeck (ROU) process defined by the Stratonovich Stochastic Differential Equation (SDE):

$$\mathrm{d}X_t = \lambda \text{Log}_{X_t}(M) \mathrm{d}t + \delta \circ \mathrm{d}B_t$$

Where:
* $\mathcal{M}$ is a Riemannian manifold.
* $M \in \mathcal{M}$ is a fixed target point (the mean).
* $\lambda > 0$ determines the drift strength, and $\delta > 0$ is the noise scale.
* $\text{Log}_{X_t}(M)$ is the Riemannian logarithmic map.
* $B_t$ is standard Brownian motion on $\mathcal{M}$.

---

### Step 1: Formulating the Drift as a Gradient Flow

**The Goal:** We want to express the drift vector field $\lambda \text{Log}_{X_t}(M)$ as the negative gradient of a scalar potential function $U(x)$.

**The Equation:** In Riemannian geometry, the gradient of the squared geodesic distance $d^2(x, M)$ with respect to $x$ is directly proportional to the logarithmic map pointing from $x$ to $M$:
$$\nabla_x \left( \frac{1}{2} d^2(x, M) \right) = -\text{Log}_x(M)$$

> **Citation:** Boumal, N. (2023). *An Introduction to Optimization on Smooth Manifolds*. Cambridge University Press. (Section 10.3, Proposition 10.32).

**Application:**
We define our potential energy function $U(x)$ as:
$$U(x) = \frac{\lambda}{2} d^2(x, M)$$
Taking the gradient of $U(x)$ yields:
$$\nabla U(x) = -\lambda \text{Log}_x(M)$$

Substituting this into the original SDE yields a standard Riemannian Langevin equation:
$$\mathrm{d}X_t = -\nabla U(X_t) \mathrm{d}t + \delta \circ \mathrm{d}B_t$$

---

### Step 2: The Infinitesimal Generator

**The Goal:** To understand how the probability distribution evolves, we first need the infinitesimal generator $\mathcal{L}$, which describes how the expected value of an observable function $f$ changes over time.

**The Equation:**
For a manifold SDE driven by Stratonovich noise ($\circ \mathrm{d}B_t$), the generator is given by the directional derivative along the drift, plus half the variance times the Laplace-Beltrami operator ($\Delta_{\mathcal{M}}$):
$$\mathcal{L}f = \langle -\nabla U, \nabla f \rangle_{\mathcal{M}} + \frac{\delta^2}{2} \Delta_{\mathcal{M}} f$$

> **Citation:** Hsu, E. P. (2002). *Stochastic Analysis on Manifolds*. American Mathematical Society. (Chapter 3, Theorem 3.1.4 establishes the mapping of Stratonovich SDEs to the Laplace-Beltrami operator).

---

### Step 3: Deriving the Probability Flux and Fokker-Planck Equation

**The Goal:** We must find the equation governing the probability density $p(x, t)$ with respect to the Riemannian volume measure.

**The Equation:**
The time evolution of $p(x, t)$ is given by the Fokker-Planck equation (Kolmogorov Forward Equation), which is defined using the formal adjoint of the generator, $\mathcal{L}^*$:
$$\frac{\partial p}{\partial t} = \mathcal{L}^* p$$

To compute the adjoint $\mathcal{L}^* p$:
1. The adjoint of the directional derivative $\langle -\nabla U, \nabla \cdot \rangle$ is the divergence operator: $\text{div}(p \nabla U)$.
2. The Laplace-Beltrami operator $\Delta_{\mathcal{M}}$ is self-adjoint, meaning its form remains the same. Furthermore, a fundamental identity on manifolds is $\Delta_{\mathcal{M}} p = \text{div}(\nabla p)$.

Substituting these into the adjoint yields:
$$\frac{\partial p}{\partial t} = \text{div}(p \nabla U) + \frac{\delta^2}{2} \text{div}(\nabla p)$$

We can factor out the divergence operator to reveal the **probability flux vector**, $J(x, t)$:
$$\frac{\partial p}{\partial t} = -\text{div} \left( -p \nabla U - \frac{\delta^2}{2} \nabla p \right) = -\text{div}(J)$$
Where the probability flux is defined as:
$$J = -p \nabla U - \frac{\delta^2}{2} \nabla p$$

> **Citation:** Ikeda, N., & Watanabe, S. (1989). *Stochastic Differential Equations and Diffusion Processes*. North-Holland. (Chapter 5, Section 1: Differential operators and diffusion processes).

---

### Step 4: The Detailed Balance Condition

**The Goal:** Solve for the *stationary* distribution $p(x)$, where the density no longer changes over time ($\frac{\partial p}{\partial t} = 0$).

**The Equation:**
If $\frac{\partial p}{\partial t} = 0$, then $\text{div}(J) = 0$. While a divergence-free flux could theoretically mean probability is flowing in closed loops, our system is a purely gradient-driven diffusion. Such systems are physically reversible and satisfy **detailed balance**, meaning the net probability flux is strictly zero everywhere:
$$J = -p \nabla U - \frac{\delta^2}{2} \nabla p = 0$$

> **Citation:** Pavliotis, G. A. (2014). *Stochastic Processes and Applications: Diffusion Processes, the Fokker-Planck and Langevin Equations*. Springer. (Chapter 4, Section 4.4 details the conditions for reversibility and the vanishing of the probability flux in gradient systems).

**Solving for $p(x)$:**
Rearrange the flux equation to isolate the terms involving $p$:
$$\frac{\delta^2}{2} \nabla p = -p \nabla U$$
$$\frac{\nabla p}{p} = -\frac{2}{\delta^2} \nabla U$$

Recognizing that $\frac{\nabla p}{p}$ is the derivative of the natural logarithm:
$$\nabla (\ln p) = \nabla \left( -\frac{2}{\delta^2} U \right)$$

---

### Step 5: Final Solution

Integrating both sides removes the gradients. We introduce a constant of integration $-\ln Z$, where $Z$ ensures the total probability integrates to 1:
$$\ln p(x) = -\frac{2}{\delta^2} U(x) - \ln Z$$

Exponentiating both sides gives the general Gibbs-Boltzmann distribution:
$$p(x) = \frac{1}{Z} \exp \left( -\frac{2}{\delta^2} U(x) \right)$$

Finally, substitute our original geometric potential $U(x) = \frac{\lambda}{2} d^2(x, M)$ back into the equation:
$$p(x) = \frac{1}{Z} \exp \left( -\frac{2}{\delta^2} \frac{\lambda}{2} d^2(x, M) \right)$$
$$p(x) = \frac{1}{Z} \exp \left( -\frac{\lambda}{\delta^2} d^2(x, M) \right)$$

> **Citation:** De Bortoli, V., Mathieu, E., Hutchinson, M., Thornton, J., Teh, Y. W., & Doucet, A. (2022). "Riemannian Score-Based Generative Modelling". *Advances in Neural Information Processing Systems (NeurIPS)*. (Section 3.2, Equation 10 states this final stationary distribution for the forward Riemannian SDE).