# Section 2: Stationary Distributions of the Intrinsic CBO Process

## 2.1. The Ideal Geometric Setting: Locally Symmetric Manifolds
* **Definition of Local Symmetry:** Introduction of manifolds where the Riemann curvature tensor is parallel ($\nabla R = 0$).
* **Conservative Drift Fields:** Proof that on locally symmetric spaces, the Riemannian Logarithm map $\text{Log}_X(M)$ is a pure gradient field of the squared distance $d_g^2(X, M)$.
* **Stationary Solution:** Derivation of the exact, tractable stationary distribution via the Fokker-Planck equation, yielding the generalized exponential form:
  $$p(X) \propto \exp\left(-\frac{\lambda}{\delta^2} d_g^2(M, X)\right)$$

## 2.2. The Exact Intrinsic Distribution on $SO(d)$
* **$SO(d)$ as a Locally Symmetric Space:** Establishing the Special Orthogonal group as a compact Lie group with a bi-invariant metric.
* **Algebraic Representation:** Transitioning from abstract distance to matrix algebra.
* **Spectral Trace Expansion:** Explicit derivation of the squared intrinsic distance as an infinite series of matrix traces:
  $$d_g^2(M, X) = \sum_{k=1}^{\infty} c_k \text{tr}((M^\top X)^k)$$
* **The Infinite Base Distribution:** Presenting the exact stationary state on $SO(d)$ as an infinite-order exponential family.

## 2.3. The Geometry of the Stiefel Manifold $V(n, k)$
* **The Quotient Relation:** Formal definition of the Stiefel manifold as the quotient space $St(n, k) \cong SO(n) / SO(n-k)$.
* **Loss of Symmetry:** Analysis of the O'Neill fundamental tensor to show that while homogeneous, $V(n, k)$ is not locally symmetric ($\nabla R \neq 0$), introducing "curl" into the drift.
* **Fiber Analysis & Marginalization:** Lifting the $SO(n)$ distribution and marginalizing over the $SO(n-k)$ fiber.
* **The Exact Intractable Form:** Presentation of the resulting density, showing the coupling between the observable base and the fiber integral $\mathcal{I}(X)$.

## 2.4. Deriving the Nearest Tractable Approximation on $V(n, k)$
* **Identification of Intractability:** Pinpointing the "Tractability Cliff" where mixed-term fiber integrations for $k \ge 2$ become non-analytic.
* **The First-Order (Linear) Truncation:** Performing "mathematical surgery" by truncating the fiber interaction to the $k=1$ term.
* **The Hypergeometric Solution:** Evaluating the linear fiber integral to yield the ${}_0F_1$ matrix hypergeometric function.
* **The Best Tractable Proxy:** Definition of the final approximating distribution:
  $$q(X) \propto \exp\left( \frac{\lambda}{\delta^2} \sum_{k=1}^{\infty} c_k \text{tr}((M^\top X)^k) \right) \times {}_0F_1\left( \frac{n-k}{2} ; \Omega(M, X) \right)$$
* **Dimension Sensitivity:** Analysis of how the $(n, k)$ gap explicitly dictates the magnitude of the "disagreement" correction.