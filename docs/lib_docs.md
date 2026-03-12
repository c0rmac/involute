# 📚 Documentation Index

Welcome to the **Involute** technical reference. This documentation is organized hierarchically by namespace to reflect the internal structure of the library.

---

### `involute::core`
The foundation of the library, containing the hardware-agnostic math layer and abstract base interfaces.

* [**`involute::core::math`**](lib_docs/math.md)
    * Detailed reference for the **Domain Specific Language (DSL)**.
    * Includes documentation for element-wise arithmetic, batched linear algebra (SVD, QR, Cayley), and hardware routing logic.
* **`involute::core::BaseSolver`**
    * The abstract interface for all Consensus-Based Optimization (CBO) solvers.
* **`involute::core::Tensor`**
    * Documentation for the opaque tensor wrapper that enables cross-backend compatibility (MLX/oneMKL).

---

### `involute::solvers`
Specialized solver implementations for manifold-constrained optimization problems.

* [**`involute::solvers::SOSolver`**](lib_docs/so_solver.md)
    * Technical documentation for the **Special Orthogonal Group** solver.
    * Covers the **Spectral Hopping Scheme**, $SO(3)$ analytic shortcuts, and configuration presets.
* **`involute::solvers::StiefelSolver`** (Coming Soon)
    * Optimization logic for the Stiefel manifold ($V_k(\mathbb{R}^n)$).
* **`involute::solvers::GrassmannianSolver`** (Coming Soon)
    * Optimization logic for the Grassmannian manifold ($Gr(k, n)$).

---