# Involute

**Involute** is a hardware-accelerated Consensus-Based Optimization (CBO) library for finding global minima of non-convex objective functions constrained to Riemannian manifolds — including $SO(d)$, the Stiefel manifold $V(n,k)$, and arbitrary Cartesian products thereof.

A pre-print describing the theoretical foundations is expected in July 2026.

## 📚 Further Documentation

1. [**Test Results:** Benchmarks and convergence performance on high-dimensional manifolds](docs/benchmarks.md)
2. [**Library Documentation**](docs/lib_docs.md)
3. [**Advanced Setup:** Customizing adapters and backend routing for SYCL/oneMKL](docs/advanced_setup.md)

---

## 💻 Installation

Involute depends on two companion libraries — [isomorphism](https://github.com/c0rmac/isomorphism) (hardware-accelerated tensor DSL) and [riemannian-gaussian-sampler](https://github.com/c0rmac/riemannian-gaussian-sampler) (exact spectral samplers for SO(d) and V(n,k)). The Homebrew formula fetches, taps, and installs both automatically.

### 1. Choose a backend and install

You must explicitly select a backend. No default is assumed.

**Apple MLX** — recommended on Apple Silicon (M1/M2/M3/M4), uses the Metal GPU:

```bash
brew tap c0rmac/homebrew-isomorphism
brew install c0rmac/homebrew-isomorphism/isomorphism --with-mlx
brew install c0rmac/homebrew-involute/involute --with-mlx
```

**LibTorch** — for LibTorch / PyTorch users or non-Apple hardware:

```bash
brew tap c0rmac/homebrew-isomorphism
brew install c0rmac/homebrew-isomorphism/isomorphism --with-torch
brew install c0rmac/homebrew-involute/involute --with-torch
```

isomorphism must be installed first with the matching backend flag. The involute formula then resolves `riemannian-gaussian-sampler` automatically as a declared dependency — no separate tap or install step is needed for it.

### 2. CMake integration

```cmake
find_package(involute REQUIRED)

add_executable(my_solver main.cpp)
target_link_libraries(my_solver PRIVATE involute::involute)
```

---

## ⚡️ Quick Start

All solvers share the same pattern:

1. Define an objective function via `FuncObj` (single manifold) or `FuncProductObj` (product manifold).
2. Construct a solver config using designated initializers.
3. Run `solver.solve(&objective)` and read `result.min_energy`.

### Choosing between CMAES and ADAM

Each manifold type ships two pre-wired solver variants that differ only in how the noise scale evolves across iterations:

| Variant | Adapter | When to use |
|---------|---------|-------------|
| **CMAES** | `CMAESParameterAdapter` | The default choice. Infers the noise schedule from the swarm's covariance structure — no learning-rate tuning required. Works well across a wide range of problem dimensions. |
| **ADAM** | `AdamParameterAdapter` | Use when you want direct control over how aggressively the swarm contracts. Exposes `delta` (initial noise scale) as the primary knob; the Adam-style moment estimates then modulate it step-by-step. |

Both variants accept the same objective function and convergence criterion. Switching is a one-line change to the config type and solver class name, as shown in each example below.

---

## $SO(d)$ — Special Orthogonal Group

Minimize a function $f : SO(d)^N \to \mathbb{R}^N$ whose input is a batch of rotation matrices.

**CMAES** infers the noise schedule from the swarm's covariance structure — no learning rate needs to be specified. **ADAM** drives the noise scale via Adam-style moment estimates; use it when you want direct control over the contraction rate via `delta`.

```cpp
#include <involute/solvers/isotropic/so_isotropic_solver_cmaes.hpp>
#include <involute/solvers/isotropic/so_isotropic_solver_adam.hpp>
#include <involute/core/math.hpp>
#include <iostream>

using namespace involute;
using namespace involute::core;
using namespace involute::solvers;

int main() {
    sampler::set_num_threads(8);
    math::set_default_device_gpu();

    const int d = 5;

    // Rastrigin on SO(d), global minimum f(I) = 0
    FuncObj objective([d](const Tensor& X) {
        const float A      = 10.0f;
        const float two_pi = 2.0f * 3.14159265f;

        Tensor I    = math::eye(d, DType::Float32);
        Tensor diff = math::subtract(X, I);

        return math::sum(
            math::subtract(
                math::add(math::square(diff), Tensor(A, DType::Float32)),
                math::multiply(math::cos(math::multiply(diff, Tensor(two_pi, DType::Float32))),
                               Tensor(A, DType::Float32))),
            {1, 2});
    });

    // --- CMAES: noise schedule derived from swarm covariance, no learning rate needed ---
    SOIsotropicSolverCMAESConfig cmaes_config{
        .N           = 200,
        .d           = d,
        .convergence = std::make_shared<MaxStepsCriterion>(500),
        .delta       = 2.0,
        .debug       = {Debugger::Log}
    };

    SOIsotropicSolverCMAES cmaes_solver(cmaes_config);
    CBOResult cmaes_result = cmaes_solver.solve(&objective);
    std::cout << "[CMAES] Energy: " << cmaes_result.min_energy
              << " | Steps: "       << cmaes_result.iterations_run << "\n";

    // --- ADAM: noise scale driven by Adam moment estimates; tune delta to control contraction ---
    SOIsotropicSolverADAMConfig adam_config{
        .N           = 200,
        .d           = d,
        .convergence = std::make_shared<MaxStepsCriterion>(500),
        .delta       = 2.0,
        .debug       = {Debugger::Log}
    };

    SOIsotropicSolverADAM adam_solver(adam_config);
    CBOResult adam_result = adam_solver.solve(&objective);
    std::cout << "[ADAM]  Energy: " << adam_result.min_energy
              << " | Steps: "       << adam_result.iterations_run << "\n";

    return 0;
}
```

---

## $V(n,k)$ — Stiefel Manifold

Minimize a function $f : V(n,k)^N \to \mathbb{R}^N$ whose input is a batch of semi-orthogonal frames (matrices $X \in \mathbb{R}^{n \times k}$ satisfying $X^\top X = I_k$).

**CMAES** adapts the noise scale per-step without any learning rate. **ADAM** applies moment-based modulation to the noise decay; supply a larger `delta` for a wider initial search radius.

```cpp
#include <involute/solvers/isotropic/stiefel_isotropic_solver_cmaes.hpp>
#include <involute/solvers/isotropic/stiefel_isotropic_solver_adam.hpp>
#include <involute/core/math.hpp>
#include <iostream>

using namespace involute;
using namespace involute::core;
using namespace involute::solvers;

int main() {
    sampler::set_num_threads(8);
    math::set_default_device_gpu();

    const int n = 10, k = 4;

    // Rastrigin on V(n,k), global minimum f(I_{n,k}) = 0
    Tensor I_nk   = math::slice(math::eye(n, DType::Float32), 0, k, 1);
    Tensor two_pi = Tensor(2.0f * 3.14159265f, DType::Float32);
    Tensor A      = Tensor(10.0f, DType::Float32);

    FuncObj objective([I_nk, two_pi, A](const Tensor& X) {
        Tensor diff = math::subtract(X, I_nk);
        return math::sum(
            math::subtract(
                math::add(math::square(diff), A),
                math::multiply(math::cos(math::multiply(diff, two_pi)), A)),
            {1, 2});
    });

    // --- CMAES: noise schedule derived from swarm covariance, no learning rate needed ---
    StiefelIsotropicSolverCMAESConfig cmaes_config{
        .N           = 200,
        .n           = n,
        .k           = k,
        .convergence = std::make_shared<MaxStepsCriterion>(400),
        .delta       = 1.5,
        .debug       = {Debugger::Log}
    };

    StiefelIsotropicSolverCMAES cmaes_solver(cmaes_config);
    CBOResult cmaes_result = cmaes_solver.solve(&objective);
    std::cout << "[CMAES] Energy: " << cmaes_result.min_energy
              << " | Steps: "       << cmaes_result.iterations_run << "\n";

    // --- ADAM: noise scale driven by Adam moment estimates; tune delta to control contraction ---
    StiefelIsotropicSolverADAMConfig adam_config{
        .N           = 200,
        .n           = n,
        .k           = k,
        .convergence = std::make_shared<MaxStepsCriterion>(400),
        .delta       = 1.5,
        .debug       = {Debugger::Log}
    };

    StiefelIsotropicSolverADAM adam_solver(adam_config);
    CBOResult adam_result = adam_solver.solve(&objective);
    std::cout << "[ADAM]  Energy: " << adam_result.min_energy
              << " | Steps: "       << adam_result.iterations_run << "\n";

    return 0;
}
```

---

## Product Manifold — $SO(3) \times SO(2)^{23}$

The product solver optimizes over a Cartesian product of manifold components simultaneously. Each component is specified with `ManifoldSpec::SO(n, lambda, delta)` or `ManifoldSpec::Stiefel(n, k, lambda, delta)`. The objective receives a `std::vector<Tensor>` — one tensor per component — and returns a single energy tensor over the particle batch.

**CMAES** adapts each component's noise scale independently from its local covariance, making it especially well suited to mixed-dimension products where components have very different geometries. **ADAM** applies a shared Adam-style schedule across all components; useful when the landscape is smooth and you want the noise to decay faster.

```cpp
#include <involute/solvers/isotropic/product_isotropic_solver_cmaes.hpp>
#include <involute/solvers/isotropic/product_isotropic_solver_adam.hpp>
#include <involute/core/math.hpp>
#include <iostream>
#include <vector>

using namespace involute;
using namespace involute::core;
using namespace involute::solvers;

int main() {
    sampler::set_num_threads(8);
    math::set_default_device_gpu();

    // SO(3) x SO(2)^23 — 24-joint humanoid configuration space
    std::vector<int> dims = {
        3,                                              // base orientation  SO(3)
        2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,              // left arm + spine  SO(2)^11
        2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2            // right arm + legs  SO(2)^12
    };

    // Build identity targets for each joint
    std::vector<Tensor> targets;
    for (int d : dims)
        targets.push_back(math::eye(d, DType::Float32));

    // Rastrigin on the product space, minimum at all-identity
    FuncProductObj objective([dims, targets](const std::vector<Tensor>& X) {
        const float A      = 10.0f;
        const float two_pi = 2.0f * 3.14159265f;

        Tensor total = Tensor(0.0f, DType::Float32);
        for (size_t i = 0; i < X.size(); ++i) {
            Tensor diff = math::divide(math::subtract(X[i], targets[i]),
                                       Tensor(4.0f, DType::Float32));
            Tensor cos_term  = math::cos(math::multiply(diff, Tensor(two_pi, DType::Float32)));
            Tensor rastrigin = math::multiply(
                Tensor(A, DType::Float32),
                math::subtract(Tensor(static_cast<float>(dims[i] * dims[i]), DType::Float32),
                               math::sum(cos_term, {1, 2})));
            total = math::add(total,
                              math::add(math::sum(math::square(diff), {1, 2}), rastrigin));
        }
        return total;
    });

    std::vector<ManifoldSpec> manifolds;
    for (int d : dims)
        manifolds.push_back(ManifoldSpec::SO(d, 1.0, 2.5));

    // --- CMAES: each component's noise scale adapted independently, no learning rate needed ---
    ProductIsotropicSolverCMAESConfig cmaes_config{
        .manifolds   = manifolds,
        .N           = 500,
        .convergence = std::make_shared<MaxStepsCriterion>(1000),
        .debug       = {Debugger::Log}
    };

    ProductIsotropicSolverCMAES cmaes_solver(cmaes_config);
    ProductCBOResult cmaes_result = cmaes_solver.solve(&objective);
    std::cout << "[CMAES] Energy: " << cmaes_result.min_energy
              << " | Steps: "       << cmaes_result.iterations_run << "\n";

    // --- ADAM: shared Adam moment schedule; tune delta per ManifoldSpec for faster contraction ---
    ProductIsotropicSolverADAMConfig adam_config{
        .manifolds   = manifolds,
        .N           = 500,
        .convergence = std::make_shared<MaxStepsCriterion>(1000),
        .debug       = {Debugger::Log}
    };

    ProductIsotropicSolverADAM adam_solver(adam_config);
    ProductCBOResult adam_result = adam_solver.solve(&objective);
    std::cout << "[ADAM]  Energy: " << adam_result.min_energy
              << " | Steps: "       << adam_result.iterations_run << "\n";

    return 0;
}
```

---

## 🧬 Theoretical Foundation

*A pre-print is expected in July 2026.*

**Consensus-Based Optimization (CBO)** maintains a swarm of $N$ particles $\{X^{(i)}\} \subset \mathcal{M}$ evolving under a coupled system of SDEs. Each particle drifts toward the *consensus point* $M_t$ — the Gibbs-weighted Fréchet mean of the ensemble, $M_t = \text{argmin}_{y} \sum_i \omega^{(i)} d_g^2(y, X^{(i)})$ with $\omega^{(i)} \propto e^{-\beta \mathcal{E}(X^{(i)})}$ — while subject to metric-adjusted Stratonovich noise whose amplitude is proportional to each particle's geodesic distance from consensus:

$$\mathrm{d}X^{(i)}_t = \lambda\,\mathrm{Log}_{X^{(i)}_t}(M_t)\,\mathrm{d}t + \sigma\, d_g\left(X^{(i)}_t,\, M_t\right) \circ \mathrm{d}B^{(i)}_t.$$

The metric-adjusted coefficient $\sigma\, d_g(X^{(i)}_t, M_t)$ ensures that particles far from consensus explore broadly while those near it are stabilized. The library exposes a solver operating directly on this SDE via Euler–Maruyama discretization on the manifold.

These dynamics inspire a more efficient class of algorithms via *consensus freezing* (Fornasier et al.). Holding $M_t$ fixed at $\widehat{M}_k$ throughout a discrete interval $[t_k, t_{k+1})$ and replacing the metric-adjusted noise with a fixed isotropic amplitude $\delta$ yields the autonomous SDE

$$\mathrm{d}X^{(i)}_t = \lambda\,\mathrm{Log}_{X^{(i)}_t}(\widehat{M}_k)\,\mathrm{d}t + \delta \circ \mathrm{d}B^{(i)}_t, \qquad t \in [t_k,\, t_{k+1}).$$

With $\widehat{M}_k$ frozen, the $N$ particles decouple into independent intrinsic Ornstein–Uhlenbeck processes, each with a unique global invariant measure — the Riemannian Gibbs distribution

$$\mu_\infty(\mathrm{d}X) \propto \exp\left(-\frac{\lambda}{\delta^2}\, d_g^2(X,\, \widehat{M}_k)\right)\mathrm{d}\mu_g(X).$$

Rather than integrating this SDE forward — accumulating Euler–Maruyama error and risking constraint violation — each particle is replaced at each step by an exact i.i.d. draw from $\mu_\infty$, with the consensus recomputed from the new ensemble. This is the *consensus hopping* scheme: the swarm does not walk toward $\widehat{M}_k$ but samples geometrically exactly around it, eliminating temporal discretization error.

Exact sampling from $\mu_\infty$ proceeds via a spectral factorization of the homogeneous space $\mathcal{M} = G/H$. Because $\mu_\infty$ depends on $X$ only through $d_g(X, \widehat{M}_k)$, the measure disintegrates over the reductive complement $\mathfrak{m}$ as

$$\mu_\infty(\mathrm{d}X) = p_\infty(\theta)\,\mathrm{d}\theta \otimes \mathrm{Haar}_H(\mathrm{d}h)$$

where $\theta$ are the invariant *shape coordinates* (principal angles in the Weyl chamber) and $h \in H$ is the unobserved orientation fiber. Sampling from the full manifold distribution reduces to: (i) drawing $\theta$ from $p_\infty$ via sequential inverse-CDF transforms on precomputed conditional splines, and (ii) conjugating by a Haar-uniform $h \sim H$ to randomize the fiber. The shape density $p_\infty(\theta)$, governed by the manifold's restricted root system, encodes the curvature-induced eigenvalue repulsion exactly — ensuring the noise explores the full constraint space without bias or cut-locus divergence.

Replacing the metric-adjusted noise with a fixed $\delta$ discards the first-order scaling that the original SDE carried: as the swarm clusters around $M_t$, the coefficient $\sigma d_g(X^{\(i)}\_t, M_t)$ naturally shrinks, providing implicit annealing that the isotropic OU limit does not reproduce. The **CMAES adapter** recovers this through cumulative step-size adaptation (CSA). At each step it computes the tangent displacement of the consensus $\delta_m = \mathrm{\ Log}\_{\\widehat{\ M}\_{\ k-1 }\ }\ (\widehat{\ M}\_{\ k}\)$ and uses it to update a cumulative evolution path $p_\sigma$ in $T_{\\widehat{\ M}\_{\ k}\}\ \mathcal{\ M }\$, parallel-transporting the historical path from the previous tangent space via the Adjoint action. The scalar $\delta$ is then adapted by comparing $\|p_\sigma\|$ against the expected norm $\mathbb{E}[\|\chi_D\|]$ of a $D$-dimensional standard normal: when the consensus moves consistently the path grows long and $\delta$ increases; when the swarm stagnates the path contracts and $\delta$ decays. This reintroduces the first-order dynamics lost in the passage to the isotropic OU limit, without introducing any directional covariance structure.

---

### 🌌 Why "Involute"?

The name refers to a core property of globally symmetric spaces. A manifold is locally symmetric if its Riemann curvature tensor is covariantly constant ($\nabla R = 0$), guaranteed by an **involution** — a smooth, isometric geodesic inversion at every point that is its own inverse. Because the space looks geometrically identical when flipped through any point, curvature does not fluctuate. Involute exploits this symmetry to stabilize the particle swarm, ensuring the Spectral Hopping Scheme remains unconditionally stable across non-convex landscapes.
