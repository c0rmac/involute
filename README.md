# Involute

**Involute** is a state-of-the-art, hardware-accelerated Consensus-Based Optimization (CBO) library designed to find the global minimum of non-convex objective functions constrained to locally symmetric manifolds, **such as $SO(d)$, Stiefel, Grassmannian, and other manifolds**.

Currently optimized exclusively for **macOS Apple Silicon (M-series)** via the Apple MLX backend, Involute allows you to solve complex manifold-constrained problems without being an expert in particle swarm optimization.

## 📚 Further Documentation

1. <del>[**Theoretical Basis:** Understanding CBO and its link with Stochastic Gradient Descent](docs/theory.md)</del>
2. [**Test Results:** Benchmarks and convergence performance on high-dimensional manifolds](docs/benchmarks.md)
3. [**Library Documentation**](docs/lib_docs.md)
4. [**Advanced Setup:** Customizing the Adam adapter and backend routing for SYCL/oneMKL](docs/advanced_setup.md)

## 💻 Installation (macOS Apple Silicon Only)

Involute is heavily optimized for M-series Macs using the Apple MLX backend. You can install it via our custom Homebrew tap:

```bash
#0. Install MLX for macOS compatibility
brew install mlx

# 1. Tap the custom Involute repository
brew tap c0rmac/homebrew-involute

# 2. Install the library
brew install involute
```

## ⚡️ Quick Start

Below is an example of configuring and running the $SO(d)$ solver to find the global optimum of the highly non-convex **Ackley Function**, centered at the Identity matrix:

```cpp
#include <involute/solvers/so_solver.hpp>
#include <involute/core/objective.hpp>
#include <iostream>

using namespace involute;
using namespace involute::core;
using namespace involute::solvers;

int main() {
    int d = 3; // Starting with dimension 3
    
    // Define the Ackley objective function centered at the Identity matrix
    FuncObj ackley_cost([d](const Tensor &X) {
        Tensor I = math::eye(d, DType::Float32);
        Tensor diff = math::subtract(X, I);
        
        // Term 1: Exponential of the root mean squared error
        Tensor sq_diff = math::square(diff);
        Tensor mean_sq = math::divide(math::sum(sq_diff, {1, 2}), Tensor(d * d, DType::Float32));
        Tensor term1 = math::multiply(Tensor(-20.0, DType::Float32), 
                                      math::exp(math::multiply(Tensor(-0.2, DType::Float32), math::sqrt(mean_sq))));
        
        // Term 2: Exponential of the mean cosine
        Tensor cos_diff = math::cos(math::multiply(diff, Tensor(2.0 * 3.14159, DType::Float32)));
        Tensor mean_cos = math::divide(math::sum(cos_diff, {1, 2}), Tensor(d * d, DType::Float32));
        Tensor term2 = math::multiply(Tensor(-1.0, DType::Float32), math::exp(mean_cos));
        
        return math::add(math::add(term1, term2), Tensor(20.0 + 2.71828, DType::Float32)); 
    });

    // Generate a solver config with Adam adaptation and debug options
    SolverConfig config = SOSolver::get_solver_config(
        SolverConfigType::Aggressive, // Also supports SolverConfigType::Safe and SolverConfigType::ExtraSafe
        d, 
        std::make_shared<MaxStepsCriterion>(500),
        1.0, // initial_variance
        {Debugger::History, Debugger::Log} // Options: Debugger::History or Debugger::Log
    );

    // Initialize and run the solver
    SOSolver solver(config);
    CBOResult result = solver.solve(&ackley_cost);

    if (result.converged) {
        std::cout << "Global Minimum Energy Found: " << result.min_energy << "\n";
    }

    return 0;
}
```

See the <a href='./docs/lib_docs/math.md'>Math API documentation</a> for more information.

## 🧬 Theoretical Foundation

While the core mathematical framework of Involute is built upon an upcoming, yet-to-be-released paper, we leverage the theoretical insights from [**Gradient is All You Need?** (arXiv:2306.09778v2)](https://arxiv.org/pdf/2306.09778v2.pdf) to bridge the gap between derivative-free optimization and classical gradient methods. This research establishes that Consensus-Based Optimization (CBO) acts as a stochastic relaxation of gradient descent, naturally approximating stochastic gradient flow dynamics under specific parameter scalings.

By exploiting this connection, we gain a direct parallel with established Stochastic Gradient Descent (SGD) methods, which means we do not have to re-invent optimization heuristics from scratch. A primary example is our integrated **Adam Parameter Adapter**, which treats the collective movement of the particles as an implicit gradient. This allows Involute to seamlessly adopt advanced momentum and adaptive step-size logic, providing high-performance global optimization that feels familiar to anyone used to training neural networks.

---

### 🌌 Why the name "Involute"?

The library's name is inspired by the foundational property of **globally symmetric spaces**. In differential geometry, a manifold is locally symmetric if its Riemann curvature tensor is covariantly constant ($\nabla R = 0$).

This property is guaranteed by the existence of an **involution**—a smooth, isometric transformation (a geodesic inversion) at every point that is its own inverse. Because the space looks geometrically identical when "flipped" through any point, the curvature does not fluctuate. Involute exploits this symmetry to stabilize the particle swarm as it explores the manifold, ensuring our Spectral Hopping Scheme remains unconditionally stable even across challenging non-convex landscapes.

## 🗺️ Roadmap

We are actively developing Involute to expand its capabilities and hardware support. Upcoming features include:

* **Expanded Manifold Support:** Implementing the **Stiefel** and **Grassmannian** manifolds to support optimization over orthonormal frames and subspaces.
* **PC / Linux Backend Integration:** Finalizing the **SYCL** and **oneMKL** backend routing within our `math::` DSL to provide seamless hardware acceleration on non-Apple silicon.
* **Alternative Parameter Adapters:** Exploring additional SGD-inspired algorithms (like RMSProp) for dynamic hyperparameter adaptation.
* **Hybrid Solvers:** Investigating hybrid CBO-GD methods that exploit gradient information when available and cheap to compute for even higher efficiency.