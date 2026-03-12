# Advanced Setup: Exploiting Consensus Hopping

Involute's Consensus-Based Optimization (CBO) relies on a stochastic hopping mechanism: particles do not just slide down gradients; they evaluate the landscape and "hop" toward a weighted consensus point. For simple, smooth objective functions, a localized swarm with low variance is sufficient. However, if your function contains wide, deceptive basins or disconnected minima, you need to exploit this hopping mechanism to its absolute limits. By increasing the diffusion coefficient (`delta`), you expand the radius of these hops. To guarantee exploration across the entire compact set of the $SO(d)$ manifold, you can push `delta` to 3.14—roughly $\pi$, which represents the maximum distance between conjugate points.

## Example: Maximizing Hopping on the Schwefel Function

To demonstrate how to configure Involute for maximum exploration, we can apply it to the Schwefel function. Because its global minimum is hidden among deep, deceptive local traps, it serves as an ideal template for an objective function that requires a high-variance, highly intelligent swarm. Note how the `relative_learning_rate` also adapts based on the dimension to maintain stability.

```cpp
#include <involute/solvers/so_solver.hpp>
#include <involute/core/objective.hpp>
#include <iostream>
#include <cmath>

using namespace involute;
using namespace involute::core;
using namespace involute::solvers;

int main() {
    int d = 5; 
    
    // 1. Scale particles significantly higher based on the dimension
    // For d=3, scale N by 13. For d=5, scale by 200.
    int particle_scale = (d == 5) ? 200 : 13; 
    int N = particle_scale * d * d; 
    
    // Define the objective function (Shifted & Scaled Schwefel)
    FuncObj schwefel_cost([d](const Tensor &X) {
        const double optimal_val = 420.968746;
        const double A = 418.9829;
        const double n = d * d; 
        
        Tensor I = math::eye(d, DType::Float32);
        Tensor diff = math::subtract(X, I);
        Tensor D_scaled = math::multiply(diff, Tensor(250.0, DType::Float32));
        Tensor Z = math::add(D_scaled, Tensor(optimal_val, DType::Float32));
        
        Tensor abs_Z = math::abs(Z);
        Tensor sqrt_abs_Z = math::sqrt(abs_Z);
        Tensor sin_term = math::sin(sqrt_abs_Z);
        Tensor Z_sin = math::multiply(Z, sin_term);
        
        Tensor sum_Z_sin = math::sum(Z_sin, {1, 2}); 
        Tensor n_A = Tensor(A * n, DType::Float32);
        
        return math::subtract(n_A, sum_Z_sin);
    });

    // 2. Configure the advanced hyperparameters
    // The relative learning rate is tuned based on the complexity of the manifold dimension
    double relative_learning_rate = (d == 5) ? 0.15 : 1.0; 
    double delta_param = 3.14; // Maximum variance covering the compact set SO(d)
    
    SolverConfig config = {
        .N = N,
        .d = d,
        .params = HyperParameters{
            // NOTE: The manual beta value is ignored here. 
            // It is computed adaptively at every step by the AdamParameterAdapter.
            .beta = 1.0,   
            .lambda = 1.0,
            .delta = delta_param 
        },
        .dtype = DType::Float32,
        .convergence = std::make_shared<MaxStepsCriterion>(800),
        
        // 3. Initialize the Adam Adapter with scaled learning and target weight
        .parameter_adapter = std::make_shared<AdamParameterAdapter>(
            // target_max_weight: Generally set between 0.3 and 0.9. 
            // Setting this to 0.3 provides more joint intelligence to the optimiser, 
            // forcing a wider consensus, albeit it may slow down convergence.
            0.3,   
            0.9,   // beta1 (momentum)
            0.999, // beta2 (variance)
            1e-8,  // epsilon
            relative_learning_rate * std::log(N) / (d * d) // Dynamic learning rate for lambda
        ),
        .debug = {Debugger::History, Debugger::Log}
    };

    // Initialize and run the solver
    SOSolver solver(config);
    CBOResult result = solver.solve(&schwefel_cost);

    if (result.converged) {
        std::cout << "\nOptimization Results:\n";
        std::cout << "Target Global Minimum Energy: 0.0\n"; 
        std::cout << "Found Minimum Energy: " << result.min_energy << "\n";
        
        if (std::abs(result.min_energy) < 0.05) {
            std::cout << "Status: Success (Global Minimum Found)\n";
        } else {
            std::cout << "Status: Trapped (Found Local Minimum)\n";
        }
    }

    return 0;
}
```

You can view the results for this setup <a href='./benchmarks.md#schwefel-function'>here</a>.