/**
 * @file test_schwefel_so_solver.cpp
 * @brief Pure C++ execution entry point for the SO(d) Solver using the Schwefel function.
 */

#include "involute/solvers/so_solver.hpp"
#include "involute/core/objective.hpp"
#include "involute/core/math.hpp"
#include <iostream>
#include <vector>
#include <string>
#include <cmath>
#include <algorithm>
#include <numeric>

#include "helper.cpp"

using namespace involute;
using namespace involute::core;
using namespace involute::solvers;

/**
 * @brief Runs a single Schwefel scenario of the SO(d) solver.
 * @param d The dimension for the SO(d) problem.
 * @param run_index The current iteration of the k-runs.
 * @param type The solver configuration type.
 * @return true if the solver successfully converged to the global minimum, false otherwise.
 */
bool run_schwefel_scenario(int d, int run_index, double delta_param, double relative_contraction_rate, double step_limit, int particle_scale) {
    const involute::DType target_dtype = involute::DType::Float32;

    // Objective: Shifted & Scaled Schwefel Function centered at the Identity matrix
    FuncObj schwefel_cost([d, target_dtype](const Tensor &X) {
        // Standard Schwefel parameters
        const double optimal_val = 420.968746;
        const double A = 418.9829;
        const double n = d * d; // Total elements per matrix

        // D = X - I  (Elements bounded roughly in [-2, 2])
        Tensor I = math::eye(d, target_dtype);
        Tensor diff = math::subtract(X, I);

        // Scale diff to stretch across the classic Schwefel domain [-500, 500]
        Tensor D_scaled = math::multiply(diff, Tensor(250.0, target_dtype));

        // Shift so that D=0 evaluates at the Schwefel global minimum (420.968746)
        // Add optimal_val to each element of D_scaled
        Tensor Z = math::add(D_scaled, Tensor(optimal_val, target_dtype));

        // term = Z * sin(sqrt(|Z|))
        Tensor abs_Z = math::abs(Z);
        Tensor sqrt_abs_Z = math::sqrt(abs_Z);
        Tensor sin_term = math::sin(sqrt_abs_Z);
        Tensor Z_sin = math::multiply(Z, sin_term);

        // sum(Z * sin(sqrt(|Z|)))
        Tensor sum_Z_sin = math::sum(Z_sin, {1, 2}); // Sum across rows and cols

        // Final: 418.9829 * n - sum_Z_sin
        Tensor n_A = Tensor(A * n, target_dtype);
        Tensor res = math::subtract(n_A, sum_Z_sin);

        return res;
    });

    int N = particle_scale * d * d;
    //int N = 5000;
    auto config = core::SolverConfig{
        .N = N,
        .d = d,
        .params = core::HyperParameters{
            .beta = 1.0,
            .lambda = 1.0,
            .delta = delta_param
        },
        .dtype = target_dtype,
        .convergence = std::make_shared<MaxStepsCriterion>(step_limit),
        //.parameter_adapter=std::make_shared<SOParameterAdapter>(50, learning_rate_scale),
        .parameter_adapter = std::make_shared<AdamParameterAdapter>(
            0.3, 0.99, 0.999, 1e-8, relative_contraction_rate * std::log(N) / (d * d)),
        .debug = std::vector<Debugger>({Debugger::History, Debugger::Log})
    };

    std::cout << "\n--- [SCHWEFEL | d=" << d << " | Run " << run_index << "] Initializing Workload ---\n";
    std::cout << "[Config] N=" << config.N << "\n";

    // --- SANITY CHECK: Evaluate at Identity ---
    Tensor I_matrix = math::eye(d, target_dtype);
    Tensor I_batched = math::expand_dims(I_matrix, {0}); // Make it shape [1, d, d]
    Tensor identity_cost_tensor = schwefel_cost.evaluate_batch(I_batched);
    double identity_energy = math::to_double(identity_cost_tensor);
    std::cout << "[Sanity Check] Energy at Identity (should be ~0): " << identity_energy << "\n";
    // ------------------------------------------

    SOSolver solver(config);
    CBOResult result = solver.solve(&schwefel_cost);

    std::cout << "Converged: " << (result.converged ? "YES" : "NO") << "\n";
    std::cout << "Final Minimum Energy: " << result.min_energy << "\n";

    // Export CSV History with run_index
    std::string file_base = "d=" + std::to_string(d) + "_type=Custom" + "_run=" + std::to_string(run_index);
    utils::export_history_to_csv(result.history, "schwefel_results", file_base + ".solver.csv");

    // Schwefel global minimum evaluates to 0, check against a small threshold
    if (!result.converged || std::abs(result.min_energy) > 0.05) {
        std::cerr << "[FAIL] Run " << run_index << " (d=" << d << ") trapped in a local minimum.\n";
        return false;
    }

    std::cout << "[PASS] Run " << run_index << " (d=" << d << ") successfully bypassed Schwefel traps and found Identity.\n";
    return true;
}

int main() {
    std::srand(42);

    // --- Test Configuration ---
    std::vector<int> dimensions = {5, 3};
    std::vector<double> deltas = {3.14, 3.14}; // Aggressive
    std::vector<int> step_limits = {950, 120}; // Aggressive
    std::vector<double> relative_contraction_rates = {0.1, 1.0}; // Aggressive
    std::vector<int> k_runs = { 50, 50 };
    std::vector<int> particle_scales = {200, 13};
    // --------------------------

    int total_runs = dimensions.size() * std::accumulate(k_runs.begin(), k_runs.end(), 0);
    int successful_runs = 0;

    std::cout << "=== Starting SO(d) Schwefel Test Suite ===\n";
    std::cout << "Dimensions to test: " << dimensions.size() << "\n";
    std::cout << "Total executions scheduled: " << total_runs << "\n";

    int i = 0;
    for (int d : dimensions) {
        for (int k = 0; k < k_runs[i]; ++k) {
            bool success = run_schwefel_scenario(d, k, deltas[i], relative_contraction_rates[i], step_limits[i], particle_scales[i]);
            if (success) {
                successful_runs++;
            }
        }
        i++;
    }

    std::cout << "\n=== Test Suite Summary ===\n";
    std::cout << "Passed: " << successful_runs << " / " << total_runs << "\n";

    if (successful_runs == total_runs) {
        std::cout << "STATUS: ALL PASSED\n";
        return 0;
    } else {
        std::cerr << "STATUS: SOME FAILURES DETECTED\n";
        return 1;
    }
}