/**
 * @file test_so_solver.cpp
 * @brief Pure C++ execution entry point for the SO(d) Solver.
 * Evaluates the Brockett Cost Function on the Special Orthogonal Group.
 */

#include "involute/solvers/so_solver.hpp"
#include "involute/core/objective.hpp"
#include "involute/core/math.hpp"
#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>

using namespace involute;
using namespace involute::core;
using namespace involute::solvers;

int main(int argc, char *argv[]) {
    std::cout << "--- Initializing Raw SO(d) Workload ---\n";

    const int d = 50;
    const involute::DType target_dtype = involute::DType::Float32;

    // ------------------------------------------------------------------
    // Objective: Brockett Cost Function f(X) = tr(X^T * A * X * S)
    // ------------------------------------------------------------------

    // 1. Define diagonal matrix A (e.g., descending values: d, d-1, ..., 1)
    std::vector<float> A_data(d * d, 0.0f);
    // 2. Define diagonal matrix S (e.g., ascending values: 1, 2, ..., d)
    std::vector<float> S_data(d * d, 0.0f);

    double theoretical_min = 0.0;
    for (int i = 0; i < d; ++i) {
        float a_val = static_cast<float>(d - i);
        float s_val = static_cast<float>(i + 1);
        A_data[i * d + i] = a_val;
        S_data[i * d + i] = s_val;
        theoretical_min += (a_val * s_val); // Minimum trace when X = Identity
    }

    // Load matrices into device-agnostic Tensors. Shape is [1, d, d] to allow broadcasting over N.
    Tensor A = math::array(A_data, {1, d, d}, target_dtype);
    Tensor S = math::array(S_data, {1, d, d}, target_dtype);

    FuncObj brockett_cost([d, A, S, target_dtype](const Tensor &X) {
        // X is of shape [N, d, d]

        // XT = X^T
        Tensor XT = math::transpose(X, {0, 2, 1});

        // AX = A * X (Matrix multiplication)
        Tensor AX = math::matmul(A, X);

        // XTAX = X^T * A * X
        Tensor XTAX = math::matmul(XT, AX);

        // XTAXS = X^T * A * X * S
        Tensor XTAXS = math::matmul(XTAX, S);

        // Compute the trace of each [d, d] matrix in the batch
        // We do this by extracting the diagonal: XTAXS * I (element-wise), then summing over axes 1 and 2
        Tensor I = math::reshape(math::eye(d, target_dtype), {1, d, d});
        Tensor masked = math::multiply(XTAXS, I);
        Tensor trace = math::sum(masked, {1, 2}); // Output shape: [N]

        return trace;
    });

    //std::cout << "[Config] euclidean_config: " << euclidean_config << ", stable_lambda=" << stable_lambda << "\n";

    SolverConfig config = SOSolver::get_solver_config(core::Aggressive, d, std::make_shared<MaxStepsCriterion>(550000), 0.3, std::vector<Debugger>({Debugger::History, Debugger::Log}), 0.5);

    std::cout << "[Config] N=" << config.N << ", d=" << config.d << "\n";

    SOSolver solver(config);

    std::cout << "[Run] Starting optimization loop on backend...\n";
    CBOResult result = solver.solve(&brockett_cost);

    std::cout << "--- Optimization Complete ---\n";
    std::cout << "Converged: " << (result.converged ? "YES" : "NO") << "\n";
    std::cout << "Final Minimum Energy: " << result.min_energy << "\n";
    std::cout << "Theoretical Minimum Energy: " << theoretical_min << "\n";

    Tensor I_mask = math::eye(d, target_dtype);
    Tensor masked_matrix = math::multiply(result.final_consensus, I_mask);
    Tensor diagonal_only = math::sum(masked_matrix, {1});

    std::cout << "Final Solution Diagonal: " << diagonal_only << "\n";

    // Evaluate convergence (allow some tolerance around the theoretical minimum)
    if (!result.converged || std::abs(result.min_energy - theoretical_min) > 1e-2) {
        std::cerr << "[FAIL] Solver did not reach the expected global minimum.\n";
        return 1;
    }

    std::cout << "[PASS] Solver successfully optimized the Brockett cost function.\n";
    return 0;
}