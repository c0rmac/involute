/**
 * @file test_so_solver.cpp
 * @brief Pure C++ execution entry point for the SO(d) Solver.
 * Zero external testing frameworks used. Supports multiple dimensions and k-runs.
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
 * @brief Runs a single scenario of the SO(d) solver.
 * * @param d The dimension for the SO(d) problem.
 * @param run_index The current iteration of the k-runs.
 * @param type The solver configuration type.
 * @return true if the solver successfully converged to the global minimum, false otherwise.
 */
bool run_solver_scenario(int d, int run_index, SolverConfigType type, double delta_param, double relative_contraction_rate, double step_limit) {
    const involute::DType target_dtype = involute::DType::Float32;

    // Objective: Ackley Function centered at the Identity matrix
    FuncObj ackley_cost([d, target_dtype](const Tensor &X) {
        // Standard Ackley parameters
        const double a = 20.0;
        const double b = 0.2;
        const double c = 2.0 * 3.14159265358979323846; // 2 * pi
        const double n = d * d; // Total elements per matrix

        // D = X - I
        Tensor I = math::eye(d, target_dtype);
        Tensor diff = math::subtract(X, I);

        // Term 1: -a * exp(-b * sqrt(1/n * sum(D^2)))
        Tensor sq_diff = math::square(diff);
        Tensor sum_sq = math::sum(sq_diff, {1, 2}); // Sum across rows and cols
        Tensor mean_sq = math::divide(sum_sq, Tensor(n, target_dtype));
        Tensor sqrt_mean_sq = math::sqrt(mean_sq);
        Tensor term1_inner = math::multiply(Tensor(-b, target_dtype), sqrt_mean_sq);
        Tensor term1 = math::multiply(Tensor(-a, target_dtype), math::exp(term1_inner));

        // Term 2: -exp(1/n * sum(cos(c * D)))
        Tensor c_diff = math::multiply(diff, Tensor(c, target_dtype));
        Tensor cos_diff = math::cos(c_diff);
        Tensor sum_cos = math::sum(cos_diff, {1, 2});
        Tensor mean_cos = math::divide(sum_cos, Tensor(n, target_dtype));
        Tensor term2 = math::multiply(Tensor(-1.0, target_dtype), math::exp(mean_cos));

        // Final: term1 + term2 + a + exp(1)
        double constant_term = a + std::exp(1.0);
        Tensor res = math::add(term1, term2);
        return math::add(res, Tensor(constant_term, target_dtype));
    });

    std::string type_string;
    if (type == core::Safe) {
        type_string = "Safe";
    } else if (type == core::ExtraSafe) {
        type_string = "ExtraSafe";
    } else if (type == core::Aggressive) {
        type_string = "Aggressive";
    }

    SolverConfig config = SOSolver::get_solver_config(
        type, d, std::make_shared<MaxStepsCriterion>(step_limit), delta_param, std::vector<Debugger>({Debugger::History, Debugger::Log}), relative_contraction_rate
    );

    std::cout << "\n--- [d=" << d << " | Run " << run_index << "] Initializing Workload ---\n";
    std::cout << "[Config] N=" << config.N << ", Type=" << type_string << "\n";

    SOSolver solver(config);
    CBOResult result = solver.solve(&ackley_cost);

    std::cout << "Converged: " << (result.converged ? "YES" : "NO") << "\n";
    std::cout << "Final Minimum Energy: " << result.min_energy << "\n";

    Tensor I_mask = math::eye(d, target_dtype);
    Tensor masked_matrix = math::multiply(result.final_consensus, I_mask);
    Tensor diagonal_only = math::sum(masked_matrix, {1}); // Sum across columns

    // Only print diagonal for smaller dimensions to avoid console spam, or print it regardless
    //if (d <= 10) {
    //    std::cout << "Final Solution Diagonal: " << diagonal_only << "\n";
    //}

    // Export CSV History with run_index to prevent overwriting
    std::string file_base = "d=" + std::to_string(d) + "_type=" + type_string + "_run=" + std::to_string(run_index);
    utils::export_history_to_csv(result.history, "ackley_results", file_base + ".solver.csv");

    if (!result.converged || std::abs(result.min_energy) > 0.05) {
        std::cerr << "[FAIL] Run " << run_index << " (d=" << d << ") did not reach the expected global minimum.\n";
        return false;
    }

    std::cout << "[PASS] Run " << run_index << " (d=" << d << ") successfully found the Identity matrix.\n";
    return true;
}

int main(int argc, char *argv[]) {
    std::srand(42);

    // --- Test Configuration ---
    std::vector<int> dimensions = {/*50,*/ /*20,*/ /*10,*/ 5, 3}; // Add or modify dimensions here

    std::vector<double> deltas = {/*0.25,*/ /*0.5,*/ 0.7, 0.9}; // Aggressive
    //std::vector<double> deltas = {/*0.03,*/ 0.15, 0.3, 0.5, 0.9}; // ExtraSafe

    //std::vector<int> step_limits = {/*30000, */4500, 3000, 1000, 600}; // ExtraSafe
    std::vector<int> step_limits = {/*500,*/ /*300,*/ 200, 600}; // Aggressive

    //std::vector<double> relative_contraction_rates = {/*1.0,*/ 1.0, 0.8, 0.8, 0.8}; // ExtraSafe
    std::vector<double> relative_contraction_rates = {/*2.0,*/ /*1.0,*/ 1.0, 0.8}; // Aggressive

    //std::vector<int> dimensions = {10};
    //std::vector<double> deltas = {0.3};
    //std::vector<int> step_limits = {1500};

    //const int k_runs = 100;                      // Number of times to run each dimension
    std::vector<int> k_runs = { /*50, 50, 100,*/ 100, 100 };
    SolverConfigType solver_type = core::Aggressive;
    // --------------------------

    int total_runs = dimensions.size() * std::accumulate(k_runs.begin(), k_runs.end(), 0);
    int successful_runs = 0;

    std::cout << "=== Starting SO(d) Solver Test Suite ===\n";
    std::cout << "Dimensions to test: " << dimensions.size() << "\n";
    //std::cout << "Runs per dimension: " << k_runs << "\n";
    std::cout << "Total executions scheduled: " << total_runs << "\n";

    int i = 0;
    for (int d : dimensions) {
        for (int k = 0; k < k_runs[i]; ++k) {
            bool success = run_solver_scenario(d, k, solver_type, deltas[i], relative_contraction_rates[i], step_limits[i]); // d=50,20 -> contraction = 1.0; d=10 -> contraction=1.5; d=5 -> contraction=2.5
            if (success) {
                successful_runs++;
            }
        }
        i++;
    }

    std::cout << "\n=== Test Suite Summary ===\n";
    std::cout << "Passed: " << successful_runs << " / " << total_runs << "\n";

    // Return 0 if all tests passed, 1 if any failed (useful for CI/CD)
    if (successful_runs == total_runs) {
        std::cout << "STATUS: ALL PASSED\n";
        return 0;
    } else {
        std::cerr << "STATUS: SOME FAILURES DETECTED\n";
        return 1;
    }
}