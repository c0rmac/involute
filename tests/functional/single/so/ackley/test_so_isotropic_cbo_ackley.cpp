/**
 * @file test_so_isotropic_cbo_ackley.cpp
 * @brief Pure C++ execution entry point for the SO(d) Isotropic CBO Solver (Ackley objective).
 * Zero external testing frameworks used. Supports multiple dimensions and k-runs.
 */

#include "involute/solvers/isotropic/so_isotropic_solver.hpp"
#include "involute/solvers/adapters/cma_es_parameter_adapter.hpp"
#include "involute/core/objective.hpp"
#include "involute/core/math.hpp"
#include <iostream>
#include <vector>
#include <string>
#include <cmath>
#include <algorithm>
#include <numeric>

#include "../../../helpers/helper.cpp"
#include "../../functions/so_objectives.hpp"

using namespace involute;
using namespace involute::core;
using namespace involute::solvers;
using namespace involute::test;

bool run_solver_scenario(int d, int run_index, double delta_param,
                         double /*relative_contraction_rate*/, double step_limit)
{
    const involute::DType target_dtype = involute::DType::Float32;

    FuncObj ackley_cost = make_ackley(d, target_dtype);

    SOIsotropicSolverConfig config{
        .N          = 50,
        .d          = d,
        .convergence = std::make_shared<MaxStepsCriterion>(step_limit),
        .adapter    = std::make_shared<CMAESParameterAdapter>(0.2, 0.2, 1.0),
        .lambda     = 1.0,
        .delta      = delta_param,
        .use_matrix_exp = true,
        .frechet_mean   = false,
        .dtype      = DType::Float32,
        .debug      = std::vector<Debugger>({Debugger::Log, Debugger::History})
    };

    std::cout << "\n--- [d=" << d << " | Run " << run_index << "] Initializing Workload ---\n";
    std::cout << "[Config] N=" << config.N << "\n";

    SOIsotropicSolver solver(config);
    CBOResult result = solver.solve(&ackley_cost);

    std::cout << "Final Minimum Energy: " << result.min_energy << "\n";

    Tensor I_mask       = math::eye(d, target_dtype);
    Tensor masked_matrix = math::multiply(result.final_consensus[0], I_mask);
    Tensor diagonal_only = math::sum(masked_matrix, {1});
    std::cout << "Final Solution Diagonal: " << diagonal_only << "\n";

    std::string file_base = "d=" + std::to_string(d) + "_run=" + std::to_string(run_index);
    utils::export_history_to_csv(result.history, "ackley_results", file_base + ".solver.csv");

    if (std::abs(result.min_energy) > 0.01) {
        std::cerr << "[FAIL] Run " << run_index << " (d=" << d << ") did not reach the expected global minimum.\n";
        return false;
    }
    std::cout << "[PASS] Run " << run_index << " (d=" << d << ") successfully found the Identity matrix.\n";
    return true;
}

int main(int /*argc*/, char * /*argv*/[]) {
    std::srand(42);

    std::vector<int>    dimensions                  = {50, 20, 10, 5, 3};
    std::vector<double> deltas                      = {0.2, 0.4, 0.6, 0.7, 0.9};
    std::vector<int>    step_limits                 = {1400, 400, 200, 100, 100};
    std::vector<double> relative_contraction_rates  = {0.1,  0.5, 0.7, 1.2, 3.8};
    std::vector<int>    k_runs                      = {200, 200, 200, 200, 200};

    sampler::set_num_threads(8);
    math::set_default_device_gpu();

    int total_runs    = std::accumulate(k_runs.begin(), k_runs.end(), 0);
    int successful_runs = 0;

    std::cout << "=== Starting SO(d) Isotropic CBO Ackley Test Suite ===\n";
    std::cout << "Total executions scheduled: " << total_runs << "\n";

    for (std::size_t i = 0; i < dimensions.size(); ++i)
        for (int k = 0; k < k_runs[i]; ++k)
            if (run_solver_scenario(dimensions[i], k, deltas[i],
                                    relative_contraction_rates[i], step_limits[i]))
                ++successful_runs;

    std::cout << "\n=== Test Suite Summary ===\n";
    std::cout << "Passed: " << successful_runs << " / " << total_runs << "\n";
    if (successful_runs == total_runs) { std::cout << "STATUS: ALL PASSED\n"; return 0; }
    std::cerr << "STATUS: SOME FAILURES DETECTED\n";
    return 1;
}
