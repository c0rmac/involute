/**
 * @file test_stiefel_isotropic_cbo_ackley.cpp
 * @brief Stiefel Isotropic CBO Solver — Ackley objective on V(n,k).
 * Supports multiple (n, k) pairs and k-runs.
 */

#include "involute/solvers/isotropic/stiefel_isotropic_solver.hpp"
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
#include "../../functions/stiefel_objectives.hpp"

using namespace involute;
using namespace involute::core;
using namespace involute::solvers;
using namespace involute::test;

bool run_solver_scenario(int n, int k, int run_index,
                         double delta_param,
                         double /*relative_contraction_rate*/,
                         double step_limit)
{
    const involute::DType target_dtype = involute::DType::Float32;

    FuncObj ackley_cost = make_ackley(n, k, target_dtype);

    const int manifold_dim = n * k - k * (k + 1) / 2;

    StiefelIsotropicSolverConfig config{
        .N          = 50,
        .n          = n,
        .k          = k,
        .convergence = std::make_shared<MaxStepsCriterion>(step_limit),
        .adapter    = std::make_shared<CMAESParameterAdapter>(0.2),
        .lambda     = 1.0,
        .delta      = delta_param,
        .use_matrix_exp = true,
        .dtype      = target_dtype,
        .debug      = std::vector<Debugger>({Debugger::Log, Debugger::History})
    };

    std::cout << "\n--- [n=" << n << ", k=" << k << " | Run " << run_index
              << "] manifold_dim=" << manifold_dim << " ---\n";

    StiefelIsotropicSolver solver(config);
    CBOResult result = solver.solve(&ackley_cost);

    std::cout << "Final Minimum Energy: " << result.min_energy << "\n";

    std::string file_base = "n=" + std::to_string(n) + "_k=" + std::to_string(k)
                          + "_run=" + std::to_string(run_index);
    utils::export_history_to_csv(result.history, "stiefel_results", file_base + ".solver.csv");

    if (std::abs(result.min_energy) > 0.01) {
        std::cerr << "[FAIL] Run " << run_index << " (n=" << n << ", k=" << k << ").\n";
        return false;
    }
    std::cout << "[PASS] Run " << run_index << " (n=" << n << ", k=" << k << ").\n";
    return true;
}

int main(int /*argc*/, char * /*argv*/[]) {
    std::srand(42);

    std::vector<std::pair<int,int>> nk_pairs              = {{6,3}, {10,5}, {20,10}, {180,35}};
    std::vector<double>             deltas                 = {0.9, 0.7, 0.5, 0.2};
    std::vector<double>             relative_contraction_rates = {6.8, 5.0, 1.5, 0.5};
    std::vector<int>                step_limits            = {100, 200, 200, 3500};
    std::vector<int>                k_runs                 = {5, 5, 5, 5};

    sampler::set_num_threads(8);
    math::set_default_device_gpu();

    int total_runs    = std::accumulate(k_runs.begin(), k_runs.end(), 0);
    int successful_runs = 0;

    std::cout << "=== Starting Stiefel Isotropic CBO Ackley Test Suite ===\n";
    std::cout << "Total executions scheduled: " << total_runs << "\n";

    for (std::size_t i = 0; i < nk_pairs.size(); ++i) {
        auto [n, k] = nk_pairs[i];
        for (int r = 0; r < k_runs[i]; ++r)
            if (run_solver_scenario(n, k, r, deltas[i],
                                    relative_contraction_rates[i], step_limits[i]))
                ++successful_runs;
    }

    std::cout << "\n=== Test Suite Summary ===\n";
    std::cout << "Passed: " << successful_runs << " / " << total_runs << "\n";
    if (successful_runs == total_runs) { std::cout << "STATUS: ALL PASSED\n"; return 0; }
    std::cerr << "STATUS: SOME FAILURES DETECTED\n";
    return 1;
}
