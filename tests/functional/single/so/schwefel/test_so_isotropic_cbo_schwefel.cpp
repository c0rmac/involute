/**
 * @file test_so_isotropic_cbo_schwefel.cpp
 * @brief SO(d) Isotropic CBO Solver — Schwefel objective.
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

bool run_schwefel_scenario(int d, int run_index, double delta_param,
                            double /*relative_contraction_rate*/,
                            double step_limit, int particle_scale)
{
    const involute::DType target_dtype = involute::DType::Float32;

    FuncObj schwefel_cost = make_schwefel(d, target_dtype);

    int N = particle_scale * d * d;

    SOIsotropicSolverConfig config{
        .N          = N,
        .d          = d,
        .convergence = std::make_shared<MaxStepsCriterion>(step_limit),
        .adapter    = std::make_shared<CMAESParameterAdapter>(0.2),
        .lambda     = 1.0,
        .delta      = delta_param,
        .dtype      = target_dtype,
        .debug      = std::vector<Debugger>({Debugger::History, Debugger::Log})
    };

    std::cout << "\n--- [SCHWEFEL | d=" << d << " | Run " << run_index << "] ---\n";
    std::cout << "[Config] N=" << config.N << "\n";

    Tensor I_batch = math::expand_dims(math::eye(d, target_dtype), {0});
    std::cout << "[Sanity] f(I) = " << math::to_double(schwefel_cost.evaluate_batch(I_batch))
              << " (should be ~0)\n";

    SOIsotropicSolver solver(config);
    CBOResult result = solver.solve(&schwefel_cost);

    std::cout << "Converged: " << (result.converged ? "YES" : "NO") << "\n";
    std::cout << "Final Minimum Energy: " << result.min_energy << "\n";

    std::string file_base = "d=" + std::to_string(d) + "_run=" + std::to_string(run_index);
    utils::export_history_to_csv(result.history, "schwefel_results", file_base + ".solver.csv");

    if (!result.converged || std::abs(result.min_energy) > 0.05) {
        std::cerr << "[FAIL] Run " << run_index << " (d=" << d << ") trapped in local minimum.\n";
        return false;
    }
    std::cout << "[PASS] Run " << run_index << " (d=" << d << ").\n";
    return true;
}

int main() {
    std::srand(42);

    std::vector<int>    dimensions               = {5, 3};
    std::vector<double> deltas                   = {5.0, 3.14};
    std::vector<int>    step_limits              = {950, 120};
    std::vector<double> relative_contraction_rates = {0.1, 1.0};
    std::vector<int>    k_runs                   = {50, 50};
    std::vector<int>    particle_scales          = {200, 13};

    int total_runs = dimensions.size() * std::accumulate(k_runs.begin(), k_runs.end(), 0);
    int successful_runs = 0;

    std::cout << "=== Starting SO(d) Isotropic CBO Schwefel Test Suite ===\n";
    std::cout << "Total executions scheduled: " << total_runs << "\n";

    for (std::size_t i = 0; i < dimensions.size(); ++i)
        for (int k = 0; k < k_runs[i]; ++k)
            if (run_schwefel_scenario(dimensions[i], k, deltas[i],
                                      relative_contraction_rates[i],
                                      step_limits[i], particle_scales[i]))
                ++successful_runs;

    std::cout << "\n=== Test Suite Summary ===\n";
    std::cout << "Passed: " << successful_runs << " / " << total_runs << "\n";
    if (successful_runs == total_runs) { std::cout << "STATUS: ALL PASSED\n"; return 0; }
    std::cerr << "STATUS: SOME FAILURES DETECTED\n";
    return 1;
}
