/**
 * @file test_so_isotropic_cbo_rastrigin.cpp
 * @brief SO(d) Isotropic CBO Solver — Rastrigin objective.
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

bool run_rastrigin_scenario(int d, int run_index, double delta_param,
                             double /*relative_contraction_rate*/,
                             double step_limit, int particle_scale)
{
    const involute::DType target_dtype = involute::DType::Float32;

    Tensor X_target = math::expand_dims(math::eye(d, DType::Float32), {0});
    std::cout << "Target Matrix: " << X_target << std::endl;

    FuncObj rastrigin_cost = make_rastrigin(d, target_dtype);

    auto adapter = std::make_shared<CMAESParameterAdapter>(0.5);

    SOIsotropicSolverConfig config{
        .N          = particle_scale,
        .d          = d,
        .convergence = std::make_shared<MaxStepsCriterion>(step_limit),
        .adapter    = adapter,
        .lambda     = 0.1,
        .delta      = delta_param,
        .dtype      = target_dtype,
        .debug      = std::vector<Debugger>({Debugger::History, Debugger::Log})
    };

    std::cout << "\n--- [RASTRIGIN | d=" << d << " | Run " << run_index << "] ---\n";
    std::cout << "[Config] N=" << config.N << " | Max Steps=" << step_limit << "\n";

    Tensor identity_cost_tensor = rastrigin_cost.evaluate_batch(X_target);
    std::cout << "[Sanity] f(I) = " << math::to_double(identity_cost_tensor) << " (should be ~0)\n";

    SOIsotropicSolver solver(config);
    CBOResult result = solver.solve(&rastrigin_cost);

    std::cout << "Final Minimum Energy: " << result.min_energy << "\n";
    std::cout << "Final Consensus:\n" << result.final_consensus[0] << "\n";

    std::string file_base = "d=" + std::to_string(d) + "_type=Rastrigin_run=" + std::to_string(run_index);
    utils::export_history_to_csv(result.history, "rastrigin_results", file_base + ".solver.csv");

    if (std::abs(result.min_energy) > 0.05) {
        std::cerr << "[FAIL] Run " << run_index << " (d=" << d << ") trapped in local minimum.\n";
        return false;
    }
    std::cout << "[PASS] Run " << run_index << " (d=" << d << ").\n";
    return true;
}

int main() {
    const double diameter = std::sqrt(std::floor(10.0 / 2.0)) * 3.14;

    std::vector<int>    dimensions               = {10, 3};
    std::vector<double> deltas                   = {diameter, diameter};
    std::vector<int>    step_limits              = {280000, 1000};
    std::vector<double> relative_contraction_rates = {0.999, 1.0};
    std::vector<int>    k_runs                   = {550, 50};
    std::vector<int>    particle_scales          = {35550, 150};

    int total_runs = 0;
    for (int kr : k_runs) total_runs += kr;
    int successful_runs = 0;

    sampler::set_num_threads(8);
    math::set_default_device_cpu();

    std::cout << "=== Starting SO(d) Isotropic CBO Rastrigin Test Suite ===\n";
    std::cout << "Total executions scheduled: " << total_runs << "\n";

    for (int i = 0; i < static_cast<int>(dimensions.size()); ++i)
        for (int k = 0; k < k_runs[i]; ++k)
            if (run_rastrigin_scenario(dimensions[i], k, deltas[i],
                                       relative_contraction_rates[i],
                                       step_limits[i], particle_scales[i]))
                ++successful_runs;

    std::cout << "\n=== Test Suite Summary ===\n";
    std::cout << "Passed: " << successful_runs << " / " << total_runs << "\n";
    if (successful_runs == total_runs) { std::cout << "STATUS: ALL PASSED\n"; return 0; }
    std::cerr << "STATUS: SOME FAILURES DETECTED\n";
    return 1;
}
