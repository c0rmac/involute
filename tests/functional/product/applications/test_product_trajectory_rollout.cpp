/**
 * @file test_product_trajectory_rollout.cpp
 * @brief CBO on a 120-dimensional product space (T=5 steps of SO(3) x SO(2)^23).
 * Simulates a brutal physics trajectory rollout with Temporal Cascading, Contact Cliffs,
 * and Information Sparsity (No Partial Credit).
 */

#include "involute/solvers/isotropic/product_isotropic_solver.hpp"
#include "involute/core/math.hpp"
#include <iostream>
#include <vector>
#include <string>
#include <cmath>
#include <numeric>
#include <algorithm>
#include <involute/solvers/adapters/adam_parameter_adapter.hpp>
#include <involute/solvers/adapters/cma_es_parameter_adapter.hpp>

#include "../../helpers/helper.cpp"

using namespace involute;
using namespace involute::core;
using namespace involute::solvers;

bool run_scenario(int run_index, const std::vector<int>& dims, int T, int joints_per_step, double delta, int max_steps) {
    if (dims.empty()) {
        std::cerr << "[FAIL] No dimensions provided.\n";
        return false;
    }

    const DType dtype = DType::Float32;

    // 1. Build identity matrices
    std::vector<Tensor> identities;
    std::string tag_desc = "T=" + std::to_string(T) + "_Trajectory_Rollout";
    std::string file_prefix = "trajectory_rollout";

    for (size_t i = 0; i < dims.size(); ++i) {
        identities.push_back(math::eye(dims[i], dtype));
    }

    // 2. The Physics Trajectory Objective Function (Decoupled Physics & Cost)
    FuncProductObj objective([=](const std::vector<Tensor> &X) {
        const double A = 5.0;
        const double c = 2.0 * 3.14159265358979323846;

        const double temporal_amplifier = 1.5;
        const double cliff_threshold = 150.0;
        const double penalty_per_missed_step = 1000.0;

        Tensor com_error = Tensor(0.0, dtype);
        Tensor is_fallen = Tensor(0.0, dtype);
        Tensor total_cost = Tensor(0.0, dtype);

        // Loop over the time horizon T
        for (int t = 0; t < T; ++t) {
            Tensor step_cost = Tensor(0.0, dtype);
            Tensor step_deviation = Tensor(0.0, dtype);

            // Calculate errors for all 24 joints at time t
            for (int j = 0; j < joints_per_step; ++j) {
                int idx = t * joints_per_step + j;
                Tensor diff = math::subtract(X[idx], identities[idx]);
                Tensor scaled_diff = math::divide(diff, Tensor(4.0, dtype));

                // 1. Physical kinematic error (Used to check if robot falls)
                Tensor sq = math::sum(math::square(scaled_diff), {1, 2});

                // 2. Optimization landscape "potholes" (Used to confuse the solver)
                double num_elements = dims[idx] * dims[idx];
                Tensor cos_term = math::cos(math::multiply(scaled_diff, Tensor(c, dtype)));
                Tensor rastrigin_local = math::multiply(Tensor(A, dtype),
                                         math::subtract(Tensor(num_elements, dtype), math::sum(cos_term, {1, 2})));

                step_deviation = math::add(step_deviation, sq);
                step_cost = math::add(step_cost, math::add(sq, rastrigin_local));
            }

            // --- THE BUTTERFLY EFFECT (Physics tracking only) ---
            Tensor amplified_prev = math::multiply(com_error, Tensor(temporal_amplifier, dtype));
            com_error = math::add(amplified_prev, step_deviation);

            // --- THE CONTACT CLIFF ---
            Tensor step_fall = math::greater(com_error, Tensor(cliff_threshold, dtype));
            is_fallen = math::where(step_fall, Tensor(1.0, dtype), is_fallen);

            // --- DENSE REWARDS (Cost Shaping) ---
            Tensor cost_for_this_step = math::where(
                math::greater(is_fallen, Tensor(0.5, dtype)),
                Tensor(penalty_per_missed_step, dtype),
                step_cost
            );

            total_cost = math::add(total_cost, cost_for_this_step);
        }

        return total_cost;
    });

    // 3. Manifold specifications
    std::vector<ManifoldSpec> manifolds;
    manifolds.reserve(dims.size());
    for (int dim : dims) {
        manifolds.push_back(ManifoldSpec::SO(dim, 1.0, delta));
    }

    ProductIsotropicSolverConfig cfg{
        .manifolds         = manifolds,
        .N                 = 1000,
        .dtype             = dtype,
        .convergence       = std::make_shared<MaxStepsCriterion>(max_steps),
        .adapter = std::make_shared<CMAESParameterAdapter>(0.2),
        .debug             = {Debugger::Log, Debugger::History}
    };

    std::cout << "\n--- [" << tag_desc << " | Run " << run_index << "] ---\n";

    ProductIsotropicSolver solver(cfg);
    ProductCBOResult result = solver.solve(&objective);

    std::cout << "\nFinal energy: " << result.min_energy
              << " | Steps: " << result.iterations_run << "\n";

    std::string tag = file_prefix + "_run=" + std::to_string(run_index);
    utils::export_history_to_csv(result.history, "product_robotics", tag + ".solver.csv");

    if (result.min_energy >= 1000.0) {
        std::cerr << "[FAIL] Run " << run_index << " ALL PARTICLES CRASHED (Flat Wasteland).\n";
        return false;
    } else if (result.min_energy > 2.0) {
        std::cerr << "[FAIL] Run " << run_index << " survived, but stuck in a pothole. Energy=" << result.min_energy << "\n";
        return false;
    }

    std::cout << "[PASS] Run " << run_index << "\n";
    return true;
}

int main() {
    std::srand(42);
    sampler::set_num_threads(8);
    math::set_default_device_gpu();

    const int    k_runs    = 5;
    const double delta     = 2.5;
    const int    max_steps = 20000;

    const int T = 5; // 5 time steps
    const int joints_per_step = 24;

    std::vector<int> single_step_manifolds = {
        3, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2
    };

    // Construct the 120-dimensional trajectory manifold
    std::vector<int> target_manifolds;
    for (int t = 0; t < T; ++t) {
        target_manifolds.insert(target_manifolds.end(), single_step_manifolds.begin(), single_step_manifolds.end());
    }

    int passed = 0;
    std::cout << "=== Product CBO: 120D Trajectory Rollout (Temporal Cascades & Cliffs) ===\n";

    for (int r = 0; r < k_runs; r++) {
        if (run_scenario(r, target_manifolds, T, joints_per_step, delta, max_steps)) passed++;
    }

    std::cout << "\n=== Summary: " << passed << " / " << k_runs << " passed ===\n";
    return (passed == k_runs) ? 0 : 1;
}
