/**
 * @file test_product_humanoid_locomotion.cpp
 * @brief CBO on a 144-dimensional product space (T=6 steps of SO(3) x SO(2)^23).
 * Simulates Demonstration-Free Locomotion: No reference trajectory.
 * The swarm must discover a walking gait by balancing Forward Progress, Base Stability, and Control Effort.
 */

#include "involute/solvers/isotropic/product_isotropic_solver.hpp"
#include "involute/core/math.hpp"
#include <iostream>
#include <vector>
#include <string>
#include <cmath>
#include <involute/solvers/adapters/adam_parameter_adapter.hpp>
#include <involute/solvers/adapters/cma_es_parameter_adapter.hpp>

#include "../../helpers/helper.cpp"

using namespace involute;
using namespace involute::core;
using namespace involute::solvers;

bool run_scenario(int run_index, const std::vector<int>& dims, int T, int joints_per_step, double delta, int max_steps) {
    const DType dtype = DType::Float32;
    std::string tag_desc = "T=" + std::to_string(T) + "_DemoFree_Humanoid";
    std::string file_prefix = "demo_free";

    // We only need the identity matrix for the Base SO(3) to keep the head upright.
    // The 23 joints DO NOT have a target identity. They must invent their own configuration.
    Tensor base_identity = math::eye(3, dtype);

    // The Demonstration-Free Objective Function
    FuncProductObj objective([=](const std::vector<Tensor> &X) {

        // Physics Simulation Parameters
        const double temporal_amplifier = 1.6;
        const double cliff_threshold    = 80.0;
        const double crash_penalty      = 2000.0;

        // Reward Weights
        const double weight_stability   = 10.0;  // Keep base upright
        const double weight_effort      = 2.0;   // Minimize joint velocities
        const double weight_progress    = -15.0; // REWARD moving forward (Negative cost)

        Tensor com_error    = Tensor(0.0, dtype);
        Tensor is_fallen    = Tensor(0.0, dtype);
        Tensor total_cost   = Tensor(0.0, dtype);
        Tensor steps_fallen = Tensor(0.0, dtype);

        for (int t = 0; t < T; ++t) {
            Tensor step_stability = Tensor(0.0, dtype);
            Tensor step_effort    = Tensor(0.0, dtype);
            Tensor step_progress  = Tensor(0.0, dtype);

            // 1. BASE STABILITY: The SO(3) base (index 0) must stay upright (close to identity)
            int base_idx = t * joints_per_step;
            Tensor base_diff = math::subtract(X[base_idx], base_identity);
            step_stability = math::sum(math::square(math::divide(base_diff, Tensor(4.0, dtype))), {1, 2});

            // 2. LEGS & ARMS (Indices 1 to 23)
            for (int j = 1; j < joints_per_step; ++j) {
                int idx = t * joints_per_step + j;

                // EFFORT: Penalize velocity (difference between current step and previous step)
                if (t > 0) {
                    int prev_idx = (t - 1) * joints_per_step + j;
                    Tensor vel_diff = math::subtract(X[idx], X[prev_idx]);
                    Tensor sq_vel = math::sum(math::square(math::divide(vel_diff, Tensor(4.0, dtype))), {1, 2});
                    step_effort = math::add(step_effort, sq_vel);
                }

                // PROGRESS (Surrogate): Reward specific continuous rotations.
                Tensor joint_val = math::sum(X[idx], {1, 2});
                step_progress = math::add(step_progress, math::divide(joint_val, Tensor(9.0, dtype)));
            }

            // --- TEMPORAL CASCADE (The Physics Engine) ---
            Tensor instability_factor = math::multiply(step_stability, Tensor(weight_stability, dtype));
            Tensor stride_destabilization = math::multiply(math::abs(step_progress), Tensor(0.5, dtype));

            Tensor amplified_prev = math::multiply(com_error, Tensor(temporal_amplifier, dtype));
            com_error = math::add(amplified_prev, math::add(instability_factor, stride_destabilization));

            // --- THE CONTACT CLIFF ---
            Tensor step_fall = math::greater(com_error, Tensor(cliff_threshold, dtype));
            is_fallen = math::where(step_fall, Tensor(1.0, dtype), is_fallen);
            steps_fallen = math::add(steps_fallen, is_fallen);

            // --- COST AGGREGATION ---
            Tensor weighted_stab     = math::multiply(step_stability, Tensor(weight_stability, dtype));
            Tensor weighted_effort   = math::multiply(step_effort, Tensor(weight_effort, dtype));
            Tensor weighted_progress = math::multiply(step_progress, Tensor(weight_progress, dtype));

            Tensor step_cost = math::add(weighted_stab, weighted_effort);
            step_cost = math::add(step_cost, weighted_progress);

            // Dense Survival Reward
            Tensor survival_cost = math::where(
                math::greater(is_fallen, Tensor(0.5, dtype)),
                Tensor(crash_penalty, dtype),
                step_cost
            );

            total_cost = math::add(total_cost, survival_cost);
        }

        // Add a massive flat penalty based on total time spent dead to guarantee slope
        Tensor scaled_crash = math::multiply(Tensor(crash_penalty, dtype), steps_fallen);
        return math::add(total_cost, scaled_crash);
    });

    std::vector<ManifoldSpec> manifolds;
    for (int dim : dims) { manifolds.push_back(ManifoldSpec::SO(dim, 1.0, delta)); }

    ProductIsotropicSolverConfig cfg{
        .manifolds         = manifolds,
        .N                 = 500,
        .dtype             = dtype,
        .convergence       = std::make_shared<MaxStepsCriterion>(max_steps),
        .adapter = std::make_shared<CMAESParameterAdapter>(0.2),
        .debug             = {Debugger::Log, Debugger::History}
    };

    std::cout << "\n--- [" << tag_desc << " | Run " << run_index << "] ---\n";
    ProductIsotropicSolver solver(cfg);
    ProductCBOResult result = solver.solve(&objective);

    std::cout << "Final energy: " << result.min_energy << " | Steps: " << result.iterations_run << "\n";
    std::string tag = file_prefix + "_run=" + std::to_string(run_index);
    utils::export_history_to_csv(result.history, "product_robotics", tag + ".solver.csv");

    if (result.min_energy > 100.0) {
        std::cerr << "[FAIL] Run " << run_index << " collapsed under its own weight.\n";
        return false;
    }
    std::cout << "[PASS] Run " << run_index << "\n";
    return true;
}

int main() {
    sampler::set_num_threads(8);
    math::set_default_device_gpu();

    const int k_runs = 5;
    const double delta = 3.0;
    const int max_steps = 25000;

    const int T = 6;
    const int joints_per_step = 24;

    std::vector<int> single_step_manifolds = {3, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2};
    std::vector<int> target_manifolds;
    for (int t = 0; t < T; ++t) {
        target_manifolds.insert(target_manifolds.end(), single_step_manifolds.begin(), single_step_manifolds.end());
    }

    int passed = 0;
    std::cout << "=== Product CBO: 144D Demonstration-Free Locomotion ===\n";
    for (int r = 0; r < k_runs; r++) {
        if (run_scenario(r, target_manifolds, T, joints_per_step, delta, max_steps)) passed++;
    }

    std::cout << "\n=== Summary: " << passed << " / " << k_runs << " passed ===\n";
    return (passed == k_runs) ? 0 : 1;
}
