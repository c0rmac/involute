/**
 * @file test_product_trajectory_rollout_randomised.cpp
 * @brief CBO on a 120-dimensional product space with a randomised physics landscape.
 * Each run uses a unique Haar-sampled target and randomised Rastrigin/cascade parameters.
 */

#include "sampler/isotropic/so_gaussian_sampler.hpp"
#include <random>
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
    std::string tag_desc = "T=" + std::to_string(T) + "_Trajectory_Rollout";
    std::string file_prefix = "trajectory_rollout";

    // --- 1. Randomize the Objective Function Landscape ---
    std::mt19937 gen(std::random_device{}() + run_index);

    std::uniform_real_distribution<double> dist_A(2.0, 8.0);
    std::uniform_real_distribution<double> dist_c(1.0, 4.0);
    std::uniform_real_distribution<double> dist_temp_amp(1.1, 2.0);
    std::uniform_real_distribution<double> dist_cliff(100.0, 250.0);

    const double rand_A = dist_A(gen);
    const double rand_c = dist_c(gen) * 3.14159265358979323846;
    const double rand_temporal_amp = dist_temp_amp(gen);
    const double rand_cliff_threshold = dist_cliff(gen);

    std::cout << "\n--- [" << tag_desc << " | Run " << run_index << "] ---\n";
    std::cout << "Landscape Profile -> Amplitude: " << rand_A
              << " | Freq: " << (rand_c / 3.14159265358979323846) << "π"
              << " | Temp Amp: " << rand_temporal_amp
              << " | Cliff: " << rand_cliff_threshold << "\n";

    // --- 2. Generate Random Global Minimum via Haar Measure ---
    std::vector<Tensor> target_point;
    target_point.reserve(dims.size());

    sampler::SOdGaussianSampler::Config samp_cfg;
    samp_cfg.num_samples = 1;
    samp_cfg.dtype = dtype;

    for (int dim : dims) {
        Tensor dummy_m_hat = math::eye(dim, dtype);
        sampler::SOdGaussianSampler haar_sampler(dummy_m_hat, dim, samp_cfg);

        Tensor haar_batch = haar_sampler.draw_haar_od();
        target_point.push_back(math::slice(haar_batch, 0, 1, 0));
    }

    // --- 3. The Physics Trajectory Objective Function ---
    FuncProductObj objective([=](const std::vector<Tensor> &X) {
        const double penalty_per_missed_step = 1000.0;

        Tensor com_error = Tensor(0.0, dtype);
        Tensor is_fallen = Tensor(0.0, dtype);
        Tensor total_cost = Tensor(0.0, dtype);

        for (int t = 0; t < T; ++t) {
            Tensor step_cost = Tensor(0.0, dtype);
            Tensor step_deviation = Tensor(0.0, dtype);

            for (int j = 0; j < joints_per_step; ++j) {
                int idx = t * joints_per_step + j;

                Tensor diff = math::subtract(X[idx], target_point[idx]);
                Tensor scaled_diff = math::divide(diff, Tensor(4.0, dtype));

                Tensor sq = math::sum(math::square(scaled_diff), {1, 2});

                double num_elements = dims[idx] * dims[idx];
                Tensor cos_term = math::cos(math::multiply(scaled_diff, Tensor(rand_c, dtype)));
                Tensor rastrigin_local = math::multiply(Tensor(rand_A, dtype),
                                         math::subtract(Tensor(num_elements, dtype), math::sum(cos_term, {1, 2})));

                step_deviation = math::add(step_deviation, sq);
                step_cost = math::add(step_cost, math::add(sq, rastrigin_local));
            }

            // --- THE BUTTERFLY EFFECT ---
            Tensor amplified_prev = math::multiply(com_error, Tensor(rand_temporal_amp, dtype));
            com_error = math::add(amplified_prev, step_deviation);

            // --- THE CONTACT CLIFF ---
            Tensor step_fall = math::greater(com_error, Tensor(rand_cliff_threshold, dtype));
            is_fallen = math::where(step_fall, Tensor(1.0, dtype), is_fallen);

            // --- DENSE REWARDS ---
            Tensor cost_for_this_step = math::where(
                math::greater(is_fallen, Tensor(0.5, dtype)),
                Tensor(penalty_per_missed_step, dtype),
                step_cost
            );

            total_cost = math::add(total_cost, cost_for_this_step);
        }

        return total_cost;
    });

    // --- 4. Manifold specifications ---
    std::vector<ManifoldSpec> manifolds;
    manifolds.reserve(dims.size());
    for (int dim : dims) {
        manifolds.push_back(ManifoldSpec::SO(dim, 1.0, delta));
    }

    ProductIsotropicSolverConfig cfg{
        .manifolds         = manifolds,
        .N                 = 10000,
        .dtype             = dtype,
        .convergence       = std::make_shared<MaxStepsCriterion>(max_steps),
        .adapter = std::make_shared<CMAESParameterAdapter>(0.2),
        .debug             = {Debugger::Log, Debugger::History}
    };

    ProductIsotropicSolver solver(cfg);
    ProductCBOResult result = solver.solve(&objective);

    std::cout << "Final energy: " << result.min_energy
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

    const int    k_runs    = 50;
    const double delta     = 2.5;
    const int    max_steps = 400;

    const int T = 5;
    const int joints_per_step = 24;

    std::vector<int> single_step_manifolds = {
        3, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2
    };

    std::vector<int> target_manifolds;
    for (int t = 0; t < T; ++t) {
        target_manifolds.insert(target_manifolds.end(), single_step_manifolds.begin(), single_step_manifolds.end());
    }

    int passed = 0;
    std::cout << "=== Product CBO: 120D Randomised Trajectory Rollout ===\n";

    for (int r = 0; r < k_runs; r++) {
        if (run_scenario(r, target_manifolds, T, joints_per_step, delta, max_steps)) passed++;
    }

    std::cout << "\n=== Summary: " << passed << " / " << k_runs << " passed ===\n";
    return (passed == k_runs) ? 0 : 1;
}
