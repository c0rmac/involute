/**
 * @file test_so_blackbox_field.cpp
 * @brief Tests the SO(d) solver against an empirical black-box field.
 * Explicitly evaluates Underfitting, Aliasing, and Overfitting.
 */

#include "involute/solvers/isotropic/so_isotropic_solver.hpp"
#include "involute/solvers/adapters/adam_parameter_adapter.hpp"
#include "involute/core/objective.hpp"
#include "involute/core/math.hpp"
#include "../../../helpers/helper.cpp"
#include <iostream>
#include <vector>
#include <cmath>
#include <iomanip>
#include <random>
#include <map>

using namespace involute;
using namespace involute::core;
using namespace involute::solvers;

// ============================================================================
// THE BLACK BOX ENVIRONMENT MODEL
// ============================================================================
Tensor evaluate_blackbox_field(const Tensor& P_rotated, float freq_mult, int seed, DType dtype) {
    float f1 = 7.0f * (1.0f + 0.1f * (seed % 13));
    float f2 = 11.0f * (1.0f + 0.1f * (seed % 17));
    float f3 = 13.0f * (1.0f + 0.1f * (seed % 19));

    Tensor px = math::slice(P_rotated, 0, 1, 1);
    Tensor py = math::slice(P_rotated, 1, 2, 1);
    Tensor pz = math::slice(P_rotated, 2, 3, 1);

    Tensor m1 = math::add(math::multiply(px, math::cos(py)), pz);
    Tensor m2 = math::add(math::multiply(py, math::cos(pz)), px);
    Tensor m3 = math::add(math::multiply(pz, math::cos(px)), py);

    Tensor s1 = math::sin(math::multiply(m1, Tensor(f1 * freq_mult, dtype)));
    Tensor s2 = math::sin(math::multiply(m2, Tensor(f2 * freq_mult, dtype)));
    Tensor s3 = math::sin(math::multiply(m3, Tensor(f3 * freq_mult, dtype)));

    Tensor orientation_bias = math::add(pz, Tensor(1.0f, dtype));

    Tensor interaction = math::multiply(s1, math::multiply(s2, s3));
    return math::multiply(interaction, orientation_bias);
}

// ============================================================================
// TEST SCENARIOS & RUNNER
// ============================================================================
struct TestScenario {
    std::string name;
    float freq_mult;
    float noise_std;
    int N;
    int max_steps;
};

void run_blackbox_scenario(const TestScenario& scenario, int k) {
    std::random_device rd;
    unsigned int session_seed = rd();

    std::cout << "\n======================================================\n";
    std::cout << ">>> RUNNING SCENARIO: " << scenario.name << " (" << k << " runs, session seed: " << session_seed << ") <<<\n";
    std::cout << "======================================================\n";

    const int d = 3;
    const int num_sensors = 25;
    const involute::DType dtype = involute::DType::Float32;

    for (int run = 0; run < k; ++run) {
        std::random_device rd2;
        unsigned int run_seed = rd2();
        std::mt19937 gen(run_seed);

        std::cout << "\n--- Run " << run + 1 << "/" << k
                  << " (Seed: " << run_seed << ") ---\n";

        std::uniform_real_distribution<float> pos_dist(-1.0f, 1.0f);
        std::vector<float> p_data;
        for (int i = 0; i < 3 * num_sensors; ++i) {
            p_data.push_back(pos_dist(gen));
        }
        Tensor P_body = math::array(p_data, {3, num_sensors}, dtype);

        std::vector<float> R_gt_data = {
            0.0f, 1.0f, 0.0f,
            0.0f, 0.0f, 1.0f,
            1.0f, 0.0f, 0.0f
        };
        Tensor R_gt = math::array(R_gt_data, {3, 3}, dtype);

        Tensor R_gt_batched = math::reshape(R_gt, {1, 3, 3});
        Tensor P_gt_rotated = math::matmul(R_gt_batched, P_body);

        Tensor S_clean = evaluate_blackbox_field(P_gt_rotated, scenario.freq_mult, session_seed, dtype);

        std::vector<float> noise_data(num_sensors, 0.0f);
        if (scenario.noise_std > 0.0f) {
            std::normal_distribution<float> noise_dist_fn(0.0f, scenario.noise_std);
            for (int i = 0; i < num_sensors; ++i) {
                noise_data[i] = noise_dist_fn(gen);
            }
        }
        Tensor target_noise = math::array(noise_data, {1, 1, num_sensors}, dtype);
        Tensor S_target = math::add(S_clean, target_noise);

        Tensor sq_noise = math::square(target_noise);
        double gt_energy = math::to_double(math::sum(sq_noise, {1, 2}));

        FuncObj field_cost([P_body, S_target, scenario, session_seed, dtype](const Tensor &X) {
            Tensor P_rot = math::matmul(X, P_body);

            Tensor S_pred = evaluate_blackbox_field(P_rot, scenario.freq_mult, session_seed, dtype);
            Tensor S_target_batch = math::broadcast_to(S_target, S_pred.shape());
            Tensor min_energy = math::sum(math::square(math::subtract(S_pred, S_target_batch)), {1, 2});

            const int num_probes = 1;
            const float search_radius = 0.1f;

            for (int i = 0; i < num_probes; ++i) {
                Tensor noise = math::random_normal(P_rot.shape(), dtype);
                Tensor jitter = math::multiply(noise, Tensor(search_radius, dtype));
                Tensor P_rot_jit = math::add(P_rot, jitter);
                Tensor S_pred_jit = evaluate_blackbox_field(P_rot_jit, scenario.freq_mult, session_seed, dtype);
                Tensor energy_jit = math::sum(math::square(math::subtract(S_pred_jit, S_target_batch)), {1, 2});
                min_energy = math::minimum(min_energy, energy_jit);
            }

            return min_energy;
        });

        math::set_default_device_gpu();

        SOIsotropicSolverConfig config{
            .N          = scenario.N,
            .d          = d,
            .convergence = std::make_shared<MaxStepsCriterion>(1000),
            .adapter    = std::make_shared<AdamParameterAdapter>(0.5, 0.99, 0.999, 1e-8, 0.005),
            .lambda     = 1.0,
            .delta      = 1.0,
            .dtype      = dtype,
            .debug      = std::vector<Debugger>({Debugger::Log})
        };

        SOIsotropicSolver solver(config);
        CBOResult result = solver.solve(&field_cost);

        std::string file_base = scenario.name + "_run_" + std::to_string(run) + "__scenario_" + std::to_string(session_seed);

        Tensor diff_gt = math::subtract(result.final_consensus[0], R_gt);
        double frob_err = math::to_double(math::sqrt(math::sum(math::square(diff_gt))));
        double angle_err_deg = 2.0 * std::asin(std::max(-1.0, std::min(1.0, frob_err / std::sqrt(8.0)))) * (180.0 / M_PI);
        double energy_gap = result.min_energy - gt_energy;

        std::cout << std::fixed << std::setprecision(5);
        std::cout << "GT Energy (Noise Floor): " << gt_energy << "\n";
        std::cout << "Solver Final Energy:     " << result.min_energy << "\n";
        std::cout << "Energy Gap:              " << energy_gap << "\n";
        std::cout << "Rotation Error:          " << angle_err_deg << " deg\n";

        std::string outcome;
        std::cout << "[DIAGNOSIS] ";
        if (energy_gap > 0.5) {
            outcome = "UNDERFITTING";
            std::cout << "UNDERFITTING (Stuck in a local minimum. Swarm failed to find the deep basin.)\n";
        } else if (energy_gap < -0.1) {
            outcome = "OVERFITTING";
            std::cout << "OVERFITTING (Hallucinated a geometry that explains the random static better than the physical truth.)\n";
        } else {
            if (angle_err_deg < 5.0) {
                outcome = "SUCCESS";
                std::cout << "PERFECT RECOVERY (Found the true geometry without overfitting the noise.)\n";
            } else {
                outcome = "ALIASING";
                std::cout << "ALIASING (Found a completely different rotation that produces the exact same sensor readings.)\n";
            }
        }

        std::map<std::string, utils::MetaValue> metadata;
        metadata["run_index"] = run;
        metadata["run_seed"] = run_seed;
        metadata["session_seed"] = session_seed;
        metadata["scenario_name"] = scenario.name;
        metadata["freq_mult"] = scenario.freq_mult;
        metadata["noise_std"] = scenario.noise_std;
        metadata["particle_count"] = scenario.N;
        metadata["final_energy"] = result.min_energy;
        metadata["ground_truth_energy"] = gt_energy;
        metadata["energy_gap"] = energy_gap;
        metadata["angle_err_deg"] = angle_err_deg;
        metadata["outcome"] = outcome;
    }
}

int main() {
    std::vector<TestScenario> suite = {
        {"rugged_trap__freq_mult_3__noise_std_0.1", 1.0f, 0.1f, 150, 500},
        {"rugged_trap__freq_mult_3__noise_std_0.1", 3.0f, 0.1f, 10000, 500},
        {"rugged_trap__freq_mult_3__noise_std_0.1", 3.0f, 0.1f, 10000, 500},
        {"rugged_trap__freq_mult_3__noise_std_0.1", 3.0f, 0.1f, 10000, 500},
        {"rugged_trap__freq_mult_3__noise_std_0.1", 3.0f, 0.1f, 10000, 500},
        {"rugged_trap__freq_mult_3__noise_std_0.1", 3.0f, 0.1f, 10000, 500},
        {"rugged_trap__freq_mult_3__noise_std_0.1", 3.0f, 0.1f, 10000, 500},
        {"rugged_trap__freq_mult_3__noise_std_0.1", 3.0f, 0.1f, 10000, 500},
        {"rugged_trap__freq_mult_3__noise_std_0.1", 3.0f, 0.1f, 10000, 500},
        {"rugged_trap__freq_mult_3__noise_std_0.1", 3.0f, 0.1f, 10000, 500},
        {"rugged_trap__freq_mult_3__noise_std_0.1", 3.0f, 0.1f, 10000, 500},
        {"rugged_trap__freq_mult_3__noise_std_0.1", 3.0f, 0.1f, 10000, 500},
    };

    for (const auto& test : suite) {
        try {
            run_blackbox_scenario(test, 1);
        } catch (const std::exception& e) {
            std::cout << "ERROR: " << e.what() << std::endl;
        }
    }
    return 0;
}
