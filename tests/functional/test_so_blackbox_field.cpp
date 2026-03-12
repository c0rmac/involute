/**
 * @file test_so_blackbox_scenarios.cpp
 * @brief Tests the SO(d) solver against an empirical black-box field.
 * Explicitly evaluates Underfitting, Aliasing, and Overfitting.
 */

#include "involute/solvers/so_solver.hpp"
#include "involute/core/objective.hpp"
#include "involute/core/math.hpp"
#include "helper.cpp"
#include <iostream>
#include <vector>
#include <cmath>
#include <iomanip>
#include <random>

using namespace involute;
using namespace involute::core;
using namespace involute::solvers;

// ============================================================================
// THE BLACK BOX ENVIRONMENT MODEL
// ============================================================================
Tensor evaluate_blackbox_field(const Tensor& P_rotated, float freq_mult, int seed, DType dtype) {
    // 1. Deterministic Chaos: Generate frequencies based on seed
    // Using simple LCG or pre-defined prime tables based on seed
    float f1 = 7.0f * (1.0f + 0.1f * (seed % 13));
    float f2 = 11.0f * (1.0f + 0.1f * (seed % 17));
    float f3 = 13.0f * (1.0f + 0.1f * (seed % 19));

    Tensor px = math::slice(P_rotated, 0, 1, 1);
    Tensor py = math::slice(P_rotated, 1, 2, 1);
    Tensor pz = math::slice(P_rotated, 2, 3, 1);

    // 2. COORDINATE ENTANGLEMENT:
    // Mix the dimensions non-linearly so dE/dx depends on y and z
    // m1 = x * cos(y) + z
    Tensor m1 = math::add(math::multiply(px, math::cos(py)), pz);
    Tensor m2 = math::add(math::multiply(py, math::cos(pz)), px);
    Tensor m3 = math::add(math::multiply(pz, math::cos(px)), py);

    // 3. MULTI-SCALE INTERFERENCE:
    // Combines high-frequency "ruggedness" with low-frequency "basins"
    Tensor s1 = math::sin(math::multiply(m1, Tensor(f1 * freq_mult, dtype)));
    Tensor s2 = math::sin(math::multiply(m2, Tensor(f2 * freq_mult, dtype)));
    Tensor s3 = math::sin(math::multiply(m3, Tensor(f3 * freq_mult, dtype)));

    // 4. ANISOTROPIC ATTENUATION (The "Shadow" Factor):
    // Imagine the signal is weaker when the robot is "upside down"
    // We use the Z-component of the first sensor as a proxy for orientation
    Tensor orientation_bias = math::add(pz, Tensor(1.0f, dtype)); // Range [0, 2]

    // Final Composition:
    // Ruggedness * Attenuation + Nonlinear Coupling
    Tensor interaction = math::multiply(s1, math::multiply(s2, s3));
    return math::multiply(interaction, orientation_bias);
}

// ============================================================================
// TEST SCENARIOS & RUNNER
// ============================================================================
struct TestScenario {
    std::string name;
    float freq_mult;    // >1.0 creates a highly rugged landscape
    float noise_std;    // >0.0 introduces unexplainable static
    int N;              // Particle count
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
        // 1. Generate a unique seed for this specific run's environment
        std::random_device rd;
        unsigned int run_seed = rd();
        std::mt19937 gen(run_seed); // Bind generator to the current seed

        std::cout << "\n--- Run " << run + 1 << "/" << k
                  << " (Seed: " << run_seed << ") ---\n";

        // 2. Generate Robot Body
        std::uniform_real_distribution<float> pos_dist(-1.0f, 1.0f);
        std::vector<float> p_data;
        for (int i = 0; i < 3 * num_sensors; ++i) {
            p_data.push_back(pos_dist(gen));
        }
        Tensor P_body = math::array(p_data, {3, num_sensors}, dtype);

        // 3. Define Ground Truth Orientation (120 deg around [1,1,1] normalized)
        std::vector<float> R_gt_data = {
            0.0f, 1.0f, 0.0f,
            0.0f, 0.0f, 1.0f,
            1.0f, 0.0f, 0.0f
        };
        Tensor R_gt = math::array(R_gt_data, {3, 3}, dtype);

        // 4. Generate Clean Target Signal & Safely Inject Noise
        Tensor R_gt_batched = math::reshape(R_gt, {1, 3, 3});
        Tensor P_gt_rotated = math::matmul(R_gt_batched, P_body);

        // Pass the CURRENT seed into the blackbox so the field shifts per run
        Tensor S_clean = evaluate_blackbox_field(P_gt_rotated, scenario.freq_mult, session_seed, dtype);

        // CRITICAL FIX: Safe noise injection to avoid Undefined Behavior
        std::vector<float> noise_data(num_sensors, 0.0f);
        if (scenario.noise_std > 0.0f) {
            std::normal_distribution<float> noise_dist(0.0f, scenario.noise_std);
            for (int i = 0; i < num_sensors; ++i) {
                noise_data[i] = noise_dist(gen);
            }
        }
        Tensor target_noise = math::array(noise_data, {1, 1, num_sensors}, dtype);
        Tensor S_target = math::add(S_clean, target_noise);

        // 5. Calculate True Ground Truth Energy (The "Noise Floor")
        Tensor sq_noise = math::square(target_noise);
        double gt_energy = math::to_double(math::sum(sq_noise, {1, 2}));

        // 6. Define Objective Function
        // Capture current_seed specifically to ensure the solver sees the same landscape
        /*
        FuncObj field_cost([P_body, S_target, scenario, session_seed, dtype](const Tensor &X) {
            Tensor P_rot = math::matmul(X, P_body);
            Tensor S_pred = evaluate_blackbox_field(P_rot, scenario.freq_mult, session_seed, dtype);
            Tensor S_target_batch = math::broadcast_to(S_target, S_pred.shape());
            Tensor diff = math::subtract(S_pred, S_target_batch);
            Tensor sq_diff = math::square(diff);
            return math::sum(sq_diff, {1, 2});
        });
        */
        FuncObj field_cost([P_body, S_target, scenario, session_seed, dtype](const Tensor &X) {
            Tensor P_rot = math::matmul(X, P_body);

            // 1. Initial Evaluation (The "Center" Point)
            Tensor S_pred = evaluate_blackbox_field(P_rot, scenario.freq_mult, session_seed, dtype);
            Tensor S_target_batch = math::broadcast_to(S_target, S_pred.shape());
            Tensor min_energy = math::sum(math::square(math::subtract(S_pred, S_target_batch)), {1, 2});

            // 2. Probing the Neighborhood (Widening the Needle)
            const int num_probes = 30;
            const float search_radius = 0.1f; // This is your sigma (standard deviation)

            for (int i = 0; i < num_probes; ++i) {
                // Generate standard normal noise [0, 1]
                Tensor noise = math::random_normal(P_rot.shape(), dtype);

                // Scale noise by search_radius to set the actual reach
                Tensor jitter = math::multiply(noise, Tensor(search_radius, dtype));
                Tensor P_rot_jit = math::add(P_rot, jitter);

                // Evaluate the jittered position
                Tensor S_pred_jit = evaluate_blackbox_field(P_rot_jit, scenario.freq_mult, session_seed, dtype);
                Tensor energy_jit = math::sum(math::square(math::subtract(S_pred_jit, S_target_batch)), {1, 2});

                // Min-Pooling: The particle "feels" the lowest energy in its vicinity
                min_energy = math::minimum(min_energy, energy_jit);
            }

            return min_energy;
        });

        // 7. Setup and Run Solver
        SolverConfig config{
            .N = scenario.N,
            .d = d,
            .params = HyperParameters {
                .beta = 1.0,
                .lambda = 1.0,
                .delta = 3.5
            },
            .dtype = dtype,
            .convergence = std::make_shared<MaxStepsCriterion>(5000),
            .parameter_adapter = std::make_shared<AdamParameterAdapter>(
                        0.3, 0.99, 0.999, 1e-8, 0.001),
            //.parameter_adapter = std::make_shared<SOParameterAdapter>(1, 10, 1.0, 1.01, 0.98, 0.9, 0.7),
            .debug = std::vector<Debugger>({Debugger::Log})
        };

        SOSolver solver(config);
        CBOResult result = solver.solve(&field_cost);

        // ==========================================
        // EXPORT LOGIC (Uniquely marked by run number)
        // ==========================================
        std::string file_base = scenario.name + "_run_" + std::to_string(run) + "__scenario_" + std::to_string(session_seed);

        // Export CSV History
        //utils::export_history_to_csv(result.history, "blackbox_results", file_base + ".solver.csv");

        // 8. Verification & Diagnosis
        Tensor diff_gt = math::subtract(result.final_consensus, R_gt);
        double frob_err = math::to_double(math::sqrt(math::sum(math::square(diff_gt))));
        double angle_err_deg = 2.0 * std::asin(std::max(-1.0, std::min(1.0, frob_err / std::sqrt(8.0)))) * (180.0 / M_PI);
        double energy_gap = result.min_energy - gt_energy;

        std::cout << std::fixed << std::setprecision(5);
        std::cout << "GT Energy (Noise Floor): " << gt_energy << "\n";
        std::cout << "Solver Final Energy:     " << result.min_energy << "\n";
        std::cout << "Energy Gap:              " << energy_gap << "\n";
        std::cout << "Rotation Error:          " << angle_err_deg << " deg\n";

        // Determine string outcome for metadata
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

        // Export JSON Metadata
        std::map<std::string, utils::MetaValue> metadata;
        metadata["run_index"] = run;
        metadata["run_seed"] = run_seed; // Ensure it parses nicely to the variant
        metadata["session_seed"] = session_seed; // Ensure it parses nicely to the variant
        metadata["scenario_name"] = scenario.name;
        metadata["freq_mult"] = scenario.freq_mult;
        metadata["noise_std"] = scenario.noise_std;
        metadata["particle_count"] = scenario.N;
        metadata["final_energy"] = result.min_energy;
        metadata["ground_truth_energy"] = gt_energy;
        metadata["energy_gap"] = energy_gap;
        metadata["angle_err_deg"] = angle_err_deg;
        metadata["outcome"] = outcome;

        //utils::export_meta(metadata, "blackbox_results", file_base + ".meta.json");
    }
}

int main() {
    std::vector<TestScenario> suite = {
        //{"SCENARIO 1: The Ideal World (No Noise, Normal Freq)", 1.0f, 0.0f, 5000, 200},
        // {"SCENARIO 2: The Noisy World (High Noise, Normal Freq)", 1.0f, 0.5f, 50000, 20},
        {"rugged_trap__freq_mult_3__noise_std_0.1", 3.0f, 0.1f, 15000, 500},
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