/**
 * @file test_so_robust_hand_eye.cpp
 * @brief Robust Hand-Eye Calibration (AX = XB) test on SO(3).
 * Tests the solver's ability to reject completely invalid rotation
 * measurements (outliers) using a robust estimator on the manifold.
 */

#include "involute/solvers/so_solver.hpp"
#include "involute/core/objective.hpp"
#include "involute/core/math.hpp"
#include <iostream>
#include <vector>
#include <cmath>
#include <iomanip>
#include <random>

using namespace involute;
using namespace involute::core;
using namespace involute::solvers;

// ============================================================================
// HELPER MATH FUNCTIONS (For generating clean SO(3) Ground Truth)
// ============================================================================

std::vector<float> make_rotation(char axis, float angle_deg) {
    float rad = angle_deg * (M_PI / 180.0f);
    float c = std::cos(rad);
    float s = std::sin(rad);

    if (axis == 'X') return {1, 0, 0, 0, c, -s, 0, s, c};
    if (axis == 'Y') return {c, 0, s, 0, 1, 0, -s, 0, c};
    if (axis == 'Z') return {c, -s, 0, s, c, 0, 0, 0, 1};
    return {1, 0, 0, 0, 1, 0, 0, 0, 1};
}

std::vector<float> matmul_3x3(const std::vector<float>& M1, const std::vector<float>& M2) {
    std::vector<float> R(9, 0.0f);
    for(int i = 0; i < 3; ++i) {
        for(int j = 0; j < 3; ++j) {
            for(int k = 0; k < 3; ++k) {
                R[i*3 + j] += M1[i*3 + k] * M2[k*3 + j];
            }
        }
    }
    return R;
}

std::vector<float> transpose_3x3(const std::vector<float>& M) {
    return {M[0], M[3], M[6], M[1], M[4], M[7], M[2], M[5], M[8]};
}

// ============================================================================
// TEST RUNNER
// ============================================================================

struct TestScenario {
    std::string name;
    int num_motions;
    float outlier_ratio;
    int max_steps;
};

void run_nightmare_hand_eye_test() {
    std::cout << "\n======================================================\n";
    std::cout << ">>> RUNNING STRESS TEST: NIGHTMARE SCENARIO <<<\n";
    std::cout << "======================================================\n";

    const int d = 3;
    const involute::DType dtype = involute::DType::Float32;
    std::mt19937 gen; // New seed for the nightmare

    // 1. Ground Truth
    std::vector<float> X_gt_data = make_rotation('Y', 90.0f);
    Tensor X_gt = math::array(X_gt_data, {3, 3}, dtype);
    std::vector<float> X_gt_T_data = transpose_3x3(X_gt_data);

    // 100 motions, but 80 of them are pure garbage.
    int num_motions = 100;
    float outlier_ratio = 0.80f;

    std::vector<Tensor> A_tensors, B_tensors;
    std::uniform_real_distribution<float> angle_dist(-180.0f, 180.0f);
    std::uniform_real_distribution<float> noise_dist(-2.0f, 2.0f); // +/- 2 degrees of real-world noise
    std::uniform_int_distribution<int> axis_dist(0, 2);
    char axes[] = {'X', 'Y', 'Z'};

    // 2. Generate Data
    for (int i = 0; i < num_motions; ++i) {
        char a_axis = axes[axis_dist(gen)];
        float a_angle = angle_dist(gen);
        std::vector<float> A_data = make_rotation(a_axis, a_angle);
        std::vector<float> B_data;

        if ((float)i / num_motions < (1.0f - outlier_ratio)) {
            // INLIER (But with sensor noise!)
            std::vector<float> AX = matmul_3x3(A_data, X_gt_data);
            std::vector<float> B_clean = matmul_3x3(X_gt_T_data, AX);

            // Inject 3D rotational noise into the camera measurement B
            char noise_axis = axes[axis_dist(gen)];
            float noise_angle = noise_dist(gen);
            std::vector<float> Noise_R = make_rotation(noise_axis, noise_angle);

            B_data = matmul_3x3(B_clean, Noise_R);
        } else {
            // OUTLIER (Vision system failed)
            char b_axis = axes[axis_dist(gen)];
            float b_angle = angle_dist(gen);
            B_data = make_rotation(b_axis, b_angle);
        }

        A_tensors.push_back(math::array(A_data, {3, 3}, dtype));
        B_tensors.push_back(math::array(B_data, {3, 3}, dtype));
    }

    // 3. Define the Robust Objective Function
    FuncObj robust_hand_eye_cost([A_tensors, B_tensors, dtype](const Tensor &X) {
        Tensor total_cost = Tensor(0.0, dtype);
        Tensor mu = Tensor(0.5, dtype);
        std::vector<int> batch_shape = X.shape();

        for (size_t i = 0; i < A_tensors.size(); ++i) {
            Tensor A_batch = math::broadcast_to(A_tensors[i], batch_shape);
            Tensor B_batch = math::broadcast_to(B_tensors[i], batch_shape);
            Tensor AX = math::matmul(A_batch, X);
            Tensor XB = math::matmul(X, B_batch);
            Tensor diff = math::subtract(AX, XB);
            Tensor sq_diff = math::square(diff);
            Tensor frob_sq = math::sum(sq_diff, {1, 2});
            Tensor robust_loss = math::divide(frob_sq, math::add(mu, frob_sq));
            total_cost = math::add(total_cost, robust_loss);
        }
        return total_cost;
    });

    // 4. Configure and Run Solver
    // We increase patience here (from 5 to 10) because navigating a noisy minimum takes more careful steps
    SolverConfig config = SOSolver::get_solver_config(core::Aggressive, d, std::make_shared<EnergyToleranceCriterion>(1e-6, 10), 0.95);

    SOSolver solver(config);
    CBOResult result = solver.solve(&robust_hand_eye_cost);

    // 5. Verification
    Tensor diff_gt = math::subtract(result.final_consensus, X_gt);
    double frob_err = math::to_double(math::sqrt(math::sum(math::square(diff_gt))));
    double angle_err_deg = 2.0 * std::asin(std::max(-1.0, std::min(1.0, frob_err / std::sqrt(8.0)))) * (180.0 / M_PI);

    std::cout << std::fixed << std::setprecision(6);
    std::cout << "Solver Final Energy:    " << result.min_energy << "\n";
    std::cout << "Rotation Error:         " << angle_err_deg << " degrees\n";

    // Because the inliers have up to 2 degrees of noise, a perfect 0.0 degree error is impossible.
    // An error under 3.0 degrees means the solver successfully found the noisy consensus!
    if (angle_err_deg < 3.0) {
        std::cout << "[STATUS] SUCCESS: Survived the nightmare! Solved within noise bounds.\n";
    } else {
        std::cout << "[STATUS] FAILURE: Solver broke under extreme noise and 80% outliers.\n";
    }
}

void run_robust_hand_eye_test(const TestScenario& scenario) {
    std::cout << "\n======================================================\n";
    std::cout << ">>> RUNNING TEST: " << scenario.name << " <<<\n";
    std::cout << "======================================================\n";

    const int d = 3;
    const involute::DType dtype = involute::DType::Float32;
    std::mt19937 gen;

    // 1. Ground Truth: Gripper-to-Camera (X) is rotated 90 deg around Y
    std::vector<float> X_gt_data = make_rotation('Y', 90.0f);
    Tensor X_gt = math::array(X_gt_data, {3, 3}, dtype);
    std::vector<float> X_gt_T_data = transpose_3x3(X_gt_data);

    std::vector<Tensor> A_tensors, B_tensors;
    std::uniform_real_distribution<float> angle_dist(-180.0f, 180.0f);
    std::uniform_int_distribution<int> axis_dist(0, 2);
    char axes[] = {'X', 'Y', 'Z'};

    // 2. Generate A (Robot Motions) and B (Camera Motions)
    for (int i = 0; i < scenario.num_motions; ++i) {
        // Randomly generate a valid robot arm movement
        char a_axis = axes[axis_dist(gen)];
        float a_angle = angle_dist(gen);
        std::vector<float> A_data = make_rotation(a_axis, a_angle);

        std::vector<float> B_data;

        // Decide if this measurement pair is an INLIER or OUTLIER
        if ((float)i / scenario.num_motions < (1.0f - scenario.outlier_ratio)) {
            // INLIER: B = X_gt^T * A * X_gt  (Perfect kinematic loop)
            std::vector<float> AX = matmul_3x3(A_data, X_gt_data);
            B_data = matmul_3x3(X_gt_T_data, AX);
        } else {
            // OUTLIER: Vision system completely failed, gave a random rotation
            char b_axis = axes[axis_dist(gen)];
            float b_angle = angle_dist(gen);
            B_data = make_rotation(b_axis, b_angle);
        }

        A_tensors.push_back(math::array(A_data, {3, 3}, dtype));
        B_tensors.push_back(math::array(B_data, {3, 3}, dtype));
    }

    // 3. Define the Robust Objective Function
    FuncObj robust_hand_eye_cost([A_tensors, B_tensors, dtype](const Tensor &X) {
        Tensor total_cost = Tensor(0.0, dtype);
        Tensor mu = Tensor(0.5, dtype); // Geman-McClure scale parameter
        std::vector<int> batch_shape = X.shape();

        for (size_t i = 0; i < A_tensors.size(); ++i) {
            Tensor A_batch = math::broadcast_to(A_tensors[i], batch_shape);
            Tensor B_batch = math::broadcast_to(B_tensors[i], batch_shape);

            // AX and XB
            Tensor AX = math::matmul(A_batch, X);
            Tensor XB = math::matmul(X, B_batch);

            // Distance metric: ||AX - XB||^2
            Tensor diff = math::subtract(AX, XB);
            Tensor sq_diff = math::square(diff);
            Tensor frob_sq = math::sum(sq_diff, {1, 2});

            // Robust Kernel: rho(d^2) = d^2 / (mu + d^2)
            // Caps the penalty so massive outliers don't skew the gradient
            Tensor robust_loss = math::divide(frob_sq, math::add(mu, frob_sq));

            total_cost = math::add(total_cost, robust_loss);
        }
        return total_cost;
    });

    // 4. Calculate True Global Min Energy (Baseline)
    Tensor X_gt_batched = math::reshape(X_gt, {1, 3, 3});
    double true_global_min = math::to_double(robust_hand_eye_cost.evaluate_batch(X_gt_batched));

    // 5. Configure and Run Solver
    /*
    SolverConfig config{
        .N = 150,
        .d = d,
        .params = HyperParameters { .beta = 1.0, .lambda = 1.0, .delta = 0.15, .h = 0.02 },
        .dtype = dtype,
        .convergence = std::make_shared<EnergyToleranceCriterion>(1e-8),
        .parameter_adapter = std::make_shared<SOParameterAdapter>()
    };*/
    SolverConfig config = SOSolver::get_solver_config(core::Aggressive, d, std::make_shared<EnergyToleranceCriterion>(1e-6, 5), 0.9);

    SOSolver solver(config);
    CBOResult result = solver.solve(&robust_hand_eye_cost);

    // 6. Verification
    Tensor diff_gt = math::subtract(result.final_consensus, X_gt);
    double frob_err = math::to_double(math::sqrt(math::sum(math::square(diff_gt))));
    double angle_err_deg = 2.0 * std::asin(std::max(-1.0, std::min(1.0, frob_err / std::sqrt(8.0)))) * (180.0 / M_PI);

    std::cout << std::fixed << std::setprecision(6);
    std::cout << "True Global Min Energy: " << true_global_min << "\n";
    std::cout << "Solver Final Energy:    " << result.min_energy << "\n";
    std::cout << "Rotation Error:         " << angle_err_deg << " degrees\n";

    // Because this is pure geometry without Gaussian point noise,
    // the error should be extremely close to 0 if the global minimum is found.
    if (angle_err_deg < 1.0 && result.min_energy <= (true_global_min + 0.05)) {
        std::cout << "[STATUS] SUCCESS: Ignored outliers and found the exact geometric truth.\n";
    } else {
        std::cout << "[STATUS] FAILURE: Solver got trapped by the outlier rotations.\n";
    }
}

int main() {
    /*
    std::vector<TestScenario> suite = {
        {"EASY: 20 Motions, 0% Outliers", 20, 0.00f, 5},
        {"MODERATE: 30 Motions, 25% Outliers", 30, 0.25f, 5},
        {"HARD: 40 Motions, 60% Outliers", 40, 0.60f, 5}
    };

    for (const auto& test : suite) {
        run_robust_hand_eye_test(test);
    }*/

    run_nightmare_hand_eye_test();

    return 0;
}