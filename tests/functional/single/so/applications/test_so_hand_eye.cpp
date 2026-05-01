/**
 * @file test_so_hand_eye.cpp
 * @brief Robust Hand-Eye Calibration (AX = XB) test on SO(3).
 * Tests the solver's ability to reject completely invalid rotation
 * measurements (outliers) using a robust estimator on the manifold.
 */

#include "involute/solvers/isotropic/so_isotropic_solver.hpp"
#include "involute/solvers/adapters/adam_parameter_adapter.hpp"
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
    const int N = 20 * d * d;
    std::mt19937 gen;

    // 1. Ground Truth
    std::vector<float> X_gt_data = make_rotation('Y', 90.0f);
    Tensor X_gt = math::array(X_gt_data, {3, 3}, dtype);
    std::vector<float> X_gt_T_data = transpose_3x3(X_gt_data);

    // 100 motions, but 80 of them are pure garbage.
    int num_motions = 100;
    float outlier_ratio = 0.80f;

    std::vector<Tensor> A_tensors, B_tensors;
    std::uniform_real_distribution<float> angle_dist(-180.0f, 180.0f);
    std::uniform_real_distribution<float> noise_dist(-2.0f, 2.0f);
    std::uniform_int_distribution<int> axis_dist(0, 2);
    char axes[] = {'X', 'Y', 'Z'};

    // 2. Generate Data
    for (int i = 0; i < num_motions; ++i) {
        char a_axis = axes[axis_dist(gen)];
        float a_angle = angle_dist(gen);
        std::vector<float> A_data = make_rotation(a_axis, a_angle);
        std::vector<float> B_data;

        if ((float)i / num_motions < (1.0f - outlier_ratio)) {
            std::vector<float> AX = matmul_3x3(A_data, X_gt_data);
            std::vector<float> B_clean = matmul_3x3(X_gt_T_data, AX);
            char noise_axis = axes[axis_dist(gen)];
            float noise_angle = noise_dist(gen);
            std::vector<float> Noise_R = make_rotation(noise_axis, noise_angle);
            B_data = matmul_3x3(B_clean, Noise_R);
        } else {
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
    const double lr = std::log(static_cast<double>(N)) / (d * d);
    SOIsotropicSolverConfig config{
        .N          = N,
        .d          = d,
        .convergence = std::make_shared<EnergyToleranceCriterion>(1e-6, 10),
        .adapter    = std::make_shared<AdamParameterAdapter>(0.8, 0.9, 0.999, 1e-8, lr),
        .lambda     = 1.0,
        .delta      = 0.95,
        .dtype      = dtype,
        .debug      = std::vector<Debugger>({Debugger::Log})
    };

    SOIsotropicSolver solver(config);
    CBOResult result = solver.solve(&robust_hand_eye_cost);

    // 5. Verification
    Tensor diff_gt = math::subtract(result.final_consensus[0], X_gt);
    double frob_err = math::to_double(math::sqrt(math::sum(math::square(diff_gt))));
    double angle_err_deg = 2.0 * std::asin(std::max(-1.0, std::min(1.0, frob_err / std::sqrt(8.0)))) * (180.0 / M_PI);

    std::cout << std::fixed << std::setprecision(6);
    std::cout << "Solver Final Energy:    " << result.min_energy << "\n";
    std::cout << "Rotation Error:         " << angle_err_deg << " degrees\n";

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
    const int N = 20 * d * d;
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
        char a_axis = axes[axis_dist(gen)];
        float a_angle = angle_dist(gen);
        std::vector<float> A_data = make_rotation(a_axis, a_angle);
        std::vector<float> B_data;

        if ((float)i / scenario.num_motions < (1.0f - scenario.outlier_ratio)) {
            std::vector<float> AX = matmul_3x3(A_data, X_gt_data);
            B_data = matmul_3x3(X_gt_T_data, AX);
        } else {
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

    // 4. Calculate True Global Min Energy (Baseline)
    Tensor X_gt_batched = math::reshape(X_gt, {1, 3, 3});
    double true_global_min = math::to_double(robust_hand_eye_cost.evaluate_batch(X_gt_batched));

    // 5. Configure and Run Solver
    const double lr = std::log(static_cast<double>(N)) / (d * d);
    SOIsotropicSolverConfig config{
        .N          = N,
        .d          = d,
        .convergence = std::make_shared<EnergyToleranceCriterion>(1e-6, 5),
        .adapter    = std::make_shared<AdamParameterAdapter>(0.8, 0.9, 0.999, 1e-8, lr),
        .lambda     = 1.0,
        .delta      = 0.9,
        .dtype      = dtype,
        .debug      = std::vector<Debugger>({Debugger::Log})
    };

    SOIsotropicSolver solver(config);
    CBOResult result = solver.solve(&robust_hand_eye_cost);

    // 6. Verification
    Tensor diff_gt = math::subtract(result.final_consensus[0], X_gt);
    double frob_err = math::to_double(math::sqrt(math::sum(math::square(diff_gt))));
    double angle_err_deg = 2.0 * std::asin(std::max(-1.0, std::min(1.0, frob_err / std::sqrt(8.0)))) * (180.0 / M_PI);

    std::cout << std::fixed << std::setprecision(6);
    std::cout << "True Global Min Energy: " << true_global_min << "\n";
    std::cout << "Solver Final Energy:    " << result.min_energy << "\n";
    std::cout << "Rotation Error:         " << angle_err_deg << " degrees\n";

    if (angle_err_deg < 1.0 && result.min_energy <= (true_global_min + 0.05)) {
        std::cout << "[STATUS] SUCCESS: Ignored outliers and found the exact geometric truth.\n";
    } else {
        std::cout << "[STATUS] FAILURE: Solver got trapped by the outlier rotations.\n";
    }
}

int main() {
    run_nightmare_hand_eye_test();
    return 0;
}
