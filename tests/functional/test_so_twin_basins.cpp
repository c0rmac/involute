/**
 * @file test_so_twin_basins.cpp
 * @brief Tests the solver's ability to handle structural ambiguity (Twin Global Minima).
 */

#include "involute/solvers/so_solver.hpp"
#include "involute/core/objective.hpp"
#include "involute/core/math.hpp"
#include <iostream>
#include <vector>
#include <cmath>
#include <iomanip>

using namespace involute;
using namespace involute::core;
using namespace involute::solvers;

// ============================================================================
// THE SYMMETRIC BLACK BOX
// ============================================================================
// This field is designed to be perfectly symmetric around the origin.
// For every rotation R, there is a symmetric rotation R' that yields
// the exact same energy.
Tensor evaluate_symmetric_field(const Tensor& P_rotated, float freq, DType dtype) {
    Tensor px = math::slice(P_rotated, 0, 1, 1);
    Tensor py = math::slice(P_rotated, 1, 2, 1);
    Tensor pz = math::slice(P_rotated, 2, 3, 1);

    // Using absolute values or squares creates an "Ambiguity Trap"
    // The sensors can't tell if they are at +X or -X.
    Tensor x_sq = math::square(px);
    Tensor y_sq = math::square(py);
    Tensor z_sq = math::square(pz);

    Tensor e1 = math::sin(math::multiply(x_sq, Tensor(freq, dtype)));
    Tensor e2 = math::cos(math::multiply(y_sq, Tensor(freq, dtype)));
    Tensor e3 = math::sin(math::multiply(z_sq, Tensor(freq, dtype)));

    return math::add(math::add(e1, e2), e3);
}

int main() {
    const int d = 3;
    const int num_sensors = 30;
    const float freq = 15.0f;
    const involute::DType dtype = involute::DType::Float32;

    // 1. Setup Data (Symmetric body)
    std::vector<float> p_data;
    for(int i=0; i<num_sensors*3; ++i) p_data.push_back(((float)rand()/RAND_MAX)*2.0f - 1.0f);
    Tensor P_body = math::array(p_data, {3, num_sensors}, dtype);

    // 2. Target Signal (at Identity)
    std::vector<float> eye = {1,0,0, 0,1,0, 0,0,1};
    Tensor R_gt = math::array(eye, {3, 3}, dtype);

    // CRITICAL FIX: Add the batch dimension so the slicer works correctly
    Tensor R_gt_batched = math::reshape(R_gt, {1, 3, 3});
    Tensor P_gt_rotated = math::matmul(R_gt_batched, P_body);
    Tensor S_target = evaluate_symmetric_field(P_gt_rotated, freq, dtype);

    // 3. Objective
    FuncObj twin_cost([P_body, S_target, freq, dtype](const Tensor &X) {
        Tensor P_rot = math::matmul(X, P_body);
        Tensor S_pred = evaluate_symmetric_field(P_rot, freq, dtype);
        Tensor diff = math::subtract(S_pred, math::broadcast_to(S_target, S_pred.shape()));
        return math::sum(math::square(diff), {1, 2});
    });

    // 4. Run with 150,000 Particles
    SolverConfig config{
        .N = 150000,
        .d = d,
        .params = HyperParameters { .beta = 10.0, .lambda = 1.0, .delta = 0.8 },
        .dtype = dtype,
        .convergence = std::make_shared<MaxStepsCriterion>(100),
        .parameter_adapter = std::make_shared<SOParameterAdapter>()
    };

    SOSolver solver(config);
    CBOResult result = solver.solve(&twin_cost);

    std::cout << "Final Energy: " << result.min_energy << "\n";
    // Check if it found the GT or the 'Ghost'
    // (Both are mathematically 'Correct' Global Minima)
    return 0;
}