/**
 * @file test_product_robotics_rastrigin.cpp
 * @brief CBO on a 24-dimensional product space (SO(3) x SO(2)^23) simulating a humanoid robot.
 * Features a Highly Coupled Rastrigin landscape with non-differentiable "Contact Cliffs".
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

bool run_scenario(int run_index, const std::vector<int>& dims, double delta, int max_steps) {
    if (dims.empty()) {
        std::cerr << "[FAIL] No dimensions provided.\n";
        return false;
    }

    const DType dtype = DType::Float32;

    // 1. Build identity matrices
    std::vector<Tensor> identities;
    double total_coords = 0.0;
    std::string tag_desc = "SO(3)xSO(2)^23_Humanoid";
    std::string file_prefix = "robotics_rastrigin";

    for (size_t i = 0; i < dims.size(); ++i) {
        identities.push_back(math::eye(dims[i], dtype));
        total_coords += (dims[i] * dims[i]);
    }

    // 2. The Robotics-Rastrigin Objective Function
    FuncProductObj objective([=](const std::vector<Tensor> &X) {
        const double A = 10.0;
        const double c = 2.0 * 3.14159265358979323846;
        const double kappa = 5.0;

        const double cliff_threshold = 0.06;
        const double cliff_penalty = 500.0;

        Tensor total_cost = Tensor(0.0, dtype);
        Tensor prev_sq = Tensor(0.0, dtype);

        for (size_t i = 0; i < dims.size(); ++i) {
            Tensor diff = math::subtract(X[i], identities[i]);
            Tensor scaled_diff = math::divide(diff, Tensor(4.0, dtype));
            Tensor sq = math::sum(math::square(scaled_diff), {1, 2});

            double num_elements = dims[i] * dims[i];
            Tensor cos_term = math::cos(math::multiply(scaled_diff, Tensor(c, dtype)));
            Tensor rastrigin_local = math::multiply(Tensor(A, dtype),
                                     math::subtract(Tensor(num_elements, dtype), math::sum(cos_term, {1, 2})));

            Tensor coupled_sq = sq;
            if (i > 0) {
                Tensor cross_term = math::multiply(math::multiply(sq, prev_sq), Tensor(kappa, dtype));
                coupled_sq = math::add(sq, cross_term);
            }
            prev_sq = sq;

            Tensor is_fallen = math::greater(sq, Tensor(cliff_threshold, dtype));
            Tensor cliff_wall = math::where(is_fallen, Tensor(cliff_penalty, dtype), Tensor(0.0, dtype));

            Tensor joint_cost = math::add(coupled_sq, rastrigin_local);
            joint_cost = math::add(joint_cost, cliff_wall);
            total_cost = math::add(total_cost, joint_cost);
        }

        return total_cost;
    });

    // 3. Manifold specifications (High noise needed to escape the cliffs)
    std::vector<ManifoldSpec> manifolds;
    manifolds.reserve(dims.size());
    for (int dim : dims) {
        manifolds.push_back(ManifoldSpec::SO(dim, 1.0, delta));
    }

    ProductIsotropicSolverConfig cfg{
        .manifolds         = manifolds,
        .N                 = 150,
        .dtype             = dtype,
        .convergence       = std::make_shared<MaxStepsCriterion>(max_steps),
        .adapter = std::make_shared<CMAESParameterAdapter>(0.3),
        .debug             = {Debugger::Log, Debugger::History}
    };

    std::cout << "\n--- [" << tag_desc << " | Run " << run_index << "] ---\n";

    ProductIsotropicSolver solver(cfg);
    ProductCBOResult result = solver.solve(&objective);

    std::cout << "\nFinal energy: " << result.min_energy
              << " | Steps: " << result.iterations_run << "\n";

    std::string tag = file_prefix + "_run=" + std::to_string(run_index);
    utils::export_history_to_csv(result.history, "product_robotics", tag + ".solver.csv");

    if (result.min_energy > 1.0) {
        std::cerr << "[FAIL] Run " << run_index << " got stuck in a pothole or fell off a cliff. Energy=" << result.min_energy << "\n";
        return false;
    }
    std::cout << "[PASS] Run " << run_index << "\n";
    return true;
}

int main() {
    sampler::set_num_threads(8);
    math::set_default_device_gpu();

    const int    k_runs    = 50;
    const double delta     = 2.5;
    const int    max_steps = 350;

    // The 24-Manifold Cartesian Product Space: SO(3) x SO(2)^23 (Unitree G1 Humanoid)
    std::vector<int> target_manifolds = {
        3, // Base Orientation
        2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, // 23 Hinge Joints
    };

    int passed = 0;
    std::cout << "=== Product CBO: Robotics Rastrigin + Cliffs ===\n";

    for (int r = 0; r < k_runs; r++) {
        if (run_scenario(r, target_manifolds, delta, max_steps)) passed++;
    }

    std::cout << "\n=== Summary: " << passed << " / " << k_runs << " passed ===\n";
    return (passed == k_runs) ? 0 : 1;
}
