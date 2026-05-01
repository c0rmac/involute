/**
 * @file test_product_so3_so2_so2.cpp
 * @brief CBO on an arbitrary Cartesian product of SO(N) manifolds with Ackley objective.
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

    // 1. Dynamically build identity matrices, coordinate count, and log tags
    std::vector<Tensor> identities;
    double total_coords = 0.0;
    std::string tag_desc = "";
    std::string file_prefix = "";

    for (size_t i = 0; i < dims.size(); ++i) {
        identities.push_back(math::eye(dims[i], dtype));
        total_coords += (dims[i] * dims[i]);

        tag_desc += "SO(" + std::to_string(dims[i]) + ")";
        file_prefix += "so" + std::to_string(dims[i]);
        if (i < dims.size() - 1) {
            tag_desc += "x";
            file_prefix += "x";
        }
    }

    // 2. Loop dynamically inside the Objective Function
    FuncProductObj objective([=](const std::vector<Tensor> &X) {
        const double a = 20.0;
        const double b = 0.2;
        const double c = 2.0 * 3.14159265358979323846;

        Tensor sum_sq;
        Tensor sum_cos;

        for (size_t i = 0; i < dims.size(); ++i) {
            Tensor diff = math::subtract(X[i], identities[i]);

            // Scale the residuals by 4.0 to prevent 2*pi aliasing.
            Tensor scaled_diff = math::divide(diff, Tensor(4.0, dtype));

            Tensor sq = math::sum(math::square(scaled_diff), {1, 2});
            Tensor cs = math::sum(math::cos(math::multiply(scaled_diff, Tensor(c, dtype))), {1, 2});

            if (i == 0) {
                sum_sq = sq;
                sum_cos = cs;
            } else {
                sum_sq = math::add(sum_sq, sq);
                sum_cos = math::add(sum_cos, cs);
            }
        }

        Tensor mean_sq = math::divide(sum_sq, Tensor(total_coords, dtype));
        Tensor sqrt_mean_sq = math::sqrt(mean_sq);
        Tensor term1_inner = math::multiply(Tensor(-b, dtype), sqrt_mean_sq);
        Tensor term1 = math::multiply(Tensor(-a, dtype), math::exp(term1_inner));

        Tensor mean_cos = math::divide(sum_cos, Tensor(total_coords, dtype));
        Tensor term2 = math::multiply(Tensor(-1.0, dtype), math::exp(mean_cos));

        double constant_term = a + std::exp(1.0);
        Tensor res = math::add(term1, term2);
        return math::add(res, Tensor(constant_term, dtype));
    });

    // 3. Dynamically populate manifold specifications
    std::vector<ManifoldSpec> manifolds;
    manifolds.reserve(dims.size());
    for (int dim : dims) {
        manifolds.push_back(ManifoldSpec::SO(dim, 1.0, delta));
    }

    auto c = 1.0 / std::sqrt(43);

    ProductIsotropicSolverConfig cfg{
        .manifolds         = manifolds,
        .N                 = 150,
        .dtype             = dtype,
        .convergence       = std::make_shared<MaxStepsCriterion>(max_steps),
        .adapter = std::make_shared<CMAESParameterAdapter>(0.5, 0.5*c, 1.0),
        .debug             = {Debugger::Log, Debugger::History}
    };

    std::cout << "\n--- [" << tag_desc << " | Run " << run_index << "] ---\n";

    ProductIsotropicSolver solver(cfg);
    ProductCBOResult result = solver.solve(&objective);

    std::cout << "\nFinal energy: " << result.min_energy
              << " | Steps: " << result.iterations_run << "\n";

    for (auto c : result.final_consensus) {
        std::cout << c << std::endl;
    }

    std::string tag = file_prefix + "_run=" + std::to_string(run_index);
    utils::export_history_to_csv(result.history, "product_dynamic", tag + ".solver.csv");

    if (result.min_energy > 0.3) {
        std::cerr << "[FAIL] Run " << run_index << " energy=" << result.min_energy << "\n";
        return false;
    }
    std::cout << "[PASS] Run " << run_index << "\n";
    return true;
}

int main() {
    std::srand(42);
    sampler::set_num_threads(8);
    math::set_default_device_gpu();

    const int    k_runs    = 10;
    const double delta     = 3.0;
    const int    max_steps = 400;

    std::vector<int> target_manifolds = {3, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2};

    int passed = 0;
    std::cout << "=== Product CBO: Dynamic Manifolds ===\n";

    for (int r = 0; r < k_runs; r++) {
        if (run_scenario(r, target_manifolds, delta, max_steps)) passed++;
    }

    std::cout << "\n=== Summary: " << passed << " / " << k_runs << " passed ===\n";
    return (passed == k_runs) ? 0 : 1;
}
