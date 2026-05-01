/**
 * @file test_product_v63_so3.cpp
 * @brief CBO on an arbitrary Cartesian product of Stiefel and SO manifolds,
 * with per-manifold lambda and delta parameters and a COUPLED Ackley objective.
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

// Helper struct to define arbitrary manifolds with custom parameters
struct TargetManifold {
    enum Type { STIEFEL, SO } type;
    int n;
    int p;             // For SO(d), p is effectively ignored
    double lambda_val; // Weight of the consensus drift term
    double delta_val;  // Weight of the noise term
};

bool run_scenario(int run_index, const std::vector<TargetManifold>& targets, int max_steps) {
    if (targets.empty()) {
        std::cerr << "[FAIL] No dimensions provided.\n";
        return false;
    }

    const DType dtype = DType::Float32;

    std::vector<Tensor> identities;
    std::vector<ManifoldSpec> manifolds;
    double total_coords = 0.0;
    std::string tag_desc = "";
    std::string file_prefix = "";

    // 1. Dynamically build identity matrices, coordinate counts, specs, and log tags
    for (size_t i = 0; i < targets.size(); ++i) {
        const auto& t = targets[i];

        if (t.type == TargetManifold::STIEFEL) {
            Tensor I = math::eye(t.n, dtype);
            identities.push_back(math::slice(I, 0, t.p, 1)); // I_{n,p}
            total_coords += (t.n * t.p);

            manifolds.push_back(ManifoldSpec::Stiefel(t.n, t.p, t.lambda_val, t.delta_val));

            tag_desc += "V(" + std::to_string(t.n) + "," + std::to_string(t.p) + ")";
            file_prefix += "v" + std::to_string(t.n) + "_" + std::to_string(t.p);
        }
        else if (t.type == TargetManifold::SO) {
            identities.push_back(math::eye(t.n, dtype));
            total_coords += (t.n * t.n);

            manifolds.push_back(ManifoldSpec::SO(t.n, t.lambda_val, t.delta_val));

            tag_desc += "SO(" + std::to_string(t.n) + ")";
            file_prefix += "so" + std::to_string(t.n);
        }

        if (i < targets.size() - 1) {
            tag_desc += "x";
            file_prefix += "x";
        }
    }

    // 2. The Highly COUPLED Objective Function
    FuncProductObj objective([=](const std::vector<Tensor> &X) {
        const double a = 20.0;
        const double b = 0.2;
        const double c = 2.0 * 3.14159265358979323846;

        const double kappa = 10.0;

        Tensor sum_sq;
        Tensor sum_cos;
        Tensor prev_sq;

        for (size_t i = 0; i < identities.size(); ++i) {
            Tensor diff = math::subtract(X[i], identities[i]);

            Tensor scaled_diff = math::divide(diff, Tensor(4.0, dtype));

            Tensor sq = math::sum(math::square(scaled_diff), {1, 2});
            Tensor cs = math::sum(math::cos(math::multiply(scaled_diff, Tensor(c, dtype))), {1, 2});

            Tensor coupled_sq = sq;
            if (i > 0) {
                Tensor cross_term = math::multiply(sq, prev_sq);
                cross_term = math::multiply(cross_term, Tensor(kappa, dtype));
                coupled_sq = math::add(sq, cross_term);
            }
            prev_sq = sq;

            if (i == 0) {
                sum_sq = coupled_sq;
                sum_cos = cs;
            } else {
                sum_sq = math::add(sum_sq, coupled_sq);
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

    ProductIsotropicSolverConfig cfg{
        .manifolds         = manifolds,
        .N                 = 250,
        .dtype             = dtype,
        .convergence       = std::make_shared<MaxStepsCriterion>(max_steps),
        .adapter = std::make_shared<CMAESParameterAdapter>(0.2),
        .debug             = {Debugger::Log, Debugger::History}
    };

    std::cout << "\n--- [COUPLED: " << tag_desc << " | Run " << run_index << "] ---\n";

    ProductIsotropicSolver solver(cfg);
    ProductCBOResult result = solver.solve(&objective);

    std::cout << "\nFinal energy: " << result.min_energy
              << " | Steps: " << result.iterations_run << "\n";

    std::string tag = file_prefix + "_coupled_run=" + std::to_string(run_index);
    utils::export_history_to_csv(result.history, "product_dynamic_coupled", tag + ".solver.csv");

    if (result.min_energy > 0.5) {
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

    const int k_runs    = 10;
    const int max_steps = 8000;

    std::vector<TargetManifold> my_manifolds = {
        {TargetManifold::STIEFEL, 6,  3, 1.0, 0.9},
        {TargetManifold::STIEFEL, 12, 6, 1.0, 0.9},
        {TargetManifold::SO,      3,  3, 1.0, 1.2},
        {TargetManifold::SO,      50, 50, 1.0, 0.1}
    };

    int passed = 0;
    std::cout << "=== Product CBO: COUPLED Arbitrary Dynamic Manifolds (V(6,3) x V(12,6) x SO(3) x SO(50)) ===\n";

    for (int r = 0; r < k_runs; r++) {
        if (run_scenario(r, my_manifolds, max_steps)) passed++;
    }

    std::cout << "\n=== Summary: " << passed << " / " << k_runs << " passed ===\n";
    return (passed == k_runs) ? 0 : 1;
}
