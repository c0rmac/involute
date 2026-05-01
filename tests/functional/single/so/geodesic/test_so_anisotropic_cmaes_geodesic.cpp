/**
 * @file test_so_anisotropic_cmaes_geodesic.cpp
 * @brief SO(d) Anisotropic CMA-ES Solver — Geodesic distance to a random target.
 * Algorithm 4.9 (Spatial precision, d×d covariance).
 * Global minimum: f(R*) = 0 for a Haar-uniform random R* ∈ SO(d).
 */

#include "involute/solvers/anisotropic/so_anisotropic_solver_cmaes_spatial.hpp"
#include "involute/core/objective.hpp"
#include "involute/core/math.hpp"
#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <string>
#include <vector>

#include "../../../helpers/helper.cpp"
#include "../../functions/so_objectives.hpp"

using namespace involute;
using namespace involute::core;
using namespace involute::solvers;
using namespace involute::test;

struct RunResult {
    bool   passed;
    int    iterations;
    double final_energy;
    double sigma_final;
};

RunResult run_scenario(int d, int run_index, int max_steps,
                       double sigma0, double pass_threshold,
                       FuncObj& cost_fn, const std::string& csv_dir)
{
    const DType dtype = DType::Float32;

    SOAnisotropicSolverCMAESConfig config{
        .N          = 50,
        .d          = d,
        .convergence = std::make_shared<MaxStepsCriterion>(max_steps),
        .sigma0     = sigma0,
        .burn_in    = 400,
        .warm_start = 50,
        .gamma_rtol = 0.05,
        .dtype      = dtype,
        .debug      = std::vector<Debugger>({Debugger::Log, Debugger::History})
    };

    SOAnisotropicSolverCMAESSpatial solver(config);
    CBOResult result = solver.solve(&cost_fn);

    const double e  = result.min_energy;
    const bool   ok = (e < pass_threshold);

    std::string file_base = "geodesic_d=" + std::to_string(d) + "_run=" + std::to_string(run_index);
    utils::export_history_to_csv(result.history, csv_dir, file_base + ".csv");

    double sigma_final = result.history.empty() ? -1.0 : result.history.back().delta;

    if (ok) {
        std::cout << "  [PASS] run=" << run_index
                  << " E=" << std::fixed << std::setprecision(5) << e
                  << " iters=" << result.iterations_run
                  << " σ_final=" << std::setprecision(4) << sigma_final << "\n";
    } else {
        std::cerr << "  [FAIL] run=" << run_index
                  << " E=" << std::fixed << std::setprecision(5) << e
                  << " iters=" << result.iterations_run
                  << " (threshold=" << pass_threshold << ")\n";
    }

    return {ok, result.iterations_run, e, sigma_final};
}

int main(int /*argc*/, char* /*argv*/[]) {
    sampler::set_num_threads(8);
    math::set_default_device_cpu();

    std::cout << "╔═══════════════════════════════════════════════════════╗\n"
              << "║   SO(d) Anisotropic CMA-ES — Geodesic Test Suite      ║\n"
              << "╚═══════════════════════════════════════════════════════╝\n";

    const DType dtype = DType::Float32;

    struct GeodDim { int d; int k; int steps; double sigma0; };
    std::vector<GeodDim> dims = {
        {3, 40, 350, 0.4},
        {5, 25, 550, 0.5},
    };

    int total = 0, passed_suites = 0;
    for (auto& gd : dims) {
        Tensor R_star = haar_rotation(gd.d, dtype);
        math::eval(R_star);
        std::cout << "\n── Geodesic d=" << gd.d << " | K=" << gd.k
                  << " | max_steps=" << gd.steps << " ──\n";
        std::cout << "  Target R*:\n" << R_star << "\n";

        FuncObj cost = make_geodesic(gd.d, dtype, R_star);

        Tensor R_batch = math::expand_dims(R_star, {0});
        double e_at_target = math::to_double(cost.evaluate_batch(R_batch));
        std::cout << "  [Sanity] f(R*) = " << e_at_target << " (should be ~0)\n";

        std::vector<int>    iters;
        std::vector<double> energies;
        int passes = 0;

        for (int k = 0; k < gd.k; k++) {
            auto r = run_scenario(gd.d, k, gd.steps, gd.sigma0, 1e-3,
                                  cost, "cmaes_results/geodesic");
            energies.push_back(r.final_energy);
            if (r.passed) { passes++; iters.push_back(r.iterations); }
            total++;
        }

        double rate = 100.0 * passes / gd.k;
        std::cout << "  Pass rate: " << std::fixed << std::setprecision(1) << rate << "%\n";
        if (!iters.empty()) {
            double avg = std::accumulate(iters.begin(), iters.end(), 0.0) / iters.size();
            std::cout << "  Avg iters: " << avg << "\n";
        }
        double best_e = *std::min_element(energies.begin(), energies.end());
        std::cout << "  Best energy: " << std::setprecision(6) << best_e << "\n";

        if (passes == gd.k) passed_suites++;
    }

    std::cout << "\nPassed suites: " << passed_suites << " / " << static_cast<int>(dims.size()) << "\n";
    if (passed_suites == static_cast<int>(dims.size())) { std::cout << "STATUS: ALL PASSED\n"; return 0; }
    std::cerr << "STATUS: SOME FAILURES DETECTED\n";
    return 1;
}
