/**
 * @file test_so_anisotropic_cmaes_ackley.cpp
 * @brief SO(d) Anisotropic CMA-ES Solver — Ackley objective.
 * Algorithm 4.9 (Spatial precision, d×d covariance).
 * Global minimum: f(I) = 0.
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
                       const std::string& csv_dir)
{
    const DType dtype = DType::Float32;
    FuncObj cost = make_ackley(d, dtype);

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
    CBOResult result = solver.solve(&cost);

    const double e  = result.min_energy;
    const bool   ok = (e < pass_threshold);

    std::string file_base = "ackley_d=" + std::to_string(d) + "_run=" + std::to_string(run_index);
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
              << "║   SO(d) Anisotropic CMA-ES — Ackley Test Suite        ║\n"
              << "╚═══════════════════════════════════════════════════════╝\n";

    struct AckleyDim { int d; int k; int steps; double sigma0; };
    std::vector<AckleyDim> dims = {
        {20, 20, 2800, 50.5},
    };

    int total = 0, passed = 0;
    for (auto& ad : dims) {
        std::cout << "\n── Ackley d=" << ad.d << " | K=" << ad.k
                  << " | max_steps=" << ad.steps << " ──\n";
        std::vector<int>    iters;
        std::vector<double> energies;

        for (int k = 0; k < ad.k; k++) {
            auto r = run_scenario(ad.d, k, ad.steps, ad.sigma0, 0.05,
                                  "cmaes_results/ackley");
            energies.push_back(r.final_energy);
            if (r.passed) { passed++; iters.push_back(r.iterations); }
            total++;
        }

        double rate = 100.0 * static_cast<int>(iters.size()) / ad.k;
        std::cout << "  Pass rate: " << std::fixed << std::setprecision(1) << rate << "%\n";
        if (!iters.empty()) {
            double avg = std::accumulate(iters.begin(), iters.end(), 0.0) / iters.size();
            std::cout << "  Avg iters: " << avg << "\n";
        }
        double best_e = *std::min_element(energies.begin(), energies.end());
        std::cout << "  Best energy: " << std::setprecision(6) << best_e << "\n";
    }

    std::cout << "\nPassed: " << passed << " / " << total << "\n";
    if (passed == total) { std::cout << "STATUS: ALL PASSED\n"; return 0; }
    std::cerr << "STATUS: SOME FAILURES DETECTED\n";
    return 1;
}
