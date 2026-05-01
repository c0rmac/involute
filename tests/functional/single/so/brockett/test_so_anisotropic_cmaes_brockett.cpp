/**
 * @file test_so_anisotropic_cmaes_brockett.cpp
 * @brief SO(d) Anisotropic CMA-ES Solver — Brockett trace cost.
 * Algorithm 4.9 (Spatial precision, d×d covariance).
 * Global minimum: tr(X^T A X S) = sum_i a_i * s_i when columns of X are eigenvectors of A.
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

    std::string file_base = "brockett_d=" + std::to_string(d) + "_run=" + std::to_string(run_index);
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
              << "║   SO(d) Anisotropic CMA-ES — Brockett Test Suite      ║\n"
              << "╚═══════════════════════════════════════════════════════╝\n";

    const DType dtype = DType::Float32;

    struct BrockettDim { int d; int k; int steps; double sigma0; double rel_tol; };
    std::vector<BrockettDim> dims = {
        {3, 40, 400, 0.4, 1e-2},
        {5, 25, 600, 0.5, 1e-2},
    };

    int total = 0, passed_suites = 0;
    for (auto& bd : dims) {
        auto [cost, tmin] = make_brockett(bd.d, dtype);
        std::cout << "\n── Brockett d=" << bd.d << " | K=" << bd.k
                  << " | max_steps=" << bd.steps
                  << " | tmin=" << tmin << " ──\n";

        const double pass_threshold = tmin * (1.0 + bd.rel_tol);
        std::vector<int>    iters;
        std::vector<double> energies;
        int passes = 0;

        for (int k = 0; k < bd.k; k++) {
            auto r = run_scenario(bd.d, k, bd.steps, bd.sigma0, pass_threshold,
                                  cost, "cmaes_results/brockett");
            energies.push_back(r.final_energy);
            if (r.passed) { passes++; iters.push_back(r.iterations); }
            total++;
        }

        double rate = 100.0 * passes / bd.k;
        std::cout << "  Pass rate: " << std::fixed << std::setprecision(1) << rate << "%\n";
        if (!iters.empty()) {
            double avg = std::accumulate(iters.begin(), iters.end(), 0.0) / iters.size();
            std::cout << "  Avg iters: " << avg << "\n";
        }
        double best_e = *std::min_element(energies.begin(), energies.end());
        std::cout << "  Best energy: " << std::setprecision(6) << best_e << "\n";

        if (passes == bd.k) passed_suites++;
    }

    std::cout << "\nPassed suites: " << passed_suites << " / " << static_cast<int>(dims.size()) << "\n";
    if (passed_suites == static_cast<int>(dims.size())) { std::cout << "STATUS: ALL PASSED\n"; return 0; }
    std::cerr << "STATUS: SOME FAILURES DETECTED\n";
    return 1;
}
