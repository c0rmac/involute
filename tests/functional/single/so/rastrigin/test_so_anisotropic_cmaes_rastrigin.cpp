/**
 * @file test_so_anisotropic_cmaes_rastrigin.cpp
 * @brief SO(d) Anisotropic CMA-ES Solver — Rastrigin objective.
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
    FuncObj cost = make_rastrigin(d, dtype);

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

    std::string file_base = "rastrigin_d=" + std::to_string(d) + "_run=" + std::to_string(run_index);
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
              << "║   SO(d) Anisotropic CMA-ES — Rastrigin Test Suite     ║\n"
              << "╚═══════════════════════════════════════════════════════╝\n";

    const DType dtype = DType::Float32;

    struct RastriginDim { int d; int k; int steps; double sigma0; };
    std::vector<RastriginDim> dims = {
        {3, 30, 1500, 10.5},
    };

    int total = 0, passed_suites = 0;
    for (auto& rd : dims) {
        FuncObj cost = make_rastrigin(rd.d, dtype);

        Tensor I_batch = math::expand_dims(math::eye(rd.d, dtype), {0});
        double e_at_identity = math::to_double(cost.evaluate_batch(I_batch));
        std::cout << "\n── Rastrigin d=" << rd.d << " | K=" << rd.k
                  << " | max_steps=" << rd.steps << " ──\n";
        std::cout << "  [Sanity] f(I) = " << e_at_identity << " (should be ~0)\n";

        std::vector<int>    iters;
        std::vector<double> energies;
        int passes = 0;

        for (int k = 0; k < rd.k; k++) {
            auto r = run_scenario(rd.d, k, rd.steps, rd.sigma0, 0.05,
                                  "cmaes_results/rastrigin");
            energies.push_back(r.final_energy);
            if (r.passed) { passes++; iters.push_back(r.iterations); }
            total++;
        }

        double rate = 100.0 * passes / rd.k;
        std::cout << "  Pass rate: " << std::fixed << std::setprecision(1) << rate << "%\n";
        if (!iters.empty()) {
            double avg = std::accumulate(iters.begin(), iters.end(), 0.0) / iters.size();
            std::cout << "  Avg iters: " << avg << "\n";
        }
        double best_e = *std::min_element(energies.begin(), energies.end());
        std::cout << "  Best energy: " << std::setprecision(6) << best_e << "\n";

        if (passes == rd.k) passed_suites++;
    }

    std::cout << "\nPassed: " << passed_suites << " / " << static_cast<int>(dims.size()) << " suite(s)\n";
    if (passed_suites == static_cast<int>(dims.size())) { std::cout << "STATUS: ALL PASSED\n"; return 0; }
    std::cerr << "STATUS: SOME FAILURES DETECTED\n";
    return 1;
}
