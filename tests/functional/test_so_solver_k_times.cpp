/**
 * @file test_so_solver_k_times.cpp
 * @brief Pure C++ execution entry point for the SO(d) Solver, running K times.
 */

#include "involute/solvers/so_solver.hpp"
#include "involute/core/objective.hpp"
#include "involute/core/math.hpp"
#include <iostream>
#include <cmath>
#include <algorithm>
#include <vector>
#include <numeric>
#include <iomanip>

using namespace involute;
using namespace involute::core;
using namespace involute::solvers;

/**
 * @brief Calculates the Global Dissipation Floor (lambda_abs) for SO(d).
 * @param d Dimension of the Special Orthogonal Group SO(d).
 * @param delta Lookahead / Perturbation parameter.
 * @param epsilon Tolerance for the wrapping probability (default 0.05).
 * @return The minimum dissipation rate lambda required for stability.
 */
double find_lambda_abs(int d, double delta, double epsilon = 0.05) {
    double D = d * (d - 1.0) / 2.0;
    double y = (2.0 / D) * std::log(1.0 / epsilon);

    double alpha = std::max(1.1, y + 1.0);
    for (int i = 0; i < 50; ++i) {
        double f_val = alpha - 1.0 - std::log(alpha) - y;
        double f_prime = 1.0 - (1.0 / alpha);

        double step = f_val / f_prime;
        alpha -= step;

        if (std::abs(step) < 1e-7) {
            break;
        }
    }

    const double PI = 3.14159265358979323846;
    double z_crit = (PI * PI / 4.0) + delta;
    double sqrt_z = std::sqrt(z_crit);
    double gamma_val = -sqrt_z / std::tan(sqrt_z);

    double prefactor = (d * (d - 1.0) * (delta * delta)) / (4.0 * (PI * PI));
    double lambda_abs = prefactor * alpha * (1.0 - gamma_val);

    return lambda_abs;
}

int main(int argc, char *argv[]) {
    std::cout << "--- Initializing K-Runs SO(d) Workload ---\n";

    const int K = 200; // Number of times to run the solver
    const int d = 10;
    const involute::DType target_dtype = involute::DType::Float32;

    // Objective: Ackley Function centered at the Identity matrix
    FuncObj ackley_cost([d](const Tensor &X) {
        const double a = 20.0;
        const double b = 0.2;
        const double c = 2.0 * 3.14159265358979323846;
        const double n = d * d;

        Tensor I = math::eye(d, target_dtype);
        Tensor diff = math::subtract(X, I);

        Tensor sq_diff = math::square(diff);
        Tensor sum_sq = math::sum(sq_diff, {1, 2});
        Tensor mean_sq = math::divide(sum_sq, Tensor(n, target_dtype));
        Tensor sqrt_mean_sq = math::sqrt(mean_sq);
        Tensor term1_inner = math::multiply(Tensor(-b, target_dtype), sqrt_mean_sq);
        Tensor term1 = math::multiply(Tensor(-a, target_dtype), math::exp(term1_inner));

        Tensor c_diff = math::multiply(diff, Tensor(c, target_dtype));
        Tensor cos_diff = math::cos(c_diff);
        Tensor sum_cos = math::sum(cos_diff, {1, 2});
        Tensor mean_cos = math::divide(sum_cos, Tensor(n, target_dtype));
        Tensor term2 = math::multiply(Tensor(-1.0, target_dtype), math::exp(mean_cos));

        double constant_term = a + std::exp(1.0);
        Tensor res = math::add(term1, term2);
        return math::add(res, Tensor(constant_term, target_dtype));
    });

    const double delta_param = 0.56;


    //std::cout << "[Config] Running K=" << K << " iterations. N=" << config.N << ", d=" << config.d << "\n";
    //std::cout << "[Run] Starting optimization loop...\n\n";

    std::vector<int> successful_iterations;
    successful_iterations.reserve(K);
    int total_successes = 0;

    for (int k = 0; k < K; ++k) {
        std::cout << "Experiment: " << k << "\n\n";
        SolverConfig config{
            .N=200 * 10,
            .d = d,
            .params = HyperParameters {
                .beta = 1.0,
                .lambda = 1.0,
                .delta = delta_param
            },
            .dtype = target_dtype,
            .convergence = std::make_shared<ConsensusToleranceCriterion>(1e-2, 5),
            .parameter_adapter = std::make_shared<SOParameterAdapter>()
        };

        // Instantiate the solver fresh for each run to avoid carrying over state
        SOSolver solver(config);
        CBOResult result = solver.solve(&ackley_cost);

        if (result.min_energy < 0.03) {
            total_successes++;
            successful_iterations.push_back(result.iterations_run);
        } else {
            successful_iterations;
        }
    }

    std::cout << "--- K-Runs Optimization Complete ---\n";

    // Calculate Statistics
    double success_rate = (static_cast<double>(total_successes) / K) * 100.0;

    std::cout << "Total Runs (K): " << K << "\n";
    std::cout << "Success Rate (min_energy < 0.05): " << std::fixed << std::setprecision(2) << success_rate << "%\n";

    if (total_successes > 0) {
        // Average Iterations
        double sum_iterations = std::accumulate(successful_iterations.begin(), successful_iterations.end(), 0.0);
        double avg_iterations = sum_iterations / total_successes;

        // Median Iterations
        std::sort(successful_iterations.begin(), successful_iterations.end());
        double median_iterations = 0.0;
        size_t n = successful_iterations.size();

        if (n % 2 == 0) {
            median_iterations = (successful_iterations[n / 2 - 1] + successful_iterations[n / 2]) / 2.0;
        } else {
            median_iterations = successful_iterations[n / 2];
        }

        std::cout << "Average Iterations (successful runs): " << avg_iterations << "\n";
        std::cout << "Median Iterations (successful runs):  " << median_iterations << "\n";
    } else {
        std::cout << "Average Iterations: N/A (0 successes)\n";
        std::cout << "Median Iterations:  N/A (0 successes)\n";
    }

    return 0;
}