#pragma once

#include "involute/core/parameter_adapter.hpp"
#include "involute/core/base_solver.hpp"
#include "involute/core/math.hpp"
#include <cmath>
#include <algorithm>
#include <iostream>

using namespace involute::core;

namespace involute::solvers {
    class SOParameterAdapter : public core::ParameterAdapter {
    private:
        int patience_ = 4; // 1 d=20 -> 3; d=10 -> 4;
        double relative_contraction_rate_ = 0.0;

        double previous_energy_ = std::numeric_limits<double>::max();
        int wait_ = 0;

    public:
        double max_beta_achieved_ = 0;

        /**
         * @brief Constructor that solves for 'q' to make initial_beta and initial_lambda valid.
         */
        SOParameterAdapter(int patience, double relative_contraction_rate)
            : patience_(patience), relative_contraction_rate_(relative_contraction_rate) {
        }

        inline double solve_for_target_beta(const std::vector<float> &shifted_costs,
                                            double target_max_weight = 0.95,
                                            double tol = 1e-7,
                                            int max_iter = 100) {
            double target_sum = 1.0 / target_max_weight;

            // 1. Separate the zeros from the strictly positive costs
            std::vector<double> valid_costs;
            int num_zeros = 0;

            for (float c: shifted_costs) {
                if (c <= 1e-8f) {
                    // Account for float inaccuracies near 0
                    num_zeros++;
                } else {
                    valid_costs.push_back(static_cast<double>(c));
                }
            }

            // Edge case: Too many particles are tied for 1st place
            // e.g., if 2 particles share the minimum energy, the max weight is capped at 0.5.
            if (valid_costs.empty() || num_zeros >= target_sum) {
                return max_beta_achieved_; // Return a small, safe beta to enforce uniform distribution
            }

            // 2. Find the minimum strictly positive cost
            double min_c = valid_costs[0];
            for (double c: valid_costs) {
                if (c < min_c) min_c = c;
            }

            // 3. Establish the guaranteed mathematical bounds
            double required_tail_sum = target_sum - num_zeros;

            // Exact upper bound calculation
            double high = -std::log(required_tail_sum / valid_costs.size()) / min_c;
            double low = 0.0;

            // 4. Bisection Search (Guaranteed to converge in ~50 iterations)
            double beta = 0.0;
            for (int iter = 0; iter < max_iter; ++iter) {
                beta = low + 0.5 * (high - low);

                double current_sum = num_zeros;
                for (double c: valid_costs) {
                    current_sum += std::exp(-beta * c);
                }

                if (std::abs(current_sum - target_sum) < tol) {
                    break;
                }

                if (current_sum > target_sum) {
                    low = beta; // Sum is too large -> beta is too small
                } else {
                    high = beta; // Sum is too small -> beta is too large
                }
            }
            max_beta_achieved_ = std::max(max_beta_achieved_, beta);

            return beta;
        }

        Tensor compute_consensus_weights(
            int step,
            const Tensor &costs,
            const Tensor &particles,
            const Tensor &prev_consensus,
            double prev_energy,
            HyperParameters &params
        ) {
            std::vector<float> cpu_shifted_costs = math::to_float_vector(costs);
            // Find exact beta
            params.beta = solve_for_target_beta(cpu_shifted_costs, 0.5); // 0.8

            // Recompute weights with exact beta
            Tensor neg_beta = math::multiply(costs, Tensor(-params.beta, DType::Float32));
            Tensor weights = math::exp(neg_beta);
            return weights;
        }

        void adapt(int step,
                   const Tensor &particles,
                   const Tensor &current_consensus,
                   const Tensor &prev_consensus,
                   double current_energy,
                   double prev_energy,
                   const Tensor &weights,
                   core::HyperParameters &params) override {
            //params.beta = beta_growth_rate_ * params.beta;
            if (previous_energy_ < current_energy) {
                wait_ += 1;

                if (wait_ > patience_) {
                    float N = particles.shape()[0];
                    float d = particles.shape()[1];
                    float growth_rate = 1.0 + std::min(relative_contraction_rate_ * (2.0 * std::log(N) / (d * d)), 1.0);
                    //
                    //float growth_rate = 1.1;

                    params.lambda = params.lambda * growth_rate;
                    wait_ = 0;
                }
            }

            previous_energy_ = current_energy;
        }

        void reset() override {
            // Reset state for a new optimization run
            previous_energy_ = 0.0;
            wait_ = 0;
        }

        bool ready_to_converge(core::HyperParameters params) override {
            // High beta is a strong indicator that the system has funneled deep into the minimum
            return params.beta > -1;
        }
    };
} // namespace involute::solvers
