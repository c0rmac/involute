#pragma once

#include "involute/core/parameter_adapter.hpp"
#include "involute/core/parameter_adapter.hpp"
#include "involute/core/math.hpp"
#include <cmath>
#include <algorithm>

using namespace involute::core;

namespace involute::solvers {
    class AdamParameterAdapter : public core::ParameterAdapter {
    private:
        double max_consensus_weight_;
        double beta1_;
        double beta2_;
        double epsilon_;
        double lr_lambda_; // How aggressively to adjust lambda
        double target_ratio_; // The "straightness" threshold (typically 0.4 - 0.6)

        Tensor m_t_;
        Tensor v_t_;
        int t_;

        int initial_patience = 1;

    public:
        explicit AdamParameterAdapter(double max_consensus_weight = 0.8,
                                        double beta1 = 0.9,
                                      double beta2 = 0.999,
                                      double epsilon = 1e-8,
                                      double lr_lambda = 0.2,
                                      double target_ratio = 0.5)
            : max_consensus_weight_(max_consensus_weight), beta1_(beta1), beta2_(beta2), epsilon_(epsilon),
              lr_lambda_(lr_lambda), target_ratio_(target_ratio), t_(0) {
            max_attempts_ = 1;
        }


        double max_beta_achieved_ = 0;

        /**
         * @brief Finds the optimal beta such that the maximum Boltzmann weight equals target_max_weight.
         * Uses the Newton-Raphson method for fast root finding.
         * * @param shifted_costs std::vector of (E_i - E_min) for all particles.
         * @param initial_beta The starting guess for beta.
         * @param target_max_weight The desired maximum weight (e.g., 0.95).
         * @param tol Convergence tolerance.
         * @param max_iter Maximum Newton-Raphson iterations.
         * @return The adjusted beta.
         */
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
            params.beta = solve_for_target_beta(cpu_shifted_costs, max_consensus_weight_); // 0.8

            // Recompute weights with exact beta
            Tensor neg_beta = math::multiply(costs, Tensor(-params.beta, DType::Float32));
            Tensor weights = math::exp(neg_beta);
            return weights;
        }

        void adapt(
            int step,
            const Tensor &particles,
            const Tensor &current_consensus,
            const Tensor &prev_consensus,
            double current_energy,
            double prev_energy,
            const Tensor &weights,
            core::HyperParameters &params
        ) override {
            // Skip step 0 since we have no previous consensus to measure movement against
            if (step == 0) {
                return;
            }

            if (step < initial_patience) {
                return;
            }

            t_++;

            // 1. Calculate the implicit "Gradient" G_t
            // In CBO, the movement of the consensus point represents the steepest descent direction
            Tensor g_t = math::subtract(current_consensus, prev_consensus);

            if (t_ == 1) {
                // Initialize moments as zero tensors of the exact same shape/device
                m_t_ = math::multiply(g_t, Tensor(0.0, DType::Float32));
                v_t_ = math::multiply(g_t, Tensor(0.0, DType::Float32));
            }

            // 2. Update First Moment M_t (Momentum)
            Tensor m_t_prev_scaled = math::multiply(m_t_, Tensor(beta1_, DType::Float32));
            Tensor g_t_scaled = math::multiply(g_t, Tensor(1.0 - beta1_, DType::Float32));
            m_t_ = math::add(m_t_prev_scaled, g_t_scaled);

            // 3. Update Second Moment V_t (Uncentered Variance)
            Tensor g_t_sq = math::square(g_t);
            Tensor v_t_prev_scaled = math::multiply(v_t_, Tensor(beta2_, DType::Float32));
            Tensor g_t_sq_scaled = math::multiply(g_t_sq, Tensor(1.0 - beta2_, DType::Float32));
            v_t_ = math::add(v_t_prev_scaled, g_t_sq_scaled);

            // 4. Bias Correction (Crucial for the early steps)
            double bias_correction1 = 1.0 - std::pow(beta1_, t_);
            double bias_correction2 = 1.0 - std::pow(beta2_, t_);

            // 5. Calculate Global Alignment Ratio (The Second-Order Test)
            // We aggregate the tensor moments into a scalar norm to adjust the global lambda.
            Tensor m_t_sq = math::square(m_t_);
            double m_norm = std::sqrt(math::to_double(math::sum(m_t_sq)));
            double v_norm = math::to_double(math::sum(v_t_)); // Sum of squared gradients

            double m_hat = m_norm / bias_correction1;
            double v_hat = v_norm / bias_correction2;

            // Ratio ranges from ~0 (pure zigzag) to ~1 (perfectly straight path)
            double ratio = 0.0;
            if (v_hat > 0) {
                ratio = m_hat / (std::sqrt(v_hat) + epsilon_);
            }

            // 6. Step Size Control (Adapt Lambda)
            // If ratio < target_ratio (zigzag), (target - ratio) is positive -> lambda increases -> step size shrinks
            // If ratio > target_ratio (straight), (target - ratio) is negative -> lambda decreases -> step size grows
            params.lambda *= std::exp(lr_lambda_ * (target_ratio_ - ratio));

            // Optional: Clamp lambda to prevent numerical blowup or stagnation
            params.lambda = std::max(1e-4, std::min(params.lambda, 1e15));
        }

        void reset() override {
            t_ = 0;
            // Tensors m_t_ and v_t_ will naturally be overwritten dynamically on step 1
        }

        bool ready_to_converge(core::HyperParameters params) override {
            // You could theoretically mandate that convergence is only allowed
            // if lambda is sufficiently high (meaning the algorithm confirms it is in a steep minima).
            return true;
        }
    };
} // namespace involute::core
