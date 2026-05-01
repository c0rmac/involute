#include "involute/solvers/adapters/cma_es_parameter_adapter.hpp"
#include "involute/core/math.hpp"
#include <cmath>
#include <algorithm>
#include <vector>
#include <iostream>

namespace involute::solvers {

    using namespace involute::core;

    CMAESParameterAdapter::CMAESParameterAdapter(double max_consensus_weight, double c_sigma, double d_sigma)
        : max_consensus_weight_(max_consensus_weight),
          c_sigma_(c_sigma),
          d_sigma_(d_sigma),
          max_beta_achieved_(0)
    {
        max_attempts_ = 1;
        is_initialized_ = false;
    }

    double CMAESParameterAdapter::solve_for_target_beta(const std::vector<float> &shifted_costs,
                                                       double target_max_weight,
                                                       double tol,
                                                       int max_iter) {
        double target_sum = 1.0 / target_max_weight;

        std::vector<double> valid_costs;
        int num_zeros = 0;
        for (float c : shifted_costs) {
            if (c <= 1e-8f) num_zeros++;
            else valid_costs.push_back(static_cast<double>(c));
        }

        if (valid_costs.empty() || num_zeros >= target_sum) {
            return max_beta_achieved_;
        }

        double min_c = valid_costs[0];
        for (double c : valid_costs) {
            if (c < min_c) min_c = c;
        }

        double required_tail_sum = target_sum - num_zeros;
        double high = -std::log(required_tail_sum / valid_costs.size()) / min_c;
        double low = 0.0;
        double beta = 0.0;

        for (int iter = 0; iter < max_iter; ++iter) {
            beta = low + 0.5 * (high - low);

            double current_sum = num_zeros;
            for (double c : valid_costs) current_sum += std::exp(-beta * c);

            if (std::abs(current_sum - target_sum) < tol) break;
            if (current_sum > target_sum) low = beta;
            else high = beta;
        }

        max_beta_achieved_ = std::max(max_beta_achieved_, beta);
        return beta;
    }

    Tensor CMAESParameterAdapter::compute_consensus_weights(
        int step,
        const Tensor &costs,
        const std::vector<Tensor> &particles,
        const std::vector<Tensor> &prev_consensus,
        double prev_energy,
        HyperParameters &params)
    {
        std::vector<float> cpu_shifted_costs = math::to_float_vector(costs);
        params.beta = solve_for_target_beta(cpu_shifted_costs, max_consensus_weight_);

        Tensor neg_beta = math::multiply(costs, Tensor(static_cast<float>(-params.beta), DType::Float32));
        return math::exp(neg_beta);
    }

    void CMAESParameterAdapter::adapt(
        int step,
        const std::vector<Tensor> &particles,
        const std::vector<Tensor> &current_consensus,
        const std::vector<Tensor> &prev_consensus,
        double current_energy,
        double prev_energy,
        const Tensor &weights,
        HyperParameters &params)
    {
        // 1. Guard: Evolution paths require velocity (difference between two points).
        if (step < 2) return;

        const size_t num_components = current_consensus.size();

        // 2. Compute Effective Sample Size (mu_eff)
        // This reflects the diversity of the swarm and influences the adaptation rates.
        Tensor sum_w = math::sum(weights);
        Tensor sum_w_sq = math::sum(math::square(weights));
        double mu_eff = std::pow(math::to_double(sum_w), 2) / (math::to_double(sum_w_sq) + 1e-9);

        // 3. Initialize State (Evolution Paths and Expected Norms)
        if (!is_initialized_) {
            p_sigma_vec_.resize(num_components);
            expected_norm_chi_vec_.resize(num_components);

            for (size_t i = 0; i < num_components; ++i) {
                const Tensor& curr_c = current_consensus[i];
                int n = curr_c.shape()[0];
                int k = curr_c.shape()[1];

                // Intrinsic dimension D for SO(n) or Stiefel V(n,k)
                double D = (n == k) ? (n * (n - 1) / 2.0) : (n * k - k * (k + 1) / 2.0);

                // E[||chi||] threshold for noise level comparison
                expected_norm_chi_vec_[i] = std::exp(0.5 * std::log(2.0) + std::lgamma((D + 1.0) / 2.0) - std::lgamma(D / 2.0));

                // Initialize path as a zero matrix
                p_sigma_vec_[i] = math::subtract(curr_c, curr_c);
            }
            is_initialized_ = true;
        }

        // 4. Component-wise Adaptation Loop
        const std::vector<int> t_axes = {1, 0};

        for (size_t i = 0; i < num_components; ++i) {
            const Tensor& curr_c = current_consensus[i];
            const Tensor& prev_c = prev_consensus[i];

            // 4.1. Identify Intrinsic Dimension
            int n = curr_c.shape()[0];
            int k = curr_c.shape()[1];
            double D = (n == k) ? (n * (n - 1) / 2.0) : (n * k - k * (k + 1) / 2.0);

            // 4.2. HEURISTIC PARAMETER CALCULATION
            double c_sig = (mu_eff + 2.0) / (D + mu_eff + 5.0);
            double d_sig = 1.0 + 2.0 * std::max(0.0, std::sqrt((mu_eff - 1.0) / (D + 1.0)) - 1.0) + c_sig;

            // ---------------------------------------------------------
            // 4.3. DYNAMIC TRANSPORT & TANGENT STEP
            // ---------------------------------------------------------
            Tensor delta_m;
            Tensor p_sigma_transported;

            if (n == k) {
                // SO(n) Logic: Lie Algebra & Adjoint Transport
                Tensor rel_rot = math::matmul(math::transpose(prev_c, t_axes), curr_c);
                delta_m = math::matrix_log(rel_rot);
                Tensor rel_rot_T = math::transpose(rel_rot, t_axes);
                p_sigma_transported = math::matmul(rel_rot_T, math::matmul(p_sigma_vec_[i], rel_rot));
            } else {
                // Stiefel V(n,k) Logic: Ambient Velocity & Tangent Projection
                Tensor ambient_v = math::subtract(curr_c, prev_c);

                // Helper lambda for projecting an ambient matrix V onto T_X V(n,k)
                // P_X(V) = V - X * sym(X^T * V)
                auto project_tangent = [&](const Tensor& X, const Tensor& V) {
                    Tensor Xt_V = math::matmul(math::transpose(X, t_axes), V);
                    Tensor Vt_X = math::transpose(Xt_V, t_axes);
                    Tensor sym = math::multiply(math::add(Xt_V, Vt_X), Tensor(0.5f, DType::Float32));
                    return math::subtract(V, math::matmul(X, sym));
                };

                delta_m = project_tangent(curr_c, ambient_v);

                // Parallel transport by projecting the old path onto the new tangent space
                p_sigma_transported = project_tangent(curr_c, p_sigma_vec_[i]);
            }

            // 4.4. Evolution Path Update
            double delta_val = params.delta_at(static_cast<int>(i));
            double lambda_val = params.lambda_at(static_cast<int>(i));
            double true_sigma = delta_val / (std::sqrt(lambda_val) + 1e-9);

            double step_scale = std::sqrt(c_sig * (2.0 - c_sig) * mu_eff) / (true_sigma + 1e-9);

            Tensor term1 = math::multiply(p_sigma_transported, Tensor(static_cast<float>(1.0 - c_sig), DType::Float32));
            Tensor term2 = math::multiply(delta_m, Tensor(static_cast<float>(step_scale), DType::Float32));
            p_sigma_vec_[i] = math::add(term1, term2);

            // 4.5. Step-Size Adaptation
            double p_sigma_norm = math::to_double(math::norm(p_sigma_vec_[i])) / std::sqrt(2.0);

            // Adjust delta based on the ratio of path length to expected random-walk length.
            double exponent = (c_sig / d_sig) * ((p_sigma_norm / expected_norm_chi_vec_[i]) - 1.0);

            // Limit change to a factor of e^0.5 (~1.6x) per step for stability.
            auto next_delta = params.delta_at(static_cast<int>(i)) * std::exp(std::min(exponent, 0.5));
            params.delta_at(static_cast<int>(i)) = std::max(1e-8, next_delta);
        }
    }

    void CMAESParameterAdapter::reset() {
        is_initialized_ = false;
        p_sigma_vec_.clear();
        expected_norm_chi_vec_.clear();
    }

    bool CMAESParameterAdapter::ready_to_converge(HyperParameters params) {
        return true;
    }

    std::shared_ptr<ParameterAdapter> CMAESParameterAdapter::clone() const {
        return std::make_shared<CMAESParameterAdapter>(max_consensus_weight_, c_sigma_, d_sigma_);
    }

} // namespace involute::solvers
