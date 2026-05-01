#include "involute/solvers/adapters/adam_parameter_adapter.hpp"
#include "involute/core/math.hpp"
#include <cmath>
#include <algorithm>
#include <limits>
#include <vector>

namespace involute::solvers {

    using namespace involute::core;

    AdamParameterAdapter::AdamParameterAdapter(double max_consensus_weight,
                                               double beta1,
                                               double beta2,
                                               double epsilon,
                                               double lr_lambda,
                                               double target_ratio)
        : max_consensus_weight_(max_consensus_weight),
          beta1_(beta1), beta2_(beta2), epsilon_(epsilon),
          lr_lambda_(lr_lambda), target_ratio_(target_ratio),
          t_(0), initial_patience(1), max_beta_achieved_(0)
    {
        max_attempts_ = 1;
    }

    double AdamParameterAdapter::solve_for_target_beta(const std::vector<float> &shifted_costs,
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

    Tensor AdamParameterAdapter::compute_consensus_weights(
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

    void AdamParameterAdapter::adapt(int step,
                                     const std::vector<Tensor> &particles,
                                     const std::vector<Tensor> &current_consensus,
                                     const std::vector<Tensor> &prev_consensus,
                                     double current_energy,
                                     double prev_energy,
                                     const Tensor &weights,
                                     HyperParameters &params)
    {
        // Adaptation starts after the initial patience period and only if energy improves or stays stable
        if (step == 0 || step < initial_patience || prev_energy > current_energy) return;

        t_++;

        const size_t num_components = current_consensus.size();

        // Initialize momentum and variance state vectors per component on the first adaptation step
        if (t_ == 1) {
            m_t_vec_.resize(num_components);
            v_t_vec_.resize(num_components);
            for (size_t i = 0; i < num_components; ++i) {
                // Initialize to zero matrices matching the shape and device of each component
                m_t_vec_[i] = math::multiply(current_consensus[i], Tensor(0.0f, DType::Float32));
                v_t_vec_[i] = math::multiply(current_consensus[i], Tensor(0.0f, DType::Float32));
            }
        }

        const double bias_correction1 = 1.0 - std::pow(beta1_, t_);
        const double bias_correction2 = 1.0 - std::pow(beta2_, t_);

        // Independently update parameters for each manifold component in the product space
        for (size_t i = 0; i < num_components; ++i) {
            const Tensor& curr_c = current_consensus[i];
            const Tensor& prev_c = prev_consensus[i];

            // Compute the consensus velocity as a gradient signal
            Tensor g_t = math::subtract(curr_c, prev_c);

            // Update first moment (momentum)
            m_t_vec_[i] = math::add(
                math::multiply(m_t_vec_[i], Tensor(static_cast<float>(beta1_), DType::Float32)),
                math::multiply(g_t, Tensor(static_cast<float>(1.0 - beta1_), DType::Float32))
            );

            // Update second moment (uncentered variance)
            Tensor g_t_sq = math::square(g_t);
            v_t_vec_[i] = math::add(
                math::multiply(v_t_vec_[i], Tensor(static_cast<float>(beta2_), DType::Float32)),
                math::multiply(g_t_sq, Tensor(static_cast<float>(1.0 - beta2_), DType::Float32))
            );

            // Calculate Adam metrics
            const double m_norm = std::sqrt(math::to_double(math::sum(math::square(m_t_vec_[i]))));
            const double v_norm = math::to_double(math::sum(v_t_vec_[i]));

            const double m_hat = m_norm / bias_correction1;
            const double v_hat = v_norm / bias_correction2;

            // Compute signal-to-noise ratio to adjust step size (lambda)
            const double ratio = (v_hat > 0) ? m_hat / (std::sqrt(v_hat) + epsilon_) : 0.0;

            // Update the specific lambda for this manifold component
            params.lambda_at(static_cast<int>(i)) *= std::exp(lr_lambda_ * (target_ratio_ - ratio));
            params.lambda_at(static_cast<int>(i)) = std::max(1e-4, std::min(params.lambda_at(static_cast<int>(i)), 1e10));
        }
    }

    void AdamParameterAdapter::reset() {
        t_ = 0;
        m_t_vec_.clear();
        v_t_vec_.clear();
    }

    bool AdamParameterAdapter::ready_to_converge(HyperParameters params) {
        return true;
    }

    std::shared_ptr<ParameterAdapter> AdamParameterAdapter::clone() const {
        return std::make_shared<AdamParameterAdapter>(
            max_consensus_weight_, beta1_, beta2_, epsilon_, lr_lambda_, target_ratio_);
    }

} // namespace involute::solvers
