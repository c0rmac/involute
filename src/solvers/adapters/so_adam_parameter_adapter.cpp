#include "involute/solvers/adapters/so_adam_parameter_adapter.hpp"
#include "involute/core/math.hpp"
#include <algorithm>
#include <cmath>

namespace involute::solvers {

    SOAdamParameterAdapter::SOAdamParameterAdapter(double max_consensus_weight,
                                                   double min_consensus_weight,
                                                   double decay_rate)
        : max_consensus_weight_(max_consensus_weight)
        , min_consensus_weight_(min_consensus_weight)
        , decay_rate_(decay_rate)
    {
        max_attempts_ = 1;
    }

    double SOAdamParameterAdapter::solve_for_target_beta(
        const std::vector<float> &shifted_costs,
        double target_max_weight,
        double tol,
        int max_iter)
    {
        double target_sum = 1.0 / target_max_weight;

        std::vector<double> valid_costs;
        int num_zeros = 0;
        for (float c : shifted_costs) {
            if (c <= 1e-8f) num_zeros++;
            else valid_costs.push_back(static_cast<double>(c));
        }

        if (valid_costs.empty() || num_zeros >= target_sum) return 0.0;

        double min_c = valid_costs[0];
        for (double c : valid_costs)
            if (c < min_c) min_c = c;

        double required_tail_sum = target_sum - num_zeros;
        double high = -std::log(required_tail_sum / valid_costs.size()) / min_c;
        double low  = 0.0;
        double beta = 0.0;

        for (int iter = 0; iter < max_iter; ++iter) {
            beta = low + 0.5 * (high - low);

            double current_sum = num_zeros;
            for (double c : valid_costs) current_sum += std::exp(-beta * c);

            if (std::abs(current_sum - target_sum) < tol) break;
            if (current_sum > target_sum) low  = beta;
            else                          high = beta;
        }

        return beta;
    }

    Tensor SOAdamParameterAdapter::compute_consensus_weights(
        int                        step,
        const Tensor              &costs,
        const std::vector<Tensor> &/*particles*/,
        const std::vector<Tensor> &/*prev_consensus*/,
        double                     /*prev_energy*/,
        HyperParameters           &params)
    {
        std::vector<float> cpu_costs = math::to_float_vector(costs);

        // Triangular-wave oscillation: peaks at max_consensus_weight_ every R steps.
        const int R   = 100;
        double phase  = static_cast<double>(step % (2 * R)) / R;
        double alpha  = (phase <= 1.0) ? phase : (2.0 - phase);

        double current_target = min_consensus_weight_
                              + alpha * (max_consensus_weight_ - min_consensus_weight_);

        params.beta = solve_for_target_beta(cpu_costs, current_target);

        Tensor neg_beta = math::multiply(costs, Tensor(-params.beta, DType::Float32));
        return math::exp(neg_beta);
    }

    void SOAdamParameterAdapter::adapt(
        int                        /*step*/,
        const std::vector<Tensor> &/*particles*/,
        const std::vector<Tensor> &/*current_consensus*/,
        const std::vector<Tensor> &/*prev_consensus*/,
        double                     /*current_energy*/,
        double                     /*prev_energy*/,
        const Tensor              &/*weights*/,
        HyperParameters           &params)
    {
        // Geometric noise annealing: shrink delta each step.
        params.delta_at(0) *= decay_rate_;
    }

    void SOAdamParameterAdapter::reset() {
        // No stateful moments to reset.
    }

    bool SOAdamParameterAdapter::ready_to_converge(HyperParameters /*params*/) {
        return true;
    }

    std::shared_ptr<ParameterAdapter> SOAdamParameterAdapter::clone() const {
        return std::make_shared<SOAdamParameterAdapter>(
            max_consensus_weight_, min_consensus_weight_, decay_rate_);
    }

} // namespace involute::solvers
