#pragma once

#include "involute/core/parameter_adapter.hpp"
#include "involute/core/hyper_parameters.hpp"
#include <memory>
#include <vector>

namespace involute::solvers {

    using namespace involute::core;

    /**
     * @brief A lightweight parameter adapter for single-manifold SO(d) solvers.
     *
     * Applies two mechanisms per step:
     *  - Beta scheduling: the consensus inverse-temperature oscillates on a
     *    triangular wave between min_consensus_weight_ and max_consensus_weight_
     *    over a period of 2R steps, so the swarm alternates between exploration
     *    and exploitation.
     *  - Noise annealing: delta is multiplied by decay_rate_ each step.
     */
    class SOAdamParameterAdapter : public ParameterAdapter {
    public:
        /**
         * @param max_consensus_weight  Peak target weight for the best particle (0, 1].
         * @param min_consensus_weight  Trough target weight for the best particle (0, 1].
         * @param decay_rate            Geometric multiplier applied to delta each step.
         */
        explicit SOAdamParameterAdapter(double max_consensus_weight = 0.8,
                                        double min_consensus_weight = 0.3,
                                        double decay_rate           = 0.999);

        ~SOAdamParameterAdapter() override = default;

        /**
         * @brief Solves for the inverse temperature beta that achieves a target
         *        maximum softmax weight via bisection.
         */
        double solve_for_target_beta(const std::vector<float> &shifted_costs,
                                     double target_max_weight,
                                     double tol      = 1e-7,
                                     int    max_iter = 100);

        Tensor compute_consensus_weights(
            int                         step,
            const Tensor               &costs,
            const std::vector<Tensor>  &particles,
            const std::vector<Tensor>  &prev_consensus,
            double                      prev_energy,
            HyperParameters            &params) override;

        void adapt(
            int                         step,
            const std::vector<Tensor>  &particles,
            const std::vector<Tensor>  &current_consensus,
            const std::vector<Tensor>  &prev_consensus,
            double                      current_energy,
            double                      prev_energy,
            const Tensor               &weights,
            HyperParameters            &params) override;

        void reset() override;

        bool ready_to_converge(HyperParameters params) override;

        std::shared_ptr<ParameterAdapter> clone() const override;

    private:
        double max_consensus_weight_;
        double min_consensus_weight_;
        double decay_rate_;
    };

} // namespace involute::solvers
