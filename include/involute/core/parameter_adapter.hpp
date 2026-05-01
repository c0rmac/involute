#pragma once

#include "involute/core/tensor.hpp"
#include "involute/core/hyper_parameters.hpp"
#include <memory>
#include <vector>

namespace involute::core {

    /**
     * @brief Interface for dynamically adapting CBO hyper-parameters.
     * * Refactored to support Product Manifolds: Consensus points and particle
     * states are passed as vectors of Tensors to avoid dimension-matching
     * issues when components have different shapes.
     */
    class ParameterAdapter {
    public:
        int max_attempts_ = 1;

        virtual ~ParameterAdapter() = default;

        /**
         * @brief Computes the weights for consensus calculation.
         * @param particles Vector of tensors [N, ni, ki] for each manifold component.
         * @param prev_consensus Vector of tensors [ni, ki] from the previous step.
         */
        virtual Tensor compute_consensus_weights(
            int step,
            const Tensor &costs,
            const std::vector<Tensor> &particles,
            const std::vector<Tensor> &prev_consensus,
            double prev_energy,
            HyperParameters &params
        ) = 0;

        /**
         * @brief Adapts lambda and delta parameters.
         * @param particles Vector of tensors [N, ni, ki].
         * @param current_consensus Vector of tensors [ni, ki] for the current step.
         * @param prev_consensus Vector of tensors [ni, ki] from the previous step.
         */
        virtual void adapt(
            int step,
            const std::vector<Tensor> &particles,
            const std::vector<Tensor> &current_consensus,
            const std::vector<Tensor> &prev_consensus,
            double current_energy,
            double prev_energy,
            const Tensor &weights,
            HyperParameters &params
        ) = 0;

        /**
         * @brief Resets internal state (e.g., Adam momentum or CMA-ES paths).
         */
        virtual void reset() = 0;

        /**
         * @brief Determines if the adapter logic is stable enough to allow convergence.
         */
        virtual bool ready_to_converge(HyperParameters params) = 0;

        /**
         * @brief Creates a deep copy of the adapter.
         */
        virtual std::shared_ptr<ParameterAdapter> clone() const = 0;
    };

} // namespace involute::core