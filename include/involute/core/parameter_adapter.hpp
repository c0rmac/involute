#pragma once

#include "involute/core/tensor.hpp"
#include "involute/core/base_solver.hpp"

namespace involute::core {
    // Forward declaration breaks the circular dependency
    struct HyperParameters;

    /**
     * @class ParameterAdapter
     * @brief Interface for strategies that adjust hyperparameters on the fly.
     */
    class ParameterAdapter {
    public:
        int max_attempts_ = 1;

        virtual ~ParameterAdapter() = default;

        /**
         * @brief Evaluates the swarm before a consensus point has been computed and adapts the hyperparameters in-place.
         * * @param step Current iteration index.
         * @param costs The current costs of each particle.
         * @param particles The current batch of particles.
         * @param prev_consensus The Fréchet mean of the previous step.
         * @param prev_energy Best objective value in the previous step.
         * @param params The hyperparameters to be modified for the next step.
         */
        virtual Tensor compute_consensus_weights(
            int step,
            const Tensor &costs,
            const Tensor &particles,
            const Tensor &prev_consensus,
            double prev_energy,
            HyperParameters &params
        ) = 0;

        /**
         * @brief Evaluates the swarm after a consensus point has been computed and adapts the hyperparameters in-place.
         * * @param step Current iteration index.
         * @param particles The current batch of particles.
         * @param current_consensus The Fréchet mean of the current step.
         * @param prev_consensus The Fréchet mean of the previous step.
         * @param current_energy Best objective value in the current step.
         * @param prev_energy Best objective value in the previous step.
         * @param params The hyperparameters to be modified for the next step.
         */
        virtual void adapt(
            int step,
            const Tensor &particles,
            const Tensor &current_consensus,
            const Tensor &prev_consensus,
            double current_energy,
            double prev_energy,
            const Tensor &weights,
            HyperParameters &params
        ) = 0;

        virtual void reset() = 0;

        virtual bool ready_to_converge(HyperParameters params) = 0;
    };
} // namespace involute::core
