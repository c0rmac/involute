/**
 * @file result.hpp
 * @brief Defines the output structure for the Involute solvers.
 * * Refactored to support Product Manifolds: final_particles and
 * final_consensus are now vectors of Tensors to match the BaseSolver interface.
 */

#pragma once

#include "involute/core/tensor.hpp"
#include <vector>

namespace involute::core {
    /**
     * @struct StepRecord
     * @brief Records the hyperparameters and energy at a specific iteration.
     * Used for debugging and analyzing convergence behavior.
     */
    struct StepRecord {
        int step;
        double energy;
        double beta;
        double lambda;
        double delta;
    };

    /**
     * @struct CBOResult
     * @brief Encapsulates the final state of a Consensus-Based Optimization run.
     * * When BaseSolver::solve() completes, it packages the final particle cloud
     * and convergence metrics into this struct.
     */
    struct CBOResult {
        /** * @brief The final state of the particle swarm.
         * Each element in the vector represents a component of the product manifold.
         * For example, in an SO(d) solver, this contains one Tensor of shape [N, d, d].
         */
        std::vector<Tensor> final_particles;

        /**
         * @brief The final consensus point (the Fréchet mean of the swarm).
         * Represented as a vector of Tensors corresponding to each manifold component.
         */
        std::vector<Tensor> final_consensus;

        /** * @brief The lowest energy (cost) achieved by the swarm.
         * Stored as a double, typically extracted via math::to_double().
         */
        double min_energy{0.0};

        /** * @brief Indicates whether the solver met a convergence criterion.
         * True if the loop terminated via a tolerance check (e.g., Energy or Consensus).
         */
        bool converged{false};

        /**
         * @brief The total number of iterations executed.
         */
        int iterations_run{0};

        /**
         * @brief A step-by-step history of the optimization loop.
         * Only populated if Debugger::History is enabled in SolverConfig.
         */
        std::vector<StepRecord> history;
    };
} // namespace involute::core