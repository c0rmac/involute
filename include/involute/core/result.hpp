/**
 * @file result.hpp
 * @brief Defines the output structure for the Involute solvers.
 * * This file contains the CBOResult struct, which encapsulates the final
 * mathematical state of the swarm after the optimization loop terminates.
 * It ensures that the heavy GPU tensors and the scalar CPU metrics (like
 * final energy) are bundled together cleanly.
 */

#pragma once

#include "involute/core/tensor.hpp"

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
     * * @example
     * // Running an experiment
     * involute::solvers::SODHoppingSolver solver(config);
     * involute::core::CBOResult result = solver.solve(&my_objective);
     * * // Evaluating the outcome
     * if (result.converged) {
     * std::cout << "Success! Minimum energy found: " << result.min_energy << "\n";
     * * // Extract the first matrix from the swarm to use as the final answer
     * // (Assuming a function to slice the tensor exists in your math DSL)
     * involute::Tensor best_matrix = involute::math::slice(result.final_X, 0);
     * }
     */
    struct CBOResult {
        /** * @brief The final tensor of the particle swarm.
         * * For matrix manifolds like SO(d) or Stiefel, this will typically be a
         * 3D Tensor of shape [N, d, d]. It resides in the backend's memory
         * (MLX array or SYCL buffer) and represents the exact state of all N
         * particles at the moment the solver halted.
         */
        Tensor final_particles;

        Tensor final_consensus;


        /** * @brief The lowest energy (cost) achieved by the swarm.
         * * This is stored as a standard C++ double. It has already been pulled
         * across the GPU-CPU bridge via math::to_double(), so reading this value
         * incurs zero synchronization overhead.
         */
        double min_energy{0.0};

        /** * @brief Indicates whether the solver met a convergence criterion.
         * * If the solver halted because it hit a tolerance threshold (like
         * EnergyToleranceCriterion), this is true. If it simply ran out of
         * allowed steps (MaxStepsCriterion), you might treat the result with
         * more skepticism.
         */
        bool converged{false};

        /**
         * @brief The total number of iterations executed.
         * * Useful for benchmarking how changes to the lambda/delta hyperparameters
         * affect the speed of convergence.
         */
        int iterations_run{0};

        /**
         * @brief A step-by-step history of the optimization loop.
         * Only populated if the solver's debug flag is set to true.
         */
        std::vector<StepRecord> history;
    };
} // namespace involute::core
