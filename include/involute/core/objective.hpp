/**
 * @file objective.hpp
 * @brief Interfaces for defining energy landscapes and cost functions.
 * * In the Consensus-Based Optimization (CBO) framework, the "Objective"
 * represents the energy landscape the swarm is attempting to minimize.
 * To maintain high throughput, this interface enforces batched evaluation,
 * allowing the backend to parallelize the cost calculation for all particles
 * simultaneously on the GPU/AMX.
 */

#pragma once

#include "involute/core/tensor.hpp"
#include "involute/core/math.hpp"
#include <functional>
#include <utility>

namespace involute {
    namespace core {
        /**
         * @class ObjectiveFunction
         * @brief Abstract base class for all optimization problems.
         * * To define a new problem, inherit from this class. All calculations within
         * evaluate_batch() should use the 'involute::math' DSL to remain
         * hardware-agnostic.
         * * @example
         * class MaxCutObjective : public ObjectiveFunction {
         * Tensor L; // Laplacian matrix
         * public:
         * Tensor evaluate_batch(const Tensor& X) const override {
         * // Trace(X^T * L * X) for a batch of particles
         * auto LX = math::matmul(L, X);
         * return math::sum(math::multiply(X, LX), {1, 2});
         * }
         * };
         */
        class ObjectiveFunction {
        public:
            virtual ~ObjectiveFunction() = default;

            /**
             * @brief Computes the cost for every particle in the swarm in parallel.
             * * @param particles A Tensor of shape [N, d, d] representing the swarm state.
             * @return A 1D Tensor of shape [N] where the i-th element is the cost of
             * the i-th particle.
             */
            virtual Tensor evaluate_batch(const Tensor &particles) const = 0;
        };

        // ==============================================================================
        // FUNCTIONAL OBJECTIVE WRAPPER
        // ==============================================================================

        /**
         * @class FunctionalObjective
         * @brief A sleek wrapper for using lambdas or free functions as objectives.
         * * Ideal for quick experiments and one-off cost functions.
         * * @example
         * // Minimizing the distance to an identity matrix
         * auto identity_cost = [](const Tensor& X) {
         * auto I = math::eye(X.shape(-1));
         * auto diff = math::subtract(X, I);
         * return math::sum(math::square(diff), {1, 2});
         * };
         * FunctionalObjective obj(identity_cost);
         */
        class FunctionalObjective : public ObjectiveFunction {
        private:
            std::function<Tensor(const Tensor &)> eval_func_;

        public:
            /**
             * @brief Constructs an objective from a lambda or function pointer.
             * @param func A callable taking a [N, d, d] Tensor and returning a [N] Tensor.
             */
            explicit FunctionalObjective(std::function<Tensor(const Tensor &)> func)
                : eval_func_(std::move(func)) {
            }

            /**
             * @brief Executes the wrapped function on the GPU/AMX backend.
             */
            Tensor evaluate_batch(const Tensor &particles) const override {
                return eval_func_(particles);
            }
        };
    } // namespace core

    /**
     * @brief Alias for quicker experiment writing.
     * Allows using 'involute::FuncObj' instead of the full namespace path.
     */
    using FuncObj = core::FunctionalObjective;
} // namespace involute
