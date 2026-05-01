#pragma once

#include "involute/core/tensor.hpp"
#include <functional>
#include <vector>

namespace involute {
    namespace core {
        // ==============================================================================
        // STANDARD OBJECTIVES (Single Manifold)
        // ==============================================================================
        class ObjectiveFunction {
        public:
            virtual ~ObjectiveFunction() = default;
            virtual Tensor evaluate_batch(const Tensor &particles) const = 0;
        };

        class FunctionalObjective : public ObjectiveFunction {
            std::function<Tensor(const Tensor &)> eval_func_;
        public:
            explicit FunctionalObjective(std::function<Tensor(const Tensor &)> func);
            Tensor evaluate_batch(const Tensor &particles) const override;
        };

        // ==============================================================================
        // PRODUCT OBJECTIVES (Multiple Manifolds / Cartesian Products)
        // ==============================================================================
        class ProductObjectiveFunction {
        public:
            virtual ~ProductObjectiveFunction() = default;
            virtual Tensor evaluate_batch(const std::vector<Tensor>& particles) const = 0;
        };

        class FunctionalProductObjective : public ProductObjectiveFunction {
            std::function<Tensor(const std::vector<Tensor>&)> eval_func_;
        public:
            explicit FunctionalProductObjective(std::function<Tensor(const std::vector<Tensor>&)> func);
            Tensor evaluate_batch(const std::vector<Tensor>& particles) const override;
        };
    } // namespace core

    // ==============================================================================
    // GLOBAL ALIASES
    // ==============================================================================
    using FuncObj        = core::FunctionalObjective;
    using FuncProductObj = core::FunctionalProductObjective;

} // namespace involute