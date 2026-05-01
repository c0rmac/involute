#include "involute/core/objective.hpp"

namespace involute::core {

    FunctionalObjective::FunctionalObjective(std::function<Tensor(const Tensor &)> func)
        : eval_func_(std::move(func)) {}

    Tensor FunctionalObjective::evaluate_batch(const Tensor &particles) const {
        return eval_func_(particles);
    }

    FunctionalProductObjective::FunctionalProductObjective(
        std::function<Tensor(const std::vector<Tensor>&)> func)
        : eval_func_(std::move(func)) {}

    Tensor FunctionalProductObjective::evaluate_batch(const std::vector<Tensor>& particles) const {
        return eval_func_(particles);
    }

} // namespace involute::core