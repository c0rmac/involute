#pragma once

#include "involute/solvers/isotropic/product_isotropic_solver.hpp"
#include "involute/solvers/adapters/adam_parameter_adapter.hpp"

#include <cmath>
#include <memory>
#include <vector>

namespace involute::solvers {

using namespace involute::core;

// =============================================================================
// ProductIsotropicSolverADAMConfig
// =============================================================================

/**
 * @brief Minimal config for ProductIsotropicSolverADAM.
 *
 * The Adam learning rate is derived automatically as log(N) / total_flat_dim,
 * where total_flat_dim = Σ nᵢ·kᵢ across all manifold components.
 * All other Adam hyper-parameters use proven defaults.
 */
struct ProductIsotropicSolverADAMConfig {
    std::vector<ManifoldSpec>             manifolds;
    int                                   N;
    DType                                 dtype       = DType::Float32;
    std::shared_ptr<ConvergenceCriterion> convergence;
    std::vector<Debugger>                 debug       = {};
};

// =============================================================================
// ProductIsotropicSolverADAM
// =============================================================================

/**
 * @brief ProductIsotropicSolver pre-wired with an AdamParameterAdapter.
 *
 * Convenience wrapper: specify only the manifolds, particle count, convergence
 * criterion, and debug flags. The adapter is constructed automatically.
 */
class ProductIsotropicSolverADAM : public ProductIsotropicSolver {
public:
    explicit ProductIsotropicSolverADAM(ProductIsotropicSolverADAMConfig config);
};

} // namespace involute::solvers
