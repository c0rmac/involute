#pragma once

#include "involute/solvers/isotropic/product_isotropic_solver.hpp"
#include "involute/solvers/adapters/cma_es_parameter_adapter.hpp"

#include <memory>
#include <vector>

namespace involute::solvers {

using namespace involute::core;

// =============================================================================
// ProductIsotropicSolverCMAESConfig
// =============================================================================

/**
 * @brief Minimal config for ProductIsotropicSolverCMAES.
 *
 * The CMAESParameterAdapter is constructed automatically with default
 * hyper-parameters (initial scale 0.5, contraction 0.3, expansion 1.0).
 */
struct ProductIsotropicSolverCMAESConfig {
    std::vector<ManifoldSpec>             manifolds;
    int                                   N;
    DType                                 dtype       = DType::Float32;
    std::shared_ptr<ConvergenceCriterion> convergence;
    std::vector<Debugger>                 debug       = {};
};

// =============================================================================
// ProductIsotropicSolverCMAES
// =============================================================================

/**
 * @brief ProductIsotropicSolver pre-wired with a CMAESParameterAdapter.
 *
 * Convenience wrapper: specify only the manifolds, particle count, convergence
 * criterion, and debug flags. The adapter is constructed automatically.
 */
class ProductIsotropicSolverCMAES : public ProductIsotropicSolver {
public:
    explicit ProductIsotropicSolverCMAES(ProductIsotropicSolverCMAESConfig config);
};

} // namespace involute::solvers
