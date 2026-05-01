#include "involute/solvers/isotropic/product_isotropic_solver_cmaes.hpp"

namespace involute::solvers {

ProductIsotropicSolverCMAES::ProductIsotropicSolverCMAES(
    ProductIsotropicSolverCMAESConfig cfg)
    : ProductIsotropicSolver(ProductIsotropicSolverConfig{
          .manifolds   = std::move(cfg.manifolds),
          .N           = cfg.N,
          .dtype       = cfg.dtype,
          .convergence = std::move(cfg.convergence),
          .adapter     = std::make_shared<CMAESParameterAdapter>(0.5, 0.3, 1.0),
          .debug       = std::move(cfg.debug)
      })
{}

} // namespace involute::solvers
