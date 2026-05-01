#include "involute/solvers/isotropic/so_isotropic_solver_cmaes.hpp"
#include <cmath>

namespace involute::solvers {
using namespace involute::core;

SOIsotropicSolverCMAES::SOIsotropicSolverCMAES(SOIsotropicSolverCMAESConfig cfg)
    : SOIsotropicSolver(SOIsotropicSolverConfig{
        .N          = cfg.N,
        .d          = cfg.d,
        .convergence = cfg.convergence,
        .adapter    = std::make_shared<CMAESParameterAdapter>(0.8, 0.3, 1.0),
        .lambda     = cfg.lambda,
        .delta      = cfg.delta,
        .debug      = cfg.debug
    })
{}

} // namespace involute::solvers
