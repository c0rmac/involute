#include "involute/solvers/isotropic/stiefel_isotropic_solver_cmaes.hpp"
#include <cmath>

namespace involute::solvers {
using namespace involute::core;

StiefelIsotropicSolverCMAES::StiefelIsotropicSolverCMAES(StiefelIsotropicSolverCMAESConfig cfg)
    : StiefelIsotropicSolver(StiefelIsotropicSolverConfig{
        .N          = cfg.N,
        .n          = cfg.n,
        .k          = cfg.k,
        .convergence = cfg.convergence,
        .adapter    = std::make_shared<CMAESParameterAdapter>(0.8, 0.3, 1.0),
        .lambda     = cfg.lambda,
        .delta      = cfg.delta,
        .debug      = cfg.debug
    })
{}

} // namespace involute::solvers
