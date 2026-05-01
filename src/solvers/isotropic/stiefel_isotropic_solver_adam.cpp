#include "involute/solvers/isotropic/stiefel_isotropic_solver_adam.hpp"
#include <cmath>

namespace involute::solvers {
using namespace involute::core;

StiefelIsotropicSolverADAM::StiefelIsotropicSolverADAM(StiefelIsotropicSolverADAMConfig cfg)
    : StiefelIsotropicSolver(StiefelIsotropicSolverConfig{
        .N          = cfg.N,
        .n          = cfg.n,
        .k          = cfg.k,
        .convergence = cfg.convergence,
        .adapter    = std::make_shared<AdamParameterAdapter>(
                          0.8, 0.9, 0.999, 1e-8,
                          std::log(static_cast<double>(cfg.N)) /
                          (cfg.n * cfg.k - cfg.k * (cfg.k + 1) / 2)),
        .lambda     = cfg.lambda,
        .delta      = cfg.delta,
        .debug      = cfg.debug
    })
{}

} // namespace involute::solvers
