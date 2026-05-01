#include "involute/solvers/isotropic/so_isotropic_solver_adam.hpp"
#include <cmath>

namespace involute::solvers {
using namespace involute::core;

SOIsotropicSolverADAM::SOIsotropicSolverADAM(SOIsotropicSolverADAMConfig cfg)
    : SOIsotropicSolver(SOIsotropicSolverConfig{
        .N          = cfg.N,
        .d          = cfg.d,
        .convergence = cfg.convergence,
        .adapter    = std::make_shared<AdamParameterAdapter>(
                          0.8, 0.9, 0.999, 1e-8,
                          std::log(static_cast<double>(cfg.N)) / (cfg.d * cfg.d)),
        .lambda     = cfg.lambda,
        .delta      = cfg.delta,   // 0.0 handled by SOIsotropicSolver::to_core_config
        .debug      = cfg.debug
    })
{}

} // namespace involute::solvers
