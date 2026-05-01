#pragma once
#include "stiefel_isotropic_solver.hpp"
#include "involute/solvers/adapters/cma_es_parameter_adapter.hpp"

namespace involute::solvers {

struct StiefelIsotropicSolverCMAESConfig {
    int N;
    int n;
    int k;
    std::shared_ptr<core::ConvergenceCriterion>  convergence;
    double                       lambda = 1.0;
    double                       delta  = 0.0;
    std::vector<core::Debugger>  debug  = {};
};

class StiefelIsotropicSolverCMAES : public StiefelIsotropicSolver {
public:
    explicit StiefelIsotropicSolverCMAES(StiefelIsotropicSolverCMAESConfig config);
};

} // namespace involute::solvers
