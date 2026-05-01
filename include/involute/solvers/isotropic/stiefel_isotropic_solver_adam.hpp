#pragma once
#include "stiefel_isotropic_solver.hpp"
#include "involute/solvers/adapters/adam_parameter_adapter.hpp"

namespace involute::solvers {

struct StiefelIsotropicSolverADAMConfig {
    int N;
    int n;
    int k;
    std::shared_ptr<core::ConvergenceCriterion>  convergence;
    double                       lambda = 1.0;
    double                       delta  = 0.0;
    std::vector<core::Debugger>  debug  = {};
};

class StiefelIsotropicSolverADAM : public StiefelIsotropicSolver {
public:
    explicit StiefelIsotropicSolverADAM(StiefelIsotropicSolverADAMConfig config);
};

} // namespace involute::solvers
