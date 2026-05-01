#pragma once
#include "so_isotropic_solver.hpp"
#include "involute/solvers/adapters/cma_es_parameter_adapter.hpp"

namespace involute::solvers {

struct SOIsotropicSolverCMAESConfig {
    int N;
    int d;
    std::shared_ptr<core::ConvergenceCriterion>  convergence;
    double                       lambda = 1.0;
    double                       delta  = 0.0;
    std::vector<core::Debugger>  debug  = {};
};

class SOIsotropicSolverCMAES : public SOIsotropicSolver {
public:
    explicit SOIsotropicSolverCMAES(SOIsotropicSolverCMAESConfig config);
};

} // namespace involute::solvers
