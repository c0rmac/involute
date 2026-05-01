#pragma once
#include "so_isotropic_solver.hpp"
#include "involute/solvers/adapters/adam_parameter_adapter.hpp"

namespace involute::solvers {

struct SOIsotropicSolverADAMConfig {
    int N;
    int d;
    std::shared_ptr<core::ConvergenceCriterion>  convergence;
    double                       lambda = 1.0;
    double                       delta  = 0.0;   // 0.0 → diameter_so(d)
    std::vector<core::Debugger>  debug  = {};
};

class SOIsotropicSolverADAM : public SOIsotropicSolver {
public:
    explicit SOIsotropicSolverADAM(SOIsotropicSolverADAMConfig config);
};

} // namespace involute::solvers
