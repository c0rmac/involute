#include "involute/solvers/isotropic/product_isotropic_solver_adam.hpp"

#include <cmath>
#include <numeric>

namespace involute::solvers {

ProductIsotropicSolverADAM::ProductIsotropicSolverADAM(
    ProductIsotropicSolverADAMConfig cfg)
    : ProductIsotropicSolver([&]() -> ProductIsotropicSolverConfig {
        // Compute total flat dimension: Σ nᵢ·kᵢ
        int total_flat_dim = 0;
        for (const auto& spec : cfg.manifolds) {
            int k = (spec.type == ManifoldSpec::Type::SO) ? spec.n : spec.k;
            total_flat_dim += spec.n * k;
        }

        // Learning rate heuristic: log(N) / total_flat_dim
        double lr = std::log(static_cast<double>(cfg.N))
                    / static_cast<double>(std::max(total_flat_dim, 1));

        return ProductIsotropicSolverConfig{
            .manifolds   = std::move(cfg.manifolds),
            .N           = cfg.N,
            .dtype       = cfg.dtype,
            .convergence = std::move(cfg.convergence),
            .adapter     = std::make_shared<AdamParameterAdapter>(
                               0.8, 0.9, 0.999, 1e-8, lr),
            .debug       = std::move(cfg.debug)
        };
    }())
{}

} // namespace involute::solvers
