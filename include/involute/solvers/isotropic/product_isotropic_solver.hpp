#pragma once

#include "involute/core/base_solver.hpp"
#include "involute/core/objective.hpp"

#include <sampler/isotropic/so_gaussian_sampler.hpp>
#include <sampler/isotropic/stiefel_gaussian_sampler.hpp>

#include <limits>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

namespace involute::solvers {

using namespace involute::core;

// =============================================================================
// ManifoldSpec — describes one component of the product space
// =============================================================================

/**
 * @brief Specification for a single manifold component within the product space.
 */
struct ManifoldSpec {
    enum class Type { SO, Stiefel };
    Type   type;
    int    n;
    int    k;
    double lambda;
    double delta;

    static ManifoldSpec SO(int n, double lambda, double delta) {
        return {Type::SO, n, n, lambda, delta};
    }
    static ManifoldSpec Stiefel(int n, int k, double lambda, double delta) {
        return {Type::Stiefel, n, k, lambda, delta};
    }
};

// =============================================================================
// ProductCBOResult
// =============================================================================

/**
 * @brief Optimization result for a product-manifold solve.
 */
struct ProductCBOResult {
    bool                    converged;
    double                  min_energy;
    int                     iterations_run;
    std::vector<StepRecord> history;
    std::vector<Tensor>     final_particles;
    std::vector<Tensor>     final_consensus;
};

// =============================================================================
// ProductIsotropicSolverConfig
// =============================================================================

/**
 * @brief Configuration for ProductIsotropicSolver.
 *
 * Supports C++20 designated-initializer syntax in test files.
 */
struct ProductIsotropicSolverConfig {
    std::vector<ManifoldSpec>              manifolds;
    int                                    N;
    DType                                  dtype  = DType::Float32;
    std::shared_ptr<ConvergenceCriterion>  convergence;
    std::shared_ptr<ParameterAdapter>      adapter;
    std::vector<Debugger>                  debug  = {};
};

// =============================================================================
// ProductIsotropicSolver
// =============================================================================

/**
 * @brief Product-manifold CBO solver using isotropic Riemannian Gaussian samplers.
 *
 * Operates natively on vectors of Tensors to support mixed-dimension manifolds.
 * Each component can be either SO(n) or V(n,k) (Stiefel).
 * A ParameterAdapter (e.g. AdamParameterAdapter, CMAESParameterAdapter) controls
 * how the noise scale evolves across iterations.
 */
class ProductIsotropicSolver : public BaseSolver {
public:
    explicit ProductIsotropicSolver(ProductIsotropicSolverConfig config);

    ~ProductIsotropicSolver() override = default;

    /** @brief Run the solver on a product-manifold objective. */
    ProductCBOResult solve(ProductObjectiveFunction* obj);

protected:
    struct SamplerPool {
        std::unique_ptr<sampler::SOdGaussianSampler>      so_sampler;
        std::unique_ptr<sampler::StiefelGaussianSampler>  stiefel_sampler;
        double cached_alpha = -1.0;
    };

    std::vector<Tensor> best_consensus_vec_;
    double              best_energy_ = std::numeric_limits<double>::infinity();

    // BaseSolver overrides
    std::vector<Tensor> initialize_particles() override;
    std::vector<Tensor> compute_consensus(const std::vector<Tensor>& p,
                                          const Tensor& w) override;
    std::vector<Tensor> step(const std::vector<Tensor>& p,
                             const std::vector<Tensor>& c,
                             HyperParameters params) override;
    bool check_manifold_constraint(const std::vector<Tensor>& p) const override;
    void on_consensus_evaluated(const std::vector<Tensor>& consensus,
                                double energy) override;

private:
    ProductIsotropicSolverConfig config_;
    int                          component_count_ = 0;

    std::vector<int>        component_pool_index_;
    std::vector<SamplerPool> sampler_pools_;

    void   ensure_sampler_pool(int idx, const Tensor& consensus, double alpha);
    int    component_pool_for(int idx) const;

    Tensor initialize_component(int idx);
    Tensor enforce_so(const Tensor& U, const Tensor& Vt, int d, DType dtype);
    Tensor compute_component_consensus(int idx, const Tensor& particles,
                                       const Tensor& weights);
    Tensor step_component(int idx, const Tensor& consensus, double alpha);
    bool   check_component_constraint(int idx, const Tensor& particles) const;
};

} // namespace involute::solvers
