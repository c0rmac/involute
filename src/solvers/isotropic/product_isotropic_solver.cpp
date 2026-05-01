#include "involute/solvers/isotropic/product_isotropic_solver.hpp"
#include "involute/core/math.hpp"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <string>
#include <unordered_map>

namespace involute::solvers {

using namespace involute::core;

// =============================================================================
// Constructor
// =============================================================================

ProductIsotropicSolver::ProductIsotropicSolver(ProductIsotropicSolverConfig config)
    : BaseSolver(SolverConfig{
          .N                 = config.N,
          .d                 = 0,   // computed below
          .params            = HyperParameters{1.0, {1.0}, {1.0}},
          .dtype             = config.dtype,
          .convergence       = config.convergence,
          .parameter_adapter = config.adapter,
          .debug             = config.debug
      }),
      config_(std::move(config))
{
    if (config_.manifolds.empty())
        throw std::invalid_argument("ProductIsotropicSolver: at least one manifold required");

    component_count_ = static_cast<int>(config_.manifolds.size());
    component_pool_index_.resize(component_count_);

    int total_flat_dim = 0;
    std::unordered_map<std::string, int> pool_index_by_key;

    for (int i = 0; i < component_count_; i++) {
        auto& spec = config_.manifolds[i];

        if (spec.type == ManifoldSpec::Type::SO)
            spec.k = spec.n;

        total_flat_dim += (spec.n * spec.k);

        const std::string key = std::to_string(static_cast<int>(spec.type)) + ":" +
                                std::to_string(spec.n) + ":" + std::to_string(spec.k);

        auto it = pool_index_by_key.find(key);
        if (it == pool_index_by_key.end()) {
            int idx = static_cast<int>(sampler_pools_.size());
            pool_index_by_key[key] = idx;
            sampler_pools_.push_back(SamplerPool{});
            component_pool_index_[i] = idx;
        } else {
            component_pool_index_[i] = it->second;
        }
    }

    BaseSolver::config_.d = total_flat_dim;

    if (!config_.convergence)
        config_.convergence = std::make_shared<MaxStepsCriterion>(200);

    config_.convergence->dimensional_normalisation_constant_ =
        Tensor(static_cast<float>(total_flat_dim), config_.dtype);

    BaseSolver::config_.params.lambda.clear();
    BaseSolver::config_.params.delta.clear();
    for (const auto& spec : config_.manifolds) {
        BaseSolver::config_.params.lambda.push_back(spec.lambda);
        BaseSolver::config_.params.delta.push_back(spec.delta);
    }
}

// =============================================================================
// Public solve
// =============================================================================

ProductCBOResult ProductIsotropicSolver::solve(ProductObjectiveFunction* obj) {
    CBOResult raw = BaseSolver::solve(obj);

    ProductCBOResult out;
    out.converged       = raw.converged;
    out.min_energy      = raw.min_energy;
    out.iterations_run  = raw.iterations_run;
    out.history         = std::move(raw.history);
    out.final_particles = std::move(raw.final_particles);
    out.final_consensus = std::move(raw.final_consensus);
    return out;
}

// =============================================================================
// BaseSolver overrides
// =============================================================================

std::vector<Tensor> ProductIsotropicSolver::initialize_particles() {
    std::vector<Tensor> components(component_count_);
    for (int i = 0; i < component_count_; i++)
        components[i] = initialize_component(i);
    return components;
}

std::vector<Tensor> ProductIsotropicSolver::compute_consensus(const std::vector<Tensor>& p,
                                                               const Tensor& w)
{
    std::vector<Tensor> consensus(component_count_);
    for (int i = 0; i < component_count_; i++)
        consensus[i] = compute_component_consensus(i, p[i], w);
    return consensus;
}

std::vector<Tensor> ProductIsotropicSolver::step(const std::vector<Tensor>& /*p*/,
                                                  const std::vector<Tensor>& c,
                                                  HyperParameters params)
{
    const auto& center = (best_energy_ < std::numeric_limits<double>::infinity())
                         ? best_consensus_vec_ : c;

    std::vector<Tensor> next(component_count_);
    for (int i = 0; i < component_count_; i++) {
        double alpha = params.lambda_at(i) / (params.delta_at(i) * params.delta_at(i));
        next[i] = step_component(i, center[i], alpha);
        math::eval(next[i]);
    }
    return next;
}

bool ProductIsotropicSolver::check_manifold_constraint(const std::vector<Tensor>& p) const {
    for (int i = 0; i < component_count_; i++)
        if (!check_component_constraint(i, p[i])) return false;
    return true;
}

void ProductIsotropicSolver::on_consensus_evaluated(const std::vector<Tensor>& consensus,
                                                     double energy)
{
    best_energy_       = energy;
    best_consensus_vec_ = consensus;
}

// =============================================================================
// Internal component logic
// =============================================================================

int ProductIsotropicSolver::component_pool_for(int idx) const {
    return component_pool_index_[idx];
}

void ProductIsotropicSolver::ensure_sampler_pool(int idx, const Tensor& consensus,
                                                  double alpha)
{
    const auto& spec = config_.manifolds[idx];
    auto& pool = sampler_pools_[component_pool_for(idx)];

    if (spec.type == ManifoldSpec::Type::SO) {
        if (!pool.so_sampler) {
            sampler::SOdGaussianSampler::Config cfg{
                .num_samples = config_.N, .alpha = alpha, .dtype = config_.dtype};
            pool.so_sampler = std::make_unique<sampler::SOdGaussianSampler>(
                consensus, spec.n, cfg);
            pool.cached_alpha = alpha;
        } else if (std::abs(alpha - pool.cached_alpha) > 1e-3 * std::abs(pool.cached_alpha)) {
            pool.so_sampler->set_m_hat(consensus);
            pool.so_sampler->update_alpha(alpha, 1500);
            pool.cached_alpha = alpha;
        } else {
            pool.so_sampler->set_m_hat(consensus);
        }
    } else {
        if (!pool.stiefel_sampler) {
            sampler::StiefelGaussianSampler::Config cfg{
                .num_samples = config_.N, .alpha = alpha, .dtype = config_.dtype};
            pool.stiefel_sampler = std::make_unique<sampler::StiefelGaussianSampler>(
                consensus, spec.n, spec.k, cfg);
            pool.cached_alpha = alpha;
        } else if (std::abs(alpha - pool.cached_alpha) > 1e-3 * std::abs(pool.cached_alpha)) {
            pool.stiefel_sampler->set_x_hat(consensus);
            pool.stiefel_sampler->update_alpha(alpha, 1500);
            pool.cached_alpha = alpha;
        } else {
            pool.stiefel_sampler->set_x_hat(consensus);
        }
    }
}

Tensor ProductIsotropicSolver::initialize_component(int idx) {
    const auto& spec = config_.manifolds[idx];
    Tensor identity = math::eye(spec.n, config_.dtype);
    ensure_sampler_pool(idx, identity, 1.0);

    auto& pool = sampler_pools_[component_pool_for(idx)];
    Tensor particles = (spec.type == ManifoldSpec::Type::SO)
        ? pool.so_sampler->draw_haar_od()
        : pool.stiefel_sampler->draw_uniform();
    math::eval(particles);
    return particles;
}

Tensor ProductIsotropicSolver::enforce_so(const Tensor& U, const Tensor& Vt,
                                           int d, DType dtype)
{
    int N = U.shape()[0];
    Tensor p    = math::matmul(U, Vt);
    Tensor dets = math::det(p);

    Tensor zero    = Tensor(0.0f, dtype);
    Tensor one     = Tensor(1.0f, dtype);
    Tensor neg_one = Tensor(-1.0f, dtype);
    Tensor clean_dets = math::where(math::greater(dets, zero), one, neg_one);
    Tensor dets_exp   = math::reshape(clean_dets, {N, 1, 1});

    std::vector<float> mask_data(d * d, 0.0f);
    mask_data[d * d - 1] = 1.0f;
    Tensor mask = math::array(mask_data, {d, d}, dtype);

    Tensor I = math::eye(d, dtype);
    Tensor D = math::add(I, math::multiply(mask, math::subtract(dets_exp, one)));
    return math::matmul(math::matmul(U, D), Vt);
}

Tensor ProductIsotropicSolver::compute_component_consensus(int idx, const Tensor& p,
                                                            const Tensor& w)
{
    const auto& spec = config_.manifolds[idx];
    Tensor w_exp = math::expand_dims(w, {1, 2});
    Tensor M_amb = math::sum(math::multiply(p, w_exp), {0});
    Tensor M_bat = math::expand_dims(M_amb, {0});
    auto [U, S, Vt] = math::svd(M_bat);

    if (spec.type == ManifoldSpec::Type::SO)
        return math::squeeze(enforce_so(U, Vt, spec.n, config_.dtype), {0});

    U = math::slice(U, 0, Vt.shape()[1], 2);
    return math::sum(math::matmul(U, Vt), {0});
}

Tensor ProductIsotropicSolver::step_component(int idx, const Tensor& c, double alpha) {
    ensure_sampler_pool(idx, c, alpha);
    auto& pool = sampler_pools_[component_pool_for(idx)];
    return (config_.manifolds[idx].type == ManifoldSpec::Type::SO)
        ? pool.so_sampler->sample()
        : pool.stiefel_sampler->sample();
}

bool ProductIsotropicSolver::check_component_constraint(int idx,
                                                         const Tensor& p) const
{
    const auto& spec = config_.manifolds[idx];
    Tensor p_T = math::transpose(p, {0, 2, 1});
    Tensor PtP = math::matmul(p_T, p);
    Tensor I   = math::eye(spec.k, config_.dtype);
    double mse = math::to_double(math::sum(math::square(math::subtract(PtP, I))))
                 / (config_.N * spec.k * spec.k);
    return mse <= 1e-5;
}

} // namespace involute::solvers
