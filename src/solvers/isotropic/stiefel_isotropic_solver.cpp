#include "involute/solvers/isotropic/stiefel_isotropic_solver.hpp"
#include "involute/core/math.hpp"
#include <cmath>
#include <algorithm>
#include <iostream>
#include <stdexcept>

namespace involute::solvers {
    using namespace involute::core;

    // ==============================================================================
    // Static helpers
    // ==============================================================================

    double StiefelIsotropicSolver::diameter_stiefel(int n, int k) {
        return std::sqrt(2.0 * static_cast<double>(k));
    }

    SolverConfig StiefelIsotropicSolver::to_core_config(StiefelIsotropicSolverConfig cfg) {
        double delta = (cfg.delta <= 0.0) ? diameter_stiefel(cfg.n, cfg.k) : cfg.delta;
        return SolverConfig{
            .N = cfg.N,
            .d = cfg.n,
            .params = HyperParameters{.beta = 1.0, .lambda = {cfg.lambda}, .delta = {delta}},
            .dtype = cfg.dtype,
            .convergence = cfg.convergence,
            .parameter_adapter = cfg.adapter,
            .weighting_scheme = cfg.weighting,
            .debug = cfg.debug
        };
    }

    // ==============================================================================
    // Constructor
    // ==============================================================================

    StiefelIsotropicSolver::StiefelIsotropicSolver(StiefelIsotropicSolverConfig config)
        : core::BaseSolver(to_core_config(config))
        , n_(config.n)
        , k_(config.k)
        , use_matrix_exp_(config.use_matrix_exp) {}

    // ==============================================================================
    // Sampler management
    // ==============================================================================

    void StiefelIsotropicSolver::ensure_sampler(const Tensor &consensus, double alpha) {
        const double alpha_rtol = 1e-3;
        const bool alpha_changed = !gaussian_sampler_ ||
                             std::abs(alpha - cached_alpha_) > alpha_rtol * std::abs(cached_alpha_);

        if (!gaussian_sampler_) {
            sampler::StiefelGaussianSampler::Config cfg;
            cfg.num_samples              = config_.N;
            cfg.alpha                    = alpha;
            cfg.dtype                    = config_.dtype;
            cfg.angle_cfg.burn_in        = 500;
            cfg.angle_cfg.leapfrog_steps = 5;
            cfg.angle_cfg.num_threads    = 4;

            gaussian_sampler_ = std::make_unique<sampler::StiefelGaussianSampler>(
                consensus, n_, k_, cfg);
            cached_alpha_ = alpha;
        } else if (alpha_changed) {
            gaussian_sampler_->set_x_hat(consensus);
            gaussian_sampler_->update_alpha(alpha, 500);
            cached_alpha_ = alpha;
        } else {
            gaussian_sampler_->set_x_hat(consensus);
        }
    }

    // ==============================================================================
    // Stiefel projection via thin SVD
    // ==============================================================================

    Tensor StiefelIsotropicSolver::project_stiefel(const Tensor &M) const {
        // Projects an ambient matrix M onto V(n,k) via U * Vt from thin SVD
        auto [U, S, Vt] = math::svd(M);
        return math::matmul(U, Vt);
    }

    // ==============================================================================
    // BaseSolver interface (Vector-Aware Overrides)
    // ==============================================================================

    std::vector<Tensor> StiefelIsotropicSolver::initialize_particles() {
        // Sample from uniform Haar measure on V(n, k) for initialization
        Tensor identity = math::eye(config_.d, config_.dtype);
        ensure_sampler(identity, 1.0);

        Tensor initial_particles = gaussian_sampler_->draw_uniform();
        math::eval(initial_particles);

        return { initial_particles }; // Return as single-component vector
    }

    std::vector<Tensor> StiefelIsotropicSolver::compute_consensus(const std::vector<Tensor> &particles_vec,
                                                       const Tensor &weights) {
        const Tensor &particles = particles_vec[0]; // Extract Stiefel component

        // Weighted ambient mean: M_amb = sum_i w_i * X_i
        Tensor weights_expanded = math::expand_dims(weights, {1, 2});
        Tensor weighted = math::multiply(particles, weights_expanded);
        Tensor M_amb = math::sum(weighted, {0});

        // Project back onto V(n, k) via thin SVD
        auto [U, S, Vt] = math::svd(math::expand_dims(M_amb, {0}));

        // Ensure dimensionality alignment for thin SVD projection
        U = math::slice(U, 0, Vt.shape()[1], 2);
        Tensor Q = math::matmul(U, Vt);

        return { math::sum(Q, {0}) }; // Return consensus as single-component vector
    }

    void StiefelIsotropicSolver::on_consensus_evaluated(const std::vector<Tensor> &consensus, double energy) {
        if (energy < best_energy_) {
            best_energy_   = energy;
            best_consensus_ = consensus; // Store vector state for drift centering
        }
    }

    std::vector<Tensor> StiefelIsotropicSolver::step(const std::vector<Tensor> &particles_vec,
                                           const std::vector<Tensor> &consensus_vec,
                                           HyperParameters params)
    {
        const double alpha = params.lambda_at(0) / (params.delta_at(0) * params.delta_at(0));

        // Use best-found consensus as the Gaussian center (drift center)
        const std::vector<Tensor> &centre_vec = (best_energy_ < std::numeric_limits<double>::infinity())
                                               ? best_consensus_
                                               : consensus_vec;

        ensure_sampler(centre_vec[0], alpha);

        // Returns [N, n, k] particles drawn from the Riemannian Gaussian on V(n,k)
        return { gaussian_sampler_->sample() };
    }

    bool StiefelIsotropicSolver::check_manifold_constraint(const std::vector<Tensor> &p_vec) const {
        const Tensor &p = p_vec[0];

        // Check X^T X ≈ I_k for all N particles
        Tensor p_T  = math::transpose(p, {0, 2, 1});
        Tensor PtP  = math::matmul(p_T, p);
        Tensor I    = math::eye(k_, config_.dtype);
        Tensor diff = math::subtract(PtP, I);

        double mse = math::to_double(math::sum(math::square(diff))) / (config_.N * k_ * k_);

        return (mse <= 1e-5);
    }

} // namespace involute::solvers
