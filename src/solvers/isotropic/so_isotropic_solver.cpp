#include "involute/solvers/isotropic/so_isotropic_solver.hpp"
#include "involute/core/math.hpp"
#include <cmath>
#include <algorithm>
#include <iostream>
#include <vector>

namespace involute::solvers {
    using namespace involute::core;

    // ==============================================================================
    // Static helpers
    // ==============================================================================

    double SOIsotropicSolver::diameter_so(int d) {
        return M_PI * std::sqrt(static_cast<double>(d * (d - 1)) / 2.0);
    }

    SolverConfig SOIsotropicSolver::to_core_config(SOIsotropicSolverConfig cfg) {
        double delta = (cfg.delta <= 0.0) ? diameter_so(cfg.d) : cfg.delta;
        return SolverConfig{
            .N = cfg.N,
            .d = cfg.d,
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

    SOIsotropicSolver::SOIsotropicSolver(SOIsotropicSolverConfig config)
        : core::BaseSolver(to_core_config(config))
        , use_matrix_exp_(config.use_matrix_exp)
        , frechet_mean_(config.frechet_mean) {}

    // ==============================================================================
    // Sampler management
    // ==============================================================================

    void SOIsotropicSolver::ensure_sampler(const Tensor &consensus, double alpha) {
        const double alpha_rtol = 1e-3;
        const bool alpha_changed = !gaussian_sampler_ ||
                             std::abs(alpha - cached_alpha_) > alpha_rtol * std::abs(cached_alpha_);

        if (!gaussian_sampler_) {
            sampler::SOdGaussianSampler::Config cfg;
            cfg.num_samples              = config_.N;
            cfg.alpha                    = alpha;
            cfg.dtype                    = config_.dtype;
            cfg.angle_cfg.burn_in        = 500;
            cfg.angle_cfg.leapfrog_steps = 5;
            cfg.angle_cfg.num_threads    = 8;

            gaussian_sampler_ = std::make_unique<sampler::SOdGaussianSampler>(
                consensus, config_.d, cfg);
            cached_alpha_ = alpha;
        } else if (alpha_changed) {
            gaussian_sampler_->set_m_hat(consensus);
            gaussian_sampler_->update_alpha(alpha, 1500);
            cached_alpha_ = alpha;
        } else {
            gaussian_sampler_->set_m_hat(consensus);
        }
    }

    // ==============================================================================
    // SO(d) SVD projection
    // ==============================================================================

    Tensor SOIsotropicSolver::enforce_so_d(const Tensor &U, const Tensor &Vt) {
        int N = U.shape()[0];
        int d = U.shape()[1];

        Tensor p_tensor = math::matmul(U, Vt);
        Tensor dets = math::det(p_tensor);

        Tensor zero    = Tensor(0.0f, U.dtype());
        Tensor one     = Tensor(1.0f, U.dtype());
        Tensor neg_one = Tensor(-1.0f, U.dtype());
        Tensor clean_dets = math::where(math::greater(dets, zero), one, neg_one);

        Tensor dets_exp = math::reshape(clean_dets, {N, 1, 1});

        std::vector<float> mask_data(d * d, 0.0f);
        mask_data[d * d - 1] = 1.0f;
        Tensor mask = math::array(mask_data, {d, d}, U.dtype());

        Tensor I = math::eye(d, U.dtype());
        Tensor D = math::add(I, math::multiply(mask, math::subtract(dets_exp, one)));

        return math::matmul(math::matmul(U, D), Vt);
    }

    // ==============================================================================
    // BaseSolver interface (Vector-Aware Overrides)
    // ==============================================================================

    std::vector<Tensor> SOIsotropicSolver::initialize_particles() {
        Tensor identity = math::eye(config_.d, config_.dtype);
        ensure_sampler(identity, 1.0);

        // Sampling from Haar measure for maximum initial spread
        Tensor initial_particles = gaussian_sampler_->draw_haar_od();
        math::eval(initial_particles);

        return { initial_particles }; // Return as single-component vector
    }

    std::vector<Tensor> SOIsotropicSolver::compute_consensus(const std::vector<Tensor> &particles_vec,
                                                   const Tensor &weights) {
        const Tensor &particles = particles_vec[0]; // Unpack the SO(d) component

        // Chordal SVD projection (Default)
        auto svd_consensus = [&]() -> Tensor {
            Tensor weighted = math::multiply(particles, math::expand_dims(weights, {1, 2}));
            Tensor M_amb    = math::sum(weighted, {0});
            Tensor M_batched = math::expand_dims(M_amb, {0});
            auto [U, S, Vt] = math::svd(M_batched);
            return math::squeeze(enforce_so_d(U, Vt), {0});
        };

        if (!frechet_mean_) return { svd_consensus() };

        // Riemannian Fréchet mean (Iterative Geodesic Descent)
        const int    max_iter = 20;
        const double tol      = 1e-3;
        Tensor M = svd_consensus();

        for (int iter = 0; iter < max_iter; ++iter) {
            Tensor MT    = math::transpose(math::expand_dims(M, {0}), {0, 2, 1});
            Tensor R     = math::matmul(MT, particles);
            Tensor V_raw = math::matrix_log(R);

            // Strict Lie algebra enforcement for skew-symmetry
            Tensor V_T = math::transpose(V_raw, {0, 2, 1});
            Tensor V   = math::multiply(math::subtract(V_raw, V_T), Tensor(0.5f, V_raw.dtype()));

            Tensor V_bar = math::sum(math::multiply(V, math::expand_dims(weights, {1, 2})), {0});
            M = math::matmul(M, math::matrix_exp(V_bar));

            double norm = std::sqrt(math::to_double(math::sum(math::square(V_bar))));
            if (norm < tol) break;
        }

        return { M }; // Return as single-component vector
    }

    void SOIsotropicSolver::on_consensus_evaluated(const std::vector<Tensor> &consensus, double energy) {
        best_energy_   = energy;
        best_consensus_ = consensus; // Store vector state
    }

    std::vector<Tensor> SOIsotropicSolver::step(const std::vector<Tensor> &particles_vec,
                                      const std::vector<Tensor> &consensus_vec,
                                      HyperParameters params)
    {
        const double alpha = params.lambda_at(0) / (params.delta_at(0) * params.delta_at(0));

        // Use best-found consensus as the drift centre
        const std::vector<Tensor> &centre_vec = (best_energy_ < std::numeric_limits<double>::infinity())
                                               ? best_consensus_
                                               : consensus_vec;

        ensure_sampler(centre_vec[0], alpha);

        // sample() returns [N, d, d] particles from the Riemannian Gaussian
        return { gaussian_sampler_->sample() };
    }

    bool SOIsotropicSolver::check_manifold_constraint(const std::vector<Tensor> &p_vec) const {
        const Tensor &p = p_vec[0];
        Tensor p_T = math::transpose(p, {0, 2, 1});
        Tensor Pt_P = math::matmul(p_T, p);
        Tensor I = math::eye(config_.d, config_.dtype);
        Tensor diff = math::subtract(Pt_P, I);

        double total_squared_error = math::to_double(math::sum(math::square(diff)));
        double mean_squared_error = total_squared_error / (config_.N * config_.d * config_.d);

        return (mean_squared_error <= 1e-5);
    }
} // namespace involute::solvers
