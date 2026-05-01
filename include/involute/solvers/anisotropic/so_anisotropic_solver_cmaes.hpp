#pragma once

#include "involute/core/base_solver.hpp"
#include <sampler/anisotropic/so_anisotropic_gaussian_sampler.hpp>
#include <limits>
#include <memory>
#include <vector>

namespace involute::solvers {

struct SOAnisotropicSolverCMAESConfig {
    int N;                        // population size λ
    int d;
    std::shared_ptr<core::ConvergenceCriterion>  convergence;
    double  sigma0     = 0.5;
    int     mu         = 0;       // 0 → N/2
    double  c_c        = 0.0;     // 0 → formula default
    double  c_sigma    = 0.0;
    double  c_1        = 0.0;
    double  c_mu       = 0.0;
    double  d_sigma    = 0.0;
    int     burn_in    = 500;
    int     warm_start = 50;
    double  gamma_rtol = 0.05;
    DType                  dtype = DType::Float32;
    std::vector<core::Debugger>  debug = {};
};

/**
 * @brief Algorithm 4.10: Exact Riemannian CMA-ES on SO(d) — full D×D tangent-space covariance.
 * * Refactored to support the vector-based Product Manifold interface.
 * This solver manages a single SO(d) component wrapped in a std::vector.
 */
class SOAnisotropicSolverCMAES : public core::BaseSolver {
public:
    explicit SOAnisotropicSolverCMAES(SOAnisotropicSolverCMAESConfig config);

    /**
     * @brief Main entry point, utilizing the base run_loop.
     */
    core::CBOResult solve(core::ObjectiveFunction* obj) override;

protected:
    // Internal CMA-ES config (populated from SOAnisotropicSolverCMAESConfig)
    struct SOCMAESConfig {
        double sigma0       = 0.5;
        int    mu           = 0;
        double c_c          = 0.0;
        double c_sigma      = 0.0;
        double c_1          = 0.0;
        double c_mu         = 0.0;
        double d_sigma      = 0.0;
        int    burn_in      = 500;
        int    warm_start   = 50;
        double gamma_rtol   = 0.05;
    };
    SOCMAESConfig cmaes_cfg_;

    int D_dof_;  // D = d*(d-1)/2 (Lie algebra dimension)

    int    lambda_, mu_;
    std::vector<double> w_;
    double mu_eff_;
    double c_c_, c_sigma_, c_1_, c_mu_, d_sigma_, chi_D_;

    double sigma_;
    Tensor C_;        // [D, D] symmetric positive-definite covariance in so(d) basis
    Tensor p_c_;      // [D]   covariance evolution path
    Tensor p_sigma_;  // [D]   step-size evolution path

    // Updated to vector format to align with SolverState requirements
    std::vector<Tensor> m_hat_;       // current mean on SO(d)
    std::vector<Tensor> best_m_hat_;  // global best mean found
    double best_energy_ = std::numeric_limits<double>::infinity();

    std::unique_ptr<sampler::anisotropic::SOdAnisotropicGaussianSampler> aniso_sampler_;

    void   init_cmaes_params();

    // --- Helper methods for SO(d) specific Lie algebra operations ---

    std::vector<double> skew_to_vec(const Tensor& V) const;
    Tensor vec_to_skew(const std::vector<double>& v, DType dtype) const;
    Tensor compute_lie_transport(const Tensor& R) const;
    double vec_norm(const Tensor& v) const;

    // --- Refactored Vector-Based Overrides ---

    /**
     * @brief Initializes particles as a vector containing the SO(d) component.
     */
    std::vector<Tensor> initialize_particles() override;

    /**
     * @brief Computes the weighted Fréchet mean for the SO(d) component.
     */
    std::vector<Tensor> compute_consensus(const std::vector<Tensor>& p, const Tensor& w) override;

    /**
     * @brief Performs the Riemannian CMA-ES update step.
     */
    std::vector<Tensor> step(const std::vector<Tensor>& p,
                             const std::vector<Tensor>& c,
                             core::HyperParameters params) override;

    /**
     * @brief Verifies the SO(d) manifold constraints (orthogonality and determinant).
     */
    bool check_manifold_constraint(const std::vector<Tensor>& p) const override;
};

} // namespace involute::solvers
