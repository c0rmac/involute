#pragma once

#include "so_anisotropic_solver_cmaes.hpp"
#include "involute/core/base_solver.hpp"
#include <sampler/anisotropic/so_anisotropic_spatial_gaussian_sampler.hpp>
#include <limits>
#include <memory>
#include <vector>

namespace involute::solvers {

/**
 * @brief Algorithm 4.9: Intrinsic Riemannian CMA-ES on SO(d) — spatial-precision variant.
 * * This version maintains the covariance as a d×d symmetric positive-definite tensor Σ.
 * Refactored to support the vector-based Product Manifold interface where the SO(d)
 * component is managed as a single-element vector of Tensors.
 */
class SOAnisotropicSolverCMAESSpatial : public core::BaseSolver {
public:
    explicit SOAnisotropicSolverCMAESSpatial(SOAnisotropicSolverCMAESConfig config);

    /**
     * @brief Main entry point for the solver, utilizing the base run_loop logic.
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

    int D_dof_; // D = d*(d-1)/2

    int    lambda_, mu_;
    std::vector<double> w_;
    double mu_eff_;
    double c_c_, c_sigma_, c_1_, c_mu_, d_sigma_, chi_D_;

    double sigma_;
    Tensor Sigma_;    // d×d spatial covariance
    Tensor p_c_;      // d×d covariance evolution path (skew-symmetric)
    Tensor p_sigma_;  // d×d step-size evolution path (skew-symmetric)

    // Consensus and best-tracking updated to vector format for BaseSolver compatibility
    std::vector<Tensor> m_hat_;
    std::vector<Tensor> best_m_hat_;
    double best_energy_ = std::numeric_limits<double>::infinity();

    std::unique_ptr<sampler::anisotropic::SOdAnisotropicSpatialGaussianSampler> aniso_sampler_;

    void   init_cmaes_params();
    Tensor transport_matrix(const Tensor& M, const Tensor& R) const;
    double intrinsic_norm(const Tensor& V) const;

    // --- Refactored Vector-Based Overrides ---

    /**
     * @brief Initializes particles as a single-component vector containing the SO(d) component.
     */
    std::vector<Tensor> initialize_particles() override;

    /**
     * @brief Computes the weighted Fréchet mean for the SO(d) component.
     */
    std::vector<Tensor> compute_consensus(const std::vector<Tensor>& p, const Tensor& w) override;

    /**
     * @brief Updates particle positions using the spatial-precision Riemannian CMA-ES step.
     */
    std::vector<Tensor> step(const std::vector<Tensor>& p,
                             const std::vector<Tensor>& c,
                             core::HyperParameters params) override;

    /**
     * @brief Validates the SO(d) manifold constraints (orthogonality and determinant).
     */
    bool check_manifold_constraint(const std::vector<Tensor>& p) const override;

    /**
     * @brief Callback to track the global best consensus point.
     */
    void on_consensus_evaluated(const std::vector<Tensor>& consensus, double energy) override;
};

} // namespace involute::solvers
