#pragma once

#include "involute/core/base_solver.hpp"
#include <sampler/anisotropic/so_anisotropic_heterogeneous_gaussian_sampler.hpp>
#include <limits>
#include <memory>
#include <vector>

namespace involute::solvers {

    struct SOAnisotropicSolverADAMConfig {
        int N;
        int d;
        std::shared_ptr<core::ConvergenceCriterion>  convergence;
        double                       lambda  = 1.0;
        double                       delta   = 0.0;    // 0.0 → diameter_so(d)
        double                       beta1   = 0.9;
        double                       beta2   = 0.999;
        double                       epsilon = 1e-8;
        DType                  dtype   = DType::Float32;
        std::vector<core::Debugger>  debug   = {};
    };

    /**
     * @brief Algorithm 4.9: Exact Adam-Consensus-Based Optimization on SO(d).
     * * Refactored to be compatible with the vector-based Product Manifold interface.
     * Each particle i maintains a per-basis-element second moment V_{t,a}^(i)
     * tracking the EMA of squared Riemannian displacements.
     */
    class SOAnisotropicSolverADAM : public core::BaseSolver {
    public:
        explicit SOAnisotropicSolverADAM(SOAnisotropicSolverADAMConfig config);

        static double diameter_so(int d);

    protected:
        // Internal Adam config (populated from SOAnisotropicSolverADAMConfig)
        struct SOAdamConfig {
            double beta1   = 0.9;
            double beta2   = 0.999;
            double epsilon = 1e-8;
        };
        SOAdamConfig adam_cfg_;

        // Per-particle D×D second-moment matrices V_t^(i): shape [N][D*D], D = d(d-1)/2.
        // Stored on CPU for EMA updates before being converted to precision tensors.
        std::vector<std::vector<double>> var_cpu_;
        int t_ = 0;

        // Algorithm 4.8 sampler — one per-particle anisotropic SO(d) Gaussian.
        std::unique_ptr<sampler::anisotropic::SOdAnisotropicHeterogeneousGaussianSampler> hetero_sampler_;

        // Best-consensus tracking: Updated to vector format for consistency with BaseSolver state.
        std::vector<Tensor> best_consensus_;
        double best_energy_ = std::numeric_limits<double>::infinity();

        // --- Refactored Vector-Based Overrides ---

        /**
         * @brief Initializes particles as a vector containing a single SO(d) component.
         */
        std::vector<Tensor> initialize_particles() override;

        /**
         * @brief Computes the weighted consensus for the SO(d) component.
         */
        std::vector<Tensor> compute_consensus(const std::vector<Tensor> &particles,
                                               const Tensor &weights) override;

        /**
         * @brief Updates particle positions using the Riemannian Adam-CBO step.
         */
        std::vector<Tensor> step(const std::vector<Tensor> &particles,
                                 const std::vector<Tensor> &consensus,
                                 core::HyperParameters params) override;

        /**
         * @brief Validates that the SO(d) component remains on the manifold (orthogonal, det=1).
         */
        bool check_manifold_constraint(const std::vector<Tensor> &p) const override;

        /**
         * @brief Callback to track the global best consensus found during the run.
         */
        void on_consensus_evaluated(const std::vector<Tensor> &consensus, double energy) override;

    private:
        /**
         * @brief Internal utility to re-project components back to SO(d) if necessary.
         */
        Tensor enforce_so_d(const Tensor &U, const Tensor &Vt);
    };

} // namespace involute::solvers
