#pragma once

#include "involute/core/base_solver.hpp"
#include "involute/solvers/adapters/adam_parameter_adapter.hpp"
#include <sampler/isotropic/stiefel_gaussian_sampler.hpp>
#include <memory>
#include <vector>

namespace involute::solvers {

    struct StiefelIsotropicSolverConfig {
        int N;
        int n;                                                      // ambient dim (rows)
        int k;                                                      // frame width (cols), k < n
        std::shared_ptr<core::ConvergenceCriterion>  convergence;
        std::shared_ptr<core::ParameterAdapter>      adapter;
        double                       lambda         = 1.0;
        double                       delta          = 0.0;   // 0.0 → diameter_stiefel(n,k)
        bool                         use_matrix_exp = true;
        core::WeightingScheme        weighting      = core::WeightingScheme::Algebraic;
        DType                  dtype          = DType::Float32;
        std::vector<core::Debugger>  debug          = {};
    };

    /**
     * @brief Consensus-Based Optimisation on the Stiefel manifold V(n, k).
     * The set of n×k real matrices with orthonormal columns (X^T X = I_k).
     * Refactored to support the vector-based Product Manifold interface.
     */
    class StiefelIsotropicSolver : public core::BaseSolver {
    public:
        explicit StiefelIsotropicSolver(StiefelIsotropicSolverConfig config);

        static double diameter_stiefel(int n, int k);
        static core::SolverConfig to_core_config(StiefelIsotropicSolverConfig cfg);

    protected:
        int  n_ = 0;             // ambient dimension (rows)
        int  k_ = 0;             // frame width (columns)
        bool use_matrix_exp_ = true;

        // Cached sampler — rebuilt only when alpha = λ/δ² changes significantly.
        std::unique_ptr<sampler::StiefelGaussianSampler> gaussian_sampler_;
        double cached_alpha_ = -1.0;

        // Best consensus tracking: Updated to vector format for BaseSolver compatibility.
        std::vector<Tensor> best_consensus_;
        double best_energy_ = std::numeric_limits<double>::infinity();

        /**
         * @brief Ensures the isotropic Stiefel sampler is valid for current consensus and precision.
         */
        void ensure_sampler(const Tensor &consensus, double alpha);

        /**
         * @brief Projects an ambient n×k matrix onto V(n,k) via thin SVD.
         */
        Tensor project_stiefel(const Tensor &M) const;

        // --- Refactored Vector-Based Overrides ---

        /**
         * @brief Initializes particles as a single-component vector containing the V(n,k) component.
         */
        std::vector<Tensor> initialize_particles() override;

        /**
         * @brief Computes the weighted consensus for the Stiefel component.
         */
        std::vector<Tensor> compute_consensus(const std::vector<Tensor> &particles,
                                               const Tensor &weights) override;

        /**
         * @brief Updates particle positions using the Stiefel update step.
         */
        std::vector<Tensor> step(const std::vector<Tensor> &particles,
                                 const std::vector<Tensor> &consensus,
                                 core::HyperParameters params) override;

        /**
         * @brief Validates the Stiefel manifold constraints (X^T X = I_k).
         */
        bool check_manifold_constraint(const std::vector<Tensor> &p) const override;

        /**
         * @brief Callback to track the global best consensus point found.
         */
        void on_consensus_evaluated(const std::vector<Tensor> &consensus, double energy) override;
    };

} // namespace involute::solvers
