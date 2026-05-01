#pragma once

#include "involute/core/base_solver.hpp"
#include "involute/solvers/adapters/adam_parameter_adapter.hpp"
#include <sampler/isotropic/so_gaussian_sampler.hpp>
#include <limits>
#include <memory>
#include <vector>

namespace involute::solvers {

    struct SOIsotropicSolverConfig {
        int N;
        int d;
        std::shared_ptr<core::ConvergenceCriterion>  convergence;
        std::shared_ptr<core::ParameterAdapter>      adapter;
        double                       lambda         = 1.0;
        double                       delta          = 0.0;   // 0.0 → diameter_so(d)
        bool                         use_matrix_exp = true;
        bool                         frechet_mean   = false;
        core::WeightingScheme        weighting      = core::WeightingScheme::Algebraic;
        DType                  dtype          = DType::Float32;
        std::vector<core::Debugger>  debug          = {};
    };

    /**
     * @brief Isotropic Consensus-Based Optimization (CBO) solver for the SO(d) manifold.
     * * Refactored to support the vector-based Product Manifold interface.
     * * Operates as a single-component manifold wrapped in a std::vector.
     */
    class SOIsotropicSolver : public core::BaseSolver {
    public:
        explicit SOIsotropicSolver(SOIsotropicSolverConfig config);

        static double diameter_so(int d);
        static core::SolverConfig to_core_config(SOIsotropicSolverConfig cfg);

    protected:
        bool use_matrix_exp_ = false;
        bool frechet_mean_   = false;

        // Cached sampler — rebuilt only when alpha = λ/δ² changes significantly.
        std::unique_ptr<sampler::SOdGaussianSampler> gaussian_sampler_;
        double cached_alpha_ = -1.0;

        // Best consensus tracking: Updated to vector format for BaseSolver compatibility.
        std::vector<Tensor> best_consensus_;
        double best_energy_ = std::numeric_limits<double>::infinity();

        /**
         * @brief Ensures the isotropic sampler is valid for the current consensus and precision.
         */
        void ensure_sampler(const Tensor &consensus, double alpha);

        /**
         * @brief Internal SVD projection utility to maintain SO(d) constraints.
         */
        Tensor enforce_so_d(const Tensor &U, const Tensor &Vt);

        // --- Refactored Vector-Based Overrides ---

        /**
         * @brief Initializes particles as a single-component vector.
         */
        std::vector<Tensor> initialize_particles() override;

        /**
         * @brief Computes the weighted consensus for the SO(d) component.
         */
        std::vector<Tensor> compute_consensus(const std::vector<Tensor> &particles,
                                               const Tensor &weights) override;

        /**
         * @brief Executes the CBO update step on SO(d).
         */
        std::vector<Tensor> step(const std::vector<Tensor> &particles,
                                 const std::vector<Tensor> &consensus,
                                 core::HyperParameters params) override;

        /**
         * @brief Validates the SO(d) manifold constraints (orthogonality and determinant).
         */
        bool check_manifold_constraint(const std::vector<Tensor> &p) const override;

        /**
         * @brief Callback to track the global best consensus point.
         */
        void on_consensus_evaluated(const std::vector<Tensor> &consensus, double energy) override;
    };
} // namespace involute::solvers
