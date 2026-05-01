#pragma once

#include "involute/core/parameter_adapter.hpp"
#include "involute/core/tensor.hpp"
#include "involute/core/math.hpp"
#include "involute/core/hyper_parameters.hpp"
#include <vector>
#include <memory>

namespace involute::solvers {

    // Bring core types into scope to use Tensor and DType directly
    using namespace involute::core;

    /**
     * @brief A Riemannian CMA-ES adapter utilizing Cumulative Step-Size Adaptation (CSA).
     * * Refactored for Product Manifold support: This adapter maintains a vector of
     * evolution paths and intrinsic parameters, allowing isotropic noise (delta)
     * to be adapted independently for each manifold component in a product space.
     */
    class CMAESParameterAdapter : public ParameterAdapter {
    public:
        double max_beta_achieved_;

        /**
         * @brief Constructs an adapter utilizing Cumulative Step-Size Adaptation (CSA).
         * @param max_consensus_weight The peak target weight for the best particle.
         * @param c_sigma Learning rate for the evolution path (typically proportional to 1/sqrt(D)).
         * @param d_sigma Damping factor for the step-size adaptation (typically close to 1.0).
         */
        CMAESParameterAdapter(double max_consensus_weight, double c_sigma = 0.3, double d_sigma = 1.0);

        ~CMAESParameterAdapter() override = default;

        /**
         * @brief Interface implementation using vector-based component states.
         */
        Tensor compute_consensus_weights(
            int step,
            const Tensor &costs,
            const std::vector<Tensor> &particles,
            const std::vector<Tensor> &prev_consensus,
            double prev_energy,
            HyperParameters &params) override;

        /**
         * @brief Adapts isotropic noise (delta) using independent evolution paths for each component.
         * * For each manifold in the product space, this method extracts the tangent step via
         * the Riemannian logarithm, parallel transports the historical path, and updates
         * the component-specific delta based on the local path length.
         */
        void adapt(
            int step,
            const std::vector<Tensor> &particles,
            const std::vector<Tensor> &current_consensus,
            const std::vector<Tensor> &prev_consensus,
            double current_energy,
            double prev_energy,
            const Tensor &weights,
            HyperParameters &params) override;

        void reset() override;

        bool ready_to_converge(HyperParameters params) override;

        std::shared_ptr<ParameterAdapter> clone() const override;

    private:
        // --- Consensus Weight Parameters ---
        double max_consensus_weight_;
        double min_consensus_weight_ = 1e-4;
        int max_attempts_;

        // --- CSA State & Parameters (Vectorized for Multi-Component Support) ---
        double c_sigma_;              // Learning rate for the evolution path
        double d_sigma_;              // Damping factor for the step-size adaptation
        bool is_initialized_ = false; // Flag to trigger state initialization

        // Evolution paths and expected norms tracked independently per manifold component.
        // These vectors allow the solver to handle SO(3) x SO(2) etc. without padding.
        std::vector<Tensor> p_sigma_vec_;
        std::vector<double> expected_norm_chi_vec_;

        /**
         * @brief Solves for the inverse temperature beta that achieves the target maximum weight.
         */
        double solve_for_target_beta(
            const std::vector<float> &shifted_costs,
            double target_max_weight,
            double tol = 1e-5,
            int max_iter = 100);
    };

} // namespace involute::solvers
