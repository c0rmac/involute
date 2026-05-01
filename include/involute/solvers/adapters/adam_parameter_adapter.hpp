#pragma once

#include "involute/core/parameter_adapter.hpp"
#include "involute/core/hyper_parameters.hpp"
#include <vector>
#include <memory>

namespace involute::solvers {

    using namespace involute::core;

    /**
     * @brief An Adam-inspired parameter adapter refactored for component-wise vector state.
     * * This adapter manages the dynamic adjustment of dispersion (lambda) and noise (beta)
     * parameters. It tracks momentum (m_t) and uncentered variance (v_t) for each
     * manifold component independently using vectors of tensors.
     */
    class AdamParameterAdapter : public ParameterAdapter {
    public:
        double max_beta_achieved_;

        explicit AdamParameterAdapter(double max_consensus_weight = 0.8,
                                      double beta1 = 0.9,
                                      double beta2 = 0.999,
                                      double epsilon = 1e-8,
                                      double lr_lambda = 0.2,
                                      double target_ratio = 0.5);

        ~AdamParameterAdapter() override = default;

        /**
         * @brief Solves for the inverse temperature beta that achieves the target maximum weight.
         */
        double solve_for_target_beta(const std::vector<float> &shifted_costs,
                                     double target_max_weight = 0.95,
                                     double tol = 1e-7,
                                     int max_iter = 100);

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
         * @brief Updates lambda for each manifold component independently.
         * * Iterates through the provided vector of component consensus points to update
         * subspace-specific Adam moments.
         */
        void adapt(int step,
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
        double max_consensus_weight_;
        double beta1_;
        double beta2_;
        double epsilon_;
        double lr_lambda_;
        double target_ratio_;

        // --- Multi-Component State ---
        // These vectors store the Adam moments for each manifold component.
        std::vector<Tensor> m_t_vec_;
        std::vector<Tensor> v_t_vec_;

        int t_;
        int initial_patience;
    };

} // namespace involute::solvers
