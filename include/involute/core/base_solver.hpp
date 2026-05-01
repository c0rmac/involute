#pragma once

#include "involute/core/tensor.hpp"
#include "involute/core/hyper_parameters.hpp"
#include "involute/core/objective.hpp"
#include "involute/core/result.hpp"
#include "involute/core/parameter_adapter.hpp"
#include <memory>
#include <vector>
#include <cmath>
#include <iostream>

namespace involute::core {

    /**
     * @brief Encapsulates the current state of the solver for convergence and adaptation.
     * Refactored: current_consensus and prev_consensus are now vectors to support
     * product manifold components natively.
     */
    struct SolverState {
        int step{};
        double current_energy{};
        double prev_energy{};
        const std::vector<Tensor> *current_consensus{};
        const std::vector<Tensor> *prev_consensus{};
        int d; // Total flattened dimension or component count depending on solver
        int N; // Swarm size
    };

    // ==============================================================================
    // CONVERGENCE STRATEGIES
    // ==============================================================================

    class ConvergenceCriterion {
    public:
        Tensor dimensional_normalisation_constant_;
        virtual ~ConvergenceCriterion() = default;
        virtual bool check(const SolverState &state) = 0;
    };

    class MaxStepsCriterion : public ConvergenceCriterion {
        int max_steps_;
    public:
        explicit MaxStepsCriterion(int max_steps);
        bool check(const SolverState &state) override;
    };

    class EnergyToleranceCriterion : public ConvergenceCriterion {
        double tolerance_;
        int min_steps_;
    public:
        explicit EnergyToleranceCriterion(double tol, int min_steps = 50);
        bool check(const SolverState &state) override;
    };

    /**
     * @brief Checks convergence based on the movement of the consensus point.
     * Updated to handle vector-based consensus components.
     */
    class ConsensusToleranceCriterion : public ConvergenceCriterion {
        double tolerance_;
        int min_steps_;
    public:
        explicit ConsensusToleranceCriterion(double tol, int min_steps = 50);
        bool check(const SolverState &state) override;
    };

    class CompositeCriterion : public ConvergenceCriterion {
        std::vector<std::shared_ptr<ConvergenceCriterion>> criteria_;
    public:
        void add(std::shared_ptr<ConvergenceCriterion> criterion);
        bool check(const SolverState &state) override;
    };

    // ==============================================================================
    // SOLVER CONFIGURATION
    // ==============================================================================

    enum class WeightingScheme {
        Exponential,
        Algebraic
    };

    enum class Debugger {
        History,
        Log
    };

    struct SolverConfig {
        int N;
        int d;
        HyperParameters params;
        DType dtype;
        std::shared_ptr<ConvergenceCriterion> convergence;
        std::shared_ptr<ParameterAdapter> parameter_adapter;
        WeightingScheme weighting_scheme = WeightingScheme::Algebraic;
        std::vector<Debugger> debug;
    };

    // ==============================================================================
    // BASE SOLVER
    // ==============================================================================

    class BaseSolver {
    protected:
        SolverConfig config_;

        // Active objective pointers — set by solve() before run_loop() is called.
        // Exactly one of these is non-null during a solver run.
        ObjectiveFunction*        obj_single_  = nullptr;
        ProductObjectiveFunction* obj_product_ = nullptr;

    public:
        explicit BaseSolver(SolverConfig config);
        virtual ~BaseSolver() = default;

        /**
         * @brief Run the solver on a single-manifold objective.
         */
        virtual CBOResult solve(ObjectiveFunction *obj);

        /**
         * @brief Run the solver on a product-manifold objective.
         */
        CBOResult solve(ProductObjectiveFunction *obj);

    protected:
        // --- Vector-Based Swarm Interface ---
        // These methods ensure each manifold component is treated as a discrete Tensor.

        virtual std::vector<Tensor> initialize_particles() = 0;
        virtual std::vector<Tensor> compute_consensus(const std::vector<Tensor> &p, const Tensor &w) = 0;
        virtual std::vector<Tensor> step(const std::vector<Tensor> &p,
                                         const std::vector<Tensor> &c,
                                         HyperParameters params) = 0;
        virtual bool check_manifold_constraint(const std::vector<Tensor> &p) const = 0;

        /**
         * @brief Evaluates the active objective against a vector of particle tensors.
         * Dispatches to obj_single_ (unwrapping the single component) or obj_product_
         * depending on which pointer was set by solve().
         */
        virtual Tensor evaluate_objective(const std::vector<Tensor> &particles);

        virtual Tensor compute_weights(const Tensor &costs,
                                       const std::vector<Tensor> &particles,
                                       const std::vector<Tensor> &prev_consensus,
                                       const SolverState &state,
                                       HyperParameters &params);

        virtual void adapt_parameters(const std::vector<Tensor> &particles,
                                      const std::vector<Tensor> &current_consensus,
                                      const std::vector<Tensor> &prev_consensus,
                                      const Tensor &weights,
                                      const SolverState &state,
                                      HyperParameters &params);

        virtual void on_consensus_evaluated(const std::vector<Tensor> & /*consensus*/, double /*energy*/) {}

        CBOResult run_loop();
    };
} // namespace involute::core