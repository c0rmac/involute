/**
 * @file base_solver.hpp
 * @brief Core Consensus-Based Optimization (CBO) logic for Involute.
 * * This file defines the BaseSolver class and the modular ConvergenceCriterion
 * system. The solver is designed to optimize cost functions over matrix
 * manifolds (like SO(d) or Stiefel) by simulating a particle system that
 * gravitates toward a weighted consensus point.
 */

#pragma once

#include "involute/core/tensor.hpp"
#include "involute/core/objective.hpp"
#include "involute/core/result.hpp"
#include "involute/core/math.hpp"
#include "involute/core/parameter_adapter.hpp"
#include <iostream>
#include <memory>
#include <vector>
#include <cmath>


namespace involute::core {
    /**
     * @struct HyperParameters
     * @brief The dynamic mathematical parameters that govern the CBO SDE.
     */
    struct HyperParameters {
        double beta; ///< Focus parameter (greediness)
        double lambda; ///< Drift coefficient (pull toward consensus)
        double delta; ///< Diffusion coefficient (noise amplitude)
    };

    /**
     * @struct SolverState
     * @brief Captures a snapshot of the solver's metrics at any given iteration.
     * * Used primarily by ConvergenceCriterion implementations to decide if the
     * optimization should terminate.
     */
    struct SolverState {
        int step{}; ///< Current iteration index
        double current_energy{}; ///< Best objective value in the current particle cloud
        double prev_energy{}; ///< Best objective value from the previous iteration
        const Tensor *current_consensus{}; ///< Pointer to current M_amb matrix on GPU
        const Tensor *prev_consensus{}; ///< Pointer to previous M_amb matrix on GPU

        int d;
        int N;
    };

    // ==============================================================================
    // CONVERGENCE STRATEGIES (MODULAR STOPPING CONDITIONS)
    // ==============================================================================

    /**
     * @class ConvergenceCriterion
     * @brief Interface for defining when the solver should stop.
     * * By implementing this interface, users can define custom stopping logic based
     * on time, energy, or geometric convergence of the particles.
     */
    class ConvergenceCriterion {
    public:
        Tensor dimensional_normalisation_constant_;

        virtual ~ConvergenceCriterion() = default;

        /**
         * @brief Evaluates the current state to determine if convergence is reached.
         * @return true if the solver should halt, false otherwise.
         */
        virtual bool check(const SolverState &state) = 0;
    };

    /** @brief Halts the solver once a maximum iteration count is reached. */
    class MaxStepsCriterion : public ConvergenceCriterion {
        int max_steps_;

    public:
        explicit MaxStepsCriterion(int max_steps) : max_steps_(max_steps) {
        }

        bool check(const SolverState &state) override { return state.step >= max_steps_; }
    };

    /** @brief Halts when the change in the minimum energy falls below a threshold. */
    class EnergyToleranceCriterion : public ConvergenceCriterion {
        double tolerance_;
        int min_steps_;

    public:
        explicit EnergyToleranceCriterion(double tol, int min_steps = 50)
            : tolerance_(tol), min_steps_(min_steps) {
        }

        bool check(const SolverState &state) override {
            if (state.step < min_steps_) return false;
            return std::abs(state.prev_energy - state.current_energy) <= tolerance_;
        }
    };

    /** @brief Halts when the consensus point (Fréchet mean) stops moving in Frobenius norm. */
    class ConsensusToleranceCriterion : public ConvergenceCriterion {
        double tolerance_;
        int min_steps_;

    public:
        explicit ConsensusToleranceCriterion(double tol, int min_steps = 50)
            : tolerance_(tol), min_steps_(min_steps) {
        }

        bool check(const SolverState &state) override {
            if (state.step < min_steps_ || !state.current_consensus || !state.prev_consensus) {
                return false;
            }

            // Geometric distance: || C_current - C_prev ||_F
            Tensor diff = math::subtract(*state.current_consensus, *state.prev_consensus);
            Tensor sq_diff = math::square(diff);
            Tensor sum_sq = math::sum(sq_diff);
            Tensor normalised = math::divide(sum_sq, dimensional_normalisation_constant_);

            // Return the scalar result to CPU for comparison
            double distance = math::to_double(math::sqrt(sum_sq));
            return distance <= tolerance_;
        }
    };

    /** @brief Combines multiple criteria; stops if ANY of them are satisfied. */
    class CompositeCriterion : public ConvergenceCriterion {
        std::vector<std::shared_ptr<ConvergenceCriterion> > criteria_;

    public:
        void add(std::shared_ptr<ConvergenceCriterion> criterion) { criteria_.push_back(criterion); }

        bool check(const SolverState &state) override {
            for (const auto &criterion: criteria_) {
                if (criterion->check(state)) return true;
            }
            return false;
        }
    };

    // ==============================================================================
    // SOLVER CONFIGURATION & BASE CLASS
    // ==============================================================================

    // Add this near the top, perhaps just above SolverConfig
    enum class WeightingScheme {
        Exponential, ///< Standard Boltzmann exponential decay
        Algebraic ///< Heavy-tailed algebraic distribution (p=2)
    };

    enum class Debugger {
        History,
        Log
    };

    /**
     * @struct SolverConfig
     * @brief Hyperparameters for the Consensus-Based Optimization.
     */
    struct SolverConfig {
        int N; ///< Total number of particles in the swarm
        int d; ///< Manifold dimension (e.g., d for SO(d))
        /*
        double beta;    ///< Focus parameter (higher beta concentrates weights on low-energy particles)
        double lambda;  ///< Drift coefficient (rate of convergence toward consensus)
        double delta;   ///< Diffusion coefficient (magnitude of anisotropic noise)
        double h;       ///< Numerical step size (discretization of the SDE)
        */
        HyperParameters params;

        involute::DType dtype;

        std::shared_ptr<ConvergenceCriterion> convergence; ///< Swappable stopping logic

        std::shared_ptr<ParameterAdapter> parameter_adapter;

        WeightingScheme weighting_scheme = WeightingScheme::Algebraic;

        std::vector<Debugger> debug;
    };

    enum SolverConfigType {
        Aggressive,
        Safe,
        ExtraSafe
    };

    /**
     * @class BaseSolver
     * @brief Abstract base class for all manifold-specific CBO solvers.
     * * Implements the hardware-agnostic CBO loop. Manifold-specific operations like
     * initialization, consensus calculation, and retraction are delegated to
     * derived classes via pure virtual functions.
     */
    class BaseSolver {
    protected:
        SolverConfig config_;

    public:
        explicit BaseSolver(SolverConfig config) : config_(std::move(config)) {
            if (!config_.convergence) {
                config_.convergence = std::make_shared<MaxStepsCriterion>(200);
            }
            config_.convergence->dimensional_normalisation_constant_ = Tensor(config_.d * config_.d, config_.dtype);
        }

        virtual ~BaseSolver() = default;

        /**
         * @brief The core optimization loop.
         * @param obj A pointer to the objective function to be minimized.
         * @return A CBOResult containing the optimized matrix and convergence stats.
         */
        virtual CBOResult solve(ObjectiveFunction *obj) {
            auto parameter_adapter = config_.parameter_adapter;
            if (parameter_adapter) {
                int attempts = config_.parameter_adapter->max_attempts_;

                CBOResult best_result;
                double best_optimum = std::numeric_limits<double>::infinity();

                for (int i = 0; i < attempts; i++) {
                    CBOResult result = run_loop(obj);
                    double min_energy = result.min_energy;

                    if (min_energy < best_optimum) {
                        best_optimum = min_energy;
                        best_result = result;
                    }
                }

                return best_result;
            } else {
                return run_loop(obj);
            }
        }

    protected:
        // ====================================================================
        // MANIFOLD INTERFACE (To be implemented by SO(d), Stiefel, etc.)
        // ====================================================================

        virtual Tensor initialize_particles() = 0;

        virtual Tensor compute_consensus(const Tensor &p, const Tensor &w) = 0;

        virtual Tensor generate_noise() = 0;

        virtual Tensor step(const Tensor &p, const Tensor &c, const Tensor &n, HyperParameters params) = 0;

        /**
         * @brief Verifies that a batch of particles strictly adheres to the manifold's geometry.
         * @param p The batched tensor of particles to check.
         * @return true if the constraints are mathematically satisfied, false otherwise.
         */
        virtual bool check_manifold_constraint(const Tensor &p) const = 0;

        CBOResult run_loop(ObjectiveFunction *obj) {
            // Step 0: Initialize swarm on the manifold (e.g., SO(d))
            Tensor particles = initialize_particles();
            //std::cout << "[Involute] Initializing particles" << particles << std::endl;
            /*
                    if (!check_manifold_constraint(particles)) {
                        throw std::runtime_error(
                            "[Involute] Fatal: Initialized particles do not satisfy the required manifold constraints."
                        );
                    }
                    */

            Tensor prev_consensus;

            CBOResult res;
            SolverState state = {
                0, std::numeric_limits<double>::infinity(), 1e9, nullptr, nullptr, config_.d, config_.N
            };

            HyperParameters params = config_.params;

            std::vector<StepRecord> debug_history;

            while (true) {
                // 1. EVALUATION
                // Run the user's objective in parallel for all N particles on GPU
                Tensor costs = obj->evaluate_batch(particles);

                // 2. BOLTZMANN WEIGHTING
                // Find global min for numerical stability, then calculate w_i = exp(-beta * cost_i)
                Tensor min_cost_tensor = math::min(costs);

                Tensor shifted_costs = math::subtract(costs, min_cost_tensor);

                Tensor weights;
                if (config_.parameter_adapter) {
                    // The adapter modifies config_.params (lambda, delta, beta, h) in-place.
                    // Note: On step 0, prev_consensus is an empty tensor.
                    weights = config_.parameter_adapter->compute_consensus_weights(
                        state.step,
                        shifted_costs,
                        particles,
                        prev_consensus,
                        state.prev_energy,
                        params
                    );
                } else {
                    // Find global min for numerical stability, then calculate w_i = exp(-beta * cost_i)
                    Tensor min_cost_tensor = math::min(costs);

                    Tensor shifted_costs = math::subtract(costs, min_cost_tensor);
                    Tensor neg_beta = math::multiply(shifted_costs, Tensor(-params.beta, DType::Float32));
                    weights = math::exp(neg_beta);
                }

                // 3. CONSENSUS CALCULATION
                // Calculate the weighted Fréchet mean of the swarm
                Tensor current_consensus = compute_consensus(particles, weights);
                state.current_consensus = &current_consensus;

                Tensor current_energy = obj->evaluate_batch(math::expand_dims(current_consensus, {0}));
                state.current_energy = math::to_double(current_energy);

                if (state.step > 0) state.prev_consensus = &prev_consensus;

                // ---------------------------------------------------------
                // PARAMETER ADAPTATION
                // ---------------------------------------------------------
                if (config_.parameter_adapter) {
                    // The adapter modifies config_.params (lambda, delta, beta, h) in-place.
                    // Note: On step 0, prev_consensus is an empty tensor.
                    config_.parameter_adapter->adapt(
                        state.step,
                        particles,
                        current_consensus,
                        prev_consensus,
                        state.current_energy,
                        state.prev_energy,
                        weights,
                        params
                    );
                }

                auto history_check = std::find(config_.debug.begin(), config_.debug.end(), Debugger::History);

                if (history_check != config_.debug.end()) {
                    debug_history.push_back({
                        state.step,
                        state.current_energy,
                        params.beta,
                        params.lambda,
                        params.delta
                    });
                }

                // 4. CONVERGENCE CHECK
                auto log_check = std::find(config_.debug.begin(), config_.debug.end(), Debugger::Log);
                if (log_check != config_.debug.end()) {
                    std::cout << "\r[Involute] At step " << state.step
                            << " | Current Energy: " << state.current_energy << " | Current lambda: " << params.lambda
                            << " | Current beta: " << params.beta << std::flush;
                }

                if (config_.convergence->check(state) and config_.parameter_adapter and config_.parameter_adapter->ready_to_converge(params)) {
                    if (log_check == config_.debug.end()) {
                        std::cout << "[Involute] Converged at step " << state.step
                                << " | Final Energy: " << state.current_energy << std::endl;
                    }
                    break;
                }

                if (state.current_energy < 0.01) {
                    res.iterations_run = state.step;
                    break;
                }

                // 5. STOCHASTIC STEP & RETRACTION
                // Generate tangent-space noise and move particles toward consensus
                Tensor noise = generate_noise();
                particles = step(particles, current_consensus, noise, params);

                math::eval(particles);

                // 6. UPDATE STATE
                prev_consensus = current_consensus;
                state.prev_energy = state.current_energy;
                state.step++;
            }

            res.converged = true;
            res.final_particles = particles;
            res.final_consensus = prev_consensus;
            res.min_energy = state.current_energy;
            res.iterations_run = state.step;
            res.history = std::move(debug_history);

            config_.parameter_adapter->reset();

            return res;
        }
    };
} // namespace involute::core
