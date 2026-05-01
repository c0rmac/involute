#include "involute/core/base_solver.hpp"
#include "involute/core/math.hpp"
#include <algorithm>
#include <limits>
#include <iostream>
#include <sstream>

namespace involute::core {
    namespace {
        /**
         * @brief Helper to format vector parameters (lambda, delta) for the console.
         */
        std::string format_values(const std::vector<double> &values) {
            std::ostringstream oss;
            oss << "[";
            for (size_t i = 0; i < values.size(); ++i) {
                if (i > 0) oss << ", ";
                oss << values[i];
            }
            oss << "]";
            return oss.str();
        }
    }

    // ==============================================================================
    // Convergence Criteria Implementations
    // ==============================================================================

    MaxStepsCriterion::MaxStepsCriterion(int max_steps) : max_steps_(max_steps) {}

    bool MaxStepsCriterion::check(const SolverState &state) {
        return state.step >= max_steps_;
    }

    EnergyToleranceCriterion::EnergyToleranceCriterion(double tol, int min_steps)
        : tolerance_(tol), min_steps_(min_steps) {}

    bool EnergyToleranceCriterion::check(const SolverState &state) {
        if (state.step < min_steps_) return false;
        return std::abs(state.prev_energy - state.current_energy) <= tolerance_;
    }

    ConsensusToleranceCriterion::ConsensusToleranceCriterion(double tol, int min_steps)
        : tolerance_(tol), min_steps_(min_steps) {}

    bool ConsensusToleranceCriterion::check(const SolverState &state) {
        if (state.step < min_steps_ || !state.current_consensus || !state.prev_consensus) {
            return false;
        }

        double total_sq_distance = 0.0;
        const auto& curr = *state.current_consensus;
        const auto& prev = *state.prev_consensus;

        for (size_t i = 0; i < curr.size(); ++i) {
            Tensor diff = math::subtract(curr[i], prev[i]);
            total_sq_distance += math::to_double(math::sum(math::square(diff)));
        }

        return std::sqrt(total_sq_distance) <= tolerance_;
    }

    void CompositeCriterion::add(std::shared_ptr<ConvergenceCriterion> criterion) {
        criteria_.push_back(std::move(criterion));
    }

    bool CompositeCriterion::check(const SolverState &state) {
        for (const auto &criterion : criteria_) {
            if (criterion->check(state)) return true;
        }
        return false;
    }

    // ==============================================================================
    // BaseSolver Implementation
    // ==============================================================================

    BaseSolver::BaseSolver(SolverConfig config) : config_(std::move(config)) {
        if (!config_.convergence) {
            config_.convergence = std::make_shared<MaxStepsCriterion>(200);
        }
        config_.convergence->dimensional_normalisation_constant_ =
            Tensor(static_cast<float>(config_.d), config_.dtype);
    }

    Tensor BaseSolver::evaluate_objective(const std::vector<Tensor> &particles) {
        if (obj_product_) {
            // Product path: expand any 2D consensus components to [1, n, k] batch form
            if (!particles.empty() && particles[0].shape().size() == 2) {
                std::vector<Tensor> expanded(particles.size());
                for (size_t i = 0; i < particles.size(); ++i)
                    expanded[i] = math::expand_dims(particles[i], {0});
                return obj_product_->evaluate_batch(expanded);
            }
            return obj_product_->evaluate_batch(particles);
        }

        // Single-manifold path: unwrap the one-element vector
        Tensor p = particles[0];
        if (p.shape().size() == 2)
            p = math::expand_dims(p, {0});
        return obj_single_->evaluate_batch(p);
    }

    Tensor BaseSolver::compute_weights(const Tensor &costs,
                                       const std::vector<Tensor> &particles,
                                       const std::vector<Tensor> &prev_consensus,
                                       const SolverState &state,
                                       HyperParameters &params) {
        Tensor shifted_costs = math::subtract(costs, math::min(costs));

        if (config_.parameter_adapter) {
            return config_.parameter_adapter->compute_consensus_weights(
                state.step, shifted_costs, particles, prev_consensus, state.prev_energy, params
            );
        }

        Tensor neg_beta = math::multiply(shifted_costs, Tensor(static_cast<float>(-params.beta), DType::Float32));
        return math::exp(neg_beta);
    }

    void BaseSolver::adapt_parameters(const std::vector<Tensor> &particles,
                                      const std::vector<Tensor> &current_consensus,
                                      const std::vector<Tensor> &prev_consensus,
                                      const Tensor &weights,
                                      const SolverState &state,
                                      HyperParameters &params) {
        if (!config_.parameter_adapter) return;
        config_.parameter_adapter->adapt(
            state.step, particles, current_consensus, prev_consensus,
            state.current_energy, state.prev_energy, weights, params
        );
    }

    CBOResult BaseSolver::solve(ObjectiveFunction *obj) {
        obj_single_  = obj;
        obj_product_ = nullptr;
        if (config_.parameter_adapter) {
            int attempts = config_.parameter_adapter->max_attempts_;
            CBOResult best_result;
            double best_optimum = std::numeric_limits<double>::infinity();

            for (int i = 0; i < attempts; i++) {
                CBOResult result = run_loop();
                if (result.min_energy < best_optimum) {
                    best_optimum = result.min_energy;
                    best_result = result;
                }
            }
            return best_result;
        }
        return run_loop();
    }

    CBOResult BaseSolver::solve(ProductObjectiveFunction *obj) {
        obj_product_ = obj;
        obj_single_  = nullptr;
        if (config_.parameter_adapter) {
            int attempts = config_.parameter_adapter->max_attempts_;
            CBOResult best_result;
            double best_optimum = std::numeric_limits<double>::infinity();

            for (int i = 0; i < attempts; i++) {
                CBOResult result = run_loop();
                if (result.min_energy < best_optimum) {
                    best_optimum = result.min_energy;
                    best_result = result;
                }
            }
            return best_result;
        }
        return run_loop();
    }

    CBOResult BaseSolver::run_loop() {
        std::vector<Tensor> particles = initialize_particles();
        std::vector<Tensor> prev_consensus;
        std::vector<Tensor> last_consensus;

        CBOResult res;
        SolverState state = {
            0, std::numeric_limits<double>::infinity(), 1e9, nullptr, nullptr, config_.d, config_.N
        };

        HyperParameters params = config_.params;
        std::vector<StepRecord> debug_history;

        while (true) {
            // 1. EVALUATION
            Tensor costs = evaluate_objective(particles);

            // 2. WEIGHTING
            Tensor weights = compute_weights(costs, particles, prev_consensus, state, params);
            weights = math::divide(weights, math::sum(weights));

            // 3. CONSENSUS
            last_consensus = compute_consensus(particles, weights);
            state.current_consensus = &last_consensus;

            Tensor current_energy_tensor = evaluate_objective(last_consensus);
            state.current_energy = math::to_double(current_energy_tensor);

            on_consensus_evaluated(last_consensus, state.current_energy);

            if (state.step > 0) state.prev_consensus = &prev_consensus;

            // 4. PARAMETER ADAPTATION
            adapt_parameters(particles, last_consensus, prev_consensus, weights, state, params);

            // Debugging / Logging
            if (std::find(config_.debug.begin(), config_.debug.end(), Debugger::History) != config_.debug.end()) {
                debug_history.push_back({
                    state.step, state.current_energy, params.beta, params.lambda_at(0), params.delta_at(0)
                });
            }

            auto log_check = std::find(config_.debug.begin(), config_.debug.end(), Debugger::Log);
            if (log_check != config_.debug.end()) {
                // COMPREHENSIVE LOGGING: Includes all hyperparameters and current state
                std::cout << "\r[Involute] Step: " << state.step
                          << " | Energy: " << state.current_energy
                          << " | beta: " << params.beta
                          << " | lambda: " << format_values(params.lambda)
                          << " | delta: " << format_values(params.delta)
                          << std::flush;
            }

            // 5. CONVERGENCE CHECK
            if (config_.convergence->check(state) &&
                (!config_.parameter_adapter || config_.parameter_adapter->ready_to_converge(params))) {
                break;
            }

            if (state.current_energy < 0.01) break;

            // 6. STEP
            particles = step(particles, last_consensus, params);

            for (auto& p : particles) math::eval(p);

            prev_consensus = last_consensus;
            state.prev_energy = state.current_energy;
            state.step++;
        }

        res.converged = true;
        res.final_particles = particles;
        res.final_consensus = last_consensus;
        res.min_energy = state.current_energy;
        res.iterations_run = state.step;
        res.history = std::move(debug_history);

        if (config_.parameter_adapter) config_.parameter_adapter->reset();

        return res;
    }
} // namespace involute::core