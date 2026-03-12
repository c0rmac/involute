/**
 * @file so_parameter_adapter.hpp
 * @brief Zero-cost parameter adaptation based on Consensus Energy Volatility.
 * Dynamically decouples spatial annealing (lambda) from informational
 * annealing (beta) using live Boltzmann weight analysis.
 */

#pragma once

#include "involute/core/parameter_adapter.hpp"
#include "involute/core/math.hpp"
#include <deque>
#include <cmath>
#include <iostream>

namespace involute::solvers {

class SOParameterAdapter : public core::ParameterAdapter {
private:
    std::deque<double> energy_history_;
    double min_energy_ = std::numeric_limits<double>::infinity();
    int max_wait_ = 2;
    double lambda_mult_ = 1.0;
    double beta_mult_ = 1.0;
    double beta_decr_ = 1.0;
    double max_weight_ = 1.0;
    double min_weight_ = 0.0;
    int current_wait_ = 0;

public:
    explicit SOParameterAdapter(int max_attempts = 1, int max_wait = 2, double lambda_mult = 1.0, double beta_mult = 1.0, double beta_decr = 1.0, double max_weight = 1.0, double min_weight = 0.0)
        : max_wait_(max_wait), lambda_mult_(lambda_mult), beta_mult_(beta_mult), beta_decr_(beta_decr), max_weight_(max_weight), min_weight_(min_weight) {
        max_attempts_ = max_attempts;
    }

    ~SOParameterAdapter() override = default;

    auto adapt(
        int step,
        const Tensor &particles,
        const Tensor &current_consensus,
        const Tensor &prev_consensus,
        double current_energy,
        double prev_energy,
        const Tensor &weights, // Live Gibbs distributions
        core::HyperParameters &params
    ) -> void override {

        // If we found a new low, reset the patience counter.
        // We do NOT change parameters while we are successfully descending.
        if (current_energy < min_energy_) {
            min_energy_ = current_energy;
            current_wait_ = 0;
        }
        // If we are stagnating/jittering...
        else {
            current_wait_++;

            if (max_wait_ < current_wait_) {
                // ============================================================
                // 1. SPATIAL ANNEALING (Lambda)
                // ============================================================
                // Unconditionally shrink the physical variance of the particle cloud.
                //if (step > 2000) {
                    params.lambda *= lambda_mult_;
                //}

                // ============================================================
                // 2. INFORMATIONAL ANNEALING (Beta)
                // ============================================================
                // Extract the maximum weight to diagnose the swarm's intelligence state.
                double max_weight = math::to_double(math::max(weights));

                if (max_weight > max_weight_) {
                    // THE REHEATING ZONE: Swarm collapsed and got stuck!
                    // Decrease beta to flatten the probability distribution. This
                    // breaks the argmax hold and forces the consensus to listen to
                    // other particles, even while the spatial variance shrinks.
                    params.beta *= beta_decr_;
                    //params.lambda *= lambda_mult_;
                }
                else if (max_weight < min_weight_) {
                    // THE COOLING ZONE: Swarm is diverse but unable to make a decision.
                    // Increase beta to make the selection greedier and force convergence.
                    params.beta *= beta_mult_;
                    //params.lambda *= 0.99;
                }
                // THE GOLDILOCKS ZONE (0.75 to 0.95):
                // The swarm is perfectly balancing exploration and exploitation.
                // We hold beta steady and let lambda do the work.
            }
        }
    }

    void reset() override {
        current_wait_ = 0;
        min_energy_ = std::numeric_limits<double>::infinity();
    }

    bool ready_to_converge(core::HyperParameters params) override {
        // Because we removed the 5000.0 cap, beta can safely race to 1e8
        // once the energy landscape goes completely flat at the global minimum.
        //return (params.lambda > 1e10);
        return (params.beta > 5000000);
    }
};

} // namespace involute::solvers