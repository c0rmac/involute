#pragma once

#include "involute/core/parameter_adapter.hpp"
#include "involute/core/base_solver.hpp"
#include "involute/core/math.hpp"
#include <cmath>
#include <algorithm>
#include <iostream>

namespace involute::solvers {

class SODParameterAdapter : public core::ParameterAdapter {
private:
    int d_;
    int d_man_;

    // Dynamic Landscape & Geometric State
    double q_ = 0.0;              // Current local sharpness gap
    double r_;              // Current success radius

    // Initial Constants (Used for quadratic scaling)
    double q_initial_ = 0.0;
    double r_initial_;

    // Weight Clamping Thresholds for Beta
    double min_weight_;
    double max_weight_;
    double beta_max_;
    double beta_growth_rate_;

    int wait = 0;
    int patience = 4; // 1 d=20 -> 3; d=10 -> 4;
    double min_energy_ = std::numeric_limits<double>::max();
    double min_energy_lambda_ = 0;
    double previous_energy_ = std::numeric_limits<double>::max();

    double min_beta = 0.0;

    double relative_contraction_rate_ = 0.0;

public:
    /**
     * @brief Constructor that solves for 'q' to make initial_beta and initial_lambda valid.
     */
    SODParameterAdapter(int d,
                        double initial_beta,
                        double initial_lambda,
                        double delta,
                        double relative_contraction_rate,
                        double r = 350, // d=5 -> 350, d=10 -> 800
                        double min_weight = 0.8,
                        double max_weight = 0.9,
                        double b_max = 1e15,
                        double beta_growth = 1.15)
        : d_(d), relative_contraction_rate_(relative_contraction_rate), r_(r), min_weight_(min_weight), max_weight_(max_weight),
          beta_max_(b_max), beta_growth_rate_(beta_growth) {

        d_man_ = d * d;
        double B = d_man_ / 2.0;

        // 1. Determine Initial R (Search Horizon)
        double sigma_sq = (delta * delta) / (2.0 * initial_lambda);
        double R_0 = std::sqrt(d_man_ * sigma_sq);
        double A = (r_ + R_0) / (delta * delta);

        // 2. Precompute initial log volume of success radius r on SO(d)
        double log_pi_term = B * std::log(M_PI);
        double log_gamma_term = std::lgamma(B + 1.0);
        double log_vol_r_initial = log_pi_term - log_gamma_term + (d_man_ * std::log(r_));

        // 3. Solve for initial q such that min_beta_required == initial_beta
        double lambda_crit = B / A;
        double V_min = A * lambda_crit - B * std::log(lambda_crit);
        double log_noise_term = B * std::log(M_PI * delta * delta);

        //q_ = (V_min - log_vol_r_initial + log_noise_term) / initial_beta;

        // Store baselines for the quadratic physical scaling
        //q_initial_ = q_;
        r_initial_ = r_;

        max_attempts_ = 1;
    }

    double old_beta_ = 0.0;

    void adapt(int step,
           const Tensor& particles,
           const Tensor& current_consensus,
           const Tensor& prev_consensus,
           double current_energy,
           double prev_energy,
           const Tensor &weights,
           core::HyperParameters& params) override {
        //wait = 0;

        // ==========================================
        // PHASE 1: ADAPT BETA (Focusing)
        // ==========================================
        Tensor max_weight_tensor = math::max(weights);
        double max_w = math::to_double(max_weight_tensor);

        if (max_w < min_weight_) {
            //params.beta = std::min(params.beta * beta_growth_rate_, beta_max_);
        } else if (max_w > max_weight_) {
            //params.beta = std::max(params.beta * 0.99, 1e-6);
            //params.beta = 1.0;
        }
        //params.beta = beta_growth_rate_ * params.beta;
        if (previous_energy_ < current_energy) {
            wait += 1;

            if (wait > patience) {
                float N = particles.shape()[0];
                float d = particles.shape()[1];
                float growth_rate = 1.0 + relative_contraction_rate_ * (2.0 * std::log(N) / (d * d) ); //
                //float growth_rate = 1.1;

                params.lambda = params.lambda * growth_rate;
                wait = 0;
            }
            //wait = std::min(wait + 1, 5000);
            //wait = std::min(wait + 1, 1);
        } else if (std::abs(previous_energy_ - current_energy) < 1e-6 and current_energy > min_energy_) {
            //min_energy_lambda_ = min_energy_lambda_ * 0.9;
            //params.lambda = 1.0;
        } else {
            min_energy_ = current_energy;
            min_energy_lambda_ = params.lambda;
        }

        /*
        if (wait > patience) {
            //params.lambda = 1.0 * params.lambda;
            //std::cout << "lambda = " << params.lambda;
            //return;
        }

        // ==========================================
        // PHASE 2: CALCULATE SPREAD & SHRINK 'r'
        // ==========================================
        double B = d_man_ / 2.0;

        // Calculate true physical spread (RMS Horizon) based on CURRENT lambda
        double sigma_tilde_sq = (params.delta * params.delta) / (2.0 * params.lambda);
        double R_empirical = std::sqrt(d_man_ * sigma_tilde_sq);
        //double R_empirical = 5.0;


        // PROACTIVE GEOMETRIC DECAY:
        // This entirely replaces the paralyzing 'solve_coupled_r' root-finder.
        // By keeping the target 'r' strictly linked to (but slightly larger than)
        // the current swarm size, we force the penalty A to continuously shrink.
        //r_ = 5.0;
        //r_ = 0.5 * estimate_empirical_r(particles, current_consensus, weights);
        //r_ = 0.1;
        //r_ = 1.001 * r_;

        // Update empirical q based on the quadratic landscape assumption
        double q = 1.0 * estimate_q_via_stddev(weights, params.beta, r_, R_empirical);

        if (!std::isinf(q) and !std::isnan(q)) {
            q_ = 1.0*q;
        } else {
            //double q_max = 0.5 * (d_ / params.beta) * (1 + std::log((r_ * r_) / (sigma_tilde_sq)));
            //q_ = q_;
            //q_ = 1.0*q;
            //q_ = q_max;
        }

        // ==========================================
        // PHASE 3: NORMALIZED DRIFT SOLVER (Dimension-Free)
        // ==========================================
        // Normalize previous drift to safely calculate bounds without Gamma(51) explosions
        double u_prev = params.lambda / B;

        // The linear penalty (A) uses the newly shrunk r_ and the current R_empirical
        double A = (r_ + R_empirical) / (params.delta * params.delta);

        // Dimension-Free Budget (C_norm) - Perfectly stable for any dimension
        // The massive Gamma function and the pi constants have been algebraically cancelled out.
        double C_norm = (params.beta * q_) / B + 1.0 + 2.0 * std::log(r_) - 2.0 * std::log(params.delta);

        // The curve minimum for f(u) = A*u - log(u)
        double u_opt = 1.0 / A;
        double min_curve_val = A * u_opt - std::log(u_opt);

        double u_new = u_prev;

        if (C_norm <= min_curve_val) {
            // Budget is tight; park at the absolute most stable drift value to wait for beta
            u_new = std::max(1.0 * u_prev, 1.0 * u_opt);
            //u_new = 1.0 * u_opt;
        } else {
            // Budget is healthy; use Newton-Raphson to find maximum safe shrinkage
            double u_guess = u_opt * 2.0;

            for (int i = 0; i < 20; ++i) {
                double f_val = A * u_guess - std::log(u_guess) - C_norm;
                double df_val = A - (1.0 / u_guess);

                // Prevent division by zero if derivative flattens out
                if (std::abs(df_val) < 1e-8) break;

                double nr_step = f_val / df_val;
                u_guess -= nr_step;

                if (std::abs(nr_step) < 1e-5) break;
            }

            // 95% safety buffer inside the valid contractive regime
            // We enforce that u_new never drops below u_prev to prevent backward slipping
            u_new = std::max(u_prev, 1.0 * u_guess);
            //u_new = 1.0 * u_guess;
        }

        // ==========================================
        // PHASE 4: COMMIT PHYSICAL UPDATE
        // ==========================================
        // Scale the normalized variable back into the true physical drift
        // a
        double new_lambda = B * u_new;
        params.lambda = new_lambda;*/

        previous_energy_ = current_energy;
    }

    /**
     * @brief Dynamically estimates the target success radius (r) by calculating
     * the spatial distance between the weighted consensus and the current best particle.
     * Keeps all heavy operations on the backend device until the final scalar extraction.
     */
    double estimate_empirical_r(const Tensor& particles,
                                const Tensor& current_consensus,
                                const Tensor& weights) const {

        // 1. Find the index of the best particle on the device
        Tensor best_idx_tensor = math::argmax(weights, 0);

        // 2. Extract the best particle directly from the batch using gather
        // Assuming 'particles' has shape [N, d, d] or [N, d], this plucks the best one.
        Tensor best_particle = math::gather(particles, best_idx_tensor, 0);

        // 3. Calculate the spatial difference (vector or matrix distance)
        Tensor diff = math::subtract(best_particle, current_consensus);

        // 4. Calculate the Frobenius / Euclidean norm squared: sum(diff^2)
        Tensor sq_diff = math::square(diff);
        Tensor sum_sq = math::sum(sq_diff); // Empty axes defaults to global scalar sum

        // 5. Calculate the root to get the true distance
        Tensor r_tensor = math::sqrt(sum_sq);

        // 6. Pull the final scalar back to the CPU
        double r_est = math::to_double(r_tensor);

        // 7. Apply a strict mathematical floor to prevent log(0) explosions
        // if the consensus lands perfectly on the best particle.
        return r_est;
    }

    /**
 * @brief Estimates the local sharpness gap (q) by calculating the standard
 * deviation of the swarm's relative energies, reverse-engineered from weights.
 */
    double estimate_q_via_stddev(const Tensor& weights,
                                 double beta,
                                 double r_target,
                                 double R_empirical) const {
        // 1. Get the maximum weight to act as our baseline (E = 0)
        Tensor w_max = math::max(weights);

        // 2. Safely compute the log of all weights
        Tensor safe_weights = math::add(weights, Tensor(0, DType::Float32));
        Tensor log_w = math::log(safe_weights);

        // 3. Reconstruct the relative energy tensor
        // E_rel_i = (log(w_max) - log(w_i)) / beta
        Tensor E_rel = math::divide(math::subtract(math::log(w_max), log_w), Tensor(beta, DType::Float32));

        // 4. Calculate the Standard Deviation of the energies
        Tensor E_mean = math::mean(E_rel);
        Tensor E_diff = math::subtract(E_rel, E_mean);
        Tensor E_diff_sq = math::multiply(E_diff, E_diff); // Element-wise square
        Tensor E_var = math::mean(E_diff_sq);

        Tensor E_stddev = math::sqrt(E_var);

        // 5. Scale the empirical gap to the target success radius
        // We cap the ratio at 1.0 to prevent artificial inflation if r > R.
        Tensor ratio_sq = math::divide(Tensor(r_target * r_target, DType::Float32), Tensor(R_empirical * R_empirical, DType::Float32));
        double scaling_factor = math::to_double(ratio_sq);
        Tensor q_emp = math::multiply(E_stddev, Tensor(scaling_factor, DType::Float32));

        // Numerical floor to guarantee the Budget equation never crashes
        return math::to_double(q_emp);
    }

    /**
     * @brief Calculates a highly accurate, noise-resilient local sharpness gap (q)
     * by extracting the mean energy gap from the swarm's Boltzmann distribution.
     */
    double estimate_empirical_q(const Tensor& weights, double beta, double r_target, double R_empirical) const {
        // 1. Get the maximum weight to represent the local minimum
        double w_max = math::to_double(math::max(weights));

        // 2. Safely calculate the mean of the log weights
        // (Assuming math::log applies element-wise and math::mean averages the tensor)
        // We add a tiny epsilon to prevent log(0) if any weights underflowed to 0.
        Tensor safe_weights = weights;
        Tensor log_w_tensor = math::log(safe_weights);
        double mean_log_w = math::to_double(math::mean(log_w_tensor));

        // 3. Calculate the exact average energy gap of the swarm
        double delta_E_avg = (1.0 / beta) * (std::log(w_max) - mean_log_w);

        // 4. Scale the empirical gap to the target success radius
        // We use std::min(1.0, ...) to ensure that if the swarm is already inside r,
        // we don't accidentally inflate q past the actual measured energy gap.
        double scaling_factor = std::min(1.0, (r_target * r_target) / (R_empirical * R_empirical));
        double q_emp = delta_E_avg * scaling_factor;

        // 5. Provide a numerical floor so the Budget equation (C_norm) never crashes
        return q_emp;
    }

    /**
     * @brief Solves the 1D transcendental root equation for 'r' to perfectly
     * couple the success radius to the newly computed drift (lambda).
     */
    double solve_coupled_r(double beta, double lambda, double delta) const {
        double B = d_man_ / 2.0;

        // The new physical search horizon based on the updated lambda
        double R_new = std::sqrt(d_man_ * (delta * delta) / (2.0 * lambda));

        // Equation: C1*r^2 - C2*r + C3*log(r) - C4 = 0
        double C1 = (beta * q_initial_) / (r_initial_ * r_initial_);
        double C2 = lambda / (delta * delta);
        double C3 = d_man_; // (which is 2 * B)

        // The pi constants perfectly cancel out during derivation!
        double C4 = (C2 * R_new) - B * std::log(lambda / (delta * delta)) + std::lgamma(B + 1.0);

        // Newton-Raphson Solver for 'r'
        double r_guess = r_; // Start guess from the current radius

        for (int i = 0; i < 100; ++i) {
            double f_val = C1 * (r_guess * r_guess) - C2 * r_guess + C3 * std::log(r_guess) - C4;
            double df_val = 2.0 * C1 * r_guess - C2 + (C3 / r_guess);

            // Prevent division by zero if derivative flattens out
            if (std::abs(df_val) < 1e-8) break;

            double step = f_val / df_val;

            if (std::isnan(r_guess - step)) {
                break;
            }

            r_guess -= step;
            r_guess = std::max(1e-6, r_guess);

            if (std::abs(step) < 1e-6) break;
        }

        // Clamp to ensure r doesn't explode past the user's initial assumption
        // or collapse to true zero (causing log(0) errors later).
        //return std::clamp(r_guess, 1e-6, r_initial_);
        return r_guess;
    }

    void reset() override {
        // Reset state for a new optimization run
        r_ = r_initial_;
        q_ = q_initial_;
    }

    bool ready_to_converge(core::HyperParameters params) override {
        // High beta is a strong indicator that the system has funneled deep into the minimum
        return params.beta > -1;
    }
};

} // namespace involute::solvers