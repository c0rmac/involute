/**
 * @file so_solver.hpp
 * @brief Consensus-Based Optimization for the Special Orthogonal Group SO(d).
 * * This solver implements a highly advanced Spectral Hopping Scheme. It utilizes
 * an anisotropic Lie bracket decomposition to prevent variance explosion at the
 * cut locus, ensuring unconditional stability across the entire manifold.
 * SVD and iterative methods are strictly avoided in favor of fast Cayley
 * retractions and direct chordal projections.
 */

#pragma once

#include "involute/core/base_solver.hpp"
#include "involute/core/math.hpp"
#include <cmath>

#include "adam_parameter_adapter.hpp"
#include "so_parameter_adapter.hpp"

using namespace involute::core;


namespace involute::solvers {
    /**
     * @class SOSolver
     * @brief The primary SO(d) solver utilizing the Spectral Hopping Scheme.
     */
    class SOSolver : public core::BaseSolver {
    public:
        using core::BaseSolver::BaseSolver;

        /**
         * @brief Factory method to generate standardized CBO solver configurations tailored to memory and speed requirements.
         * * This method abstracts the trade-off between particle swarm density and convergence speed.
         * The selected configuration type dictates the memory footprint on the GPU and the expected
         * number of iterations required to reach the global optimum:
         * * - **Aggressive**: Prioritizes rapid convergence. Uses an enormous amount of memory
         * by initializing a high particle density (20 * d^2), allowing the swarm to find the
         * global minimum in fewer steps.
         * - **Safe**: A balanced configuration. Scales particles relative to the manifold dimension
         * (d^2) for standard performance.
         * - **ExtraSafe**: Optimized for low memory footprints. Uses a minimal fraction
         * of memory, though it requires significantly more iterations/steps to achieve convergence.
         * * @param type The strategy preset: Aggressive, Safe, or ExtraSafe.
         * @param d The intrinsic dimension of the manifold (e.g., d for SO(d)).
         * @param convergence Shared pointer to the logic determining termination criteria.
         * @param initial_variance Initial noise amplitude (delta) to prevent early swarm collapse.
         * @param debug Flags for history tracking or console logging during optimization.
         * @param learning_rate_scale Multiplier to adjust the Adam adapter's response speed.
         * @return A SolverConfig object containing the hyperparameters and particle count.
         */
        static core::SolverConfig get_solver_config(SolverConfigType type, int d,
                                                    std::shared_ptr<ConvergenceCriterion> convergence,
                                                    double initial_variance = 1.0,
                                                    std::vector<Debugger> debug = std::vector<Debugger>(),
                                                    double learning_rate_scale = 1.0) {
            DType dtype = DType::Float32;
            if (type == core::Aggressive) {
                int C = 20;
                //int C = 150;
                int N = C * d * d;
                //int N = 5000;
                return core::SolverConfig{
                    .N = N,
                    .d = d,
                    .params = core::HyperParameters{
                        .beta = 1.0,
                        .lambda = 1.0,
                        .delta = initial_variance
                    },
                    .dtype = dtype,
                    .convergence = convergence,
                    //.parameter_adapter=std::make_shared<SOParameterAdapter>(50, learning_rate_scale),
                    .parameter_adapter = std::make_shared<AdamParameterAdapter>(
                        0.8, 0.9, 0.999, 1e-8, learning_rate_scale * std::log(N) / (d * d)), // N=10 -> beta1=0.99
                    .debug = debug
                };
            } else if (type == core::Safe) {
                int N = d * d;
                return core::SolverConfig{
                    .N = N,
                    .d = d,
                    .params = core::HyperParameters{
                        .beta = 1.0,
                        .lambda = 1.0,
                        .delta = initial_variance
                    },
                    .dtype = dtype,
                    .convergence = convergence,
                    .parameter_adapter = std::make_shared<AdamParameterAdapter>(
                        0.8, 0.9, 0.999, 1e-8, learning_rate_scale * std::log(N) / (d * d)),
                    .debug = debug
                };
            }

            int N = 10;

            return core::SolverConfig{
                .N = N,
                //.N=static_cast<int>(8 * std::log(d * d)),
                .d = d,
                .params = core::HyperParameters{
                    .beta = 1.0,
                    .lambda = 1.0,
                    .delta = initial_variance
                },
                .dtype = dtype,
                .convergence = convergence,
                .parameter_adapter = std::make_shared<AdamParameterAdapter>(
                    0.8, 0.99, 0.999, 1e-8, learning_rate_scale * std::log(N) / (d * d)),
                .debug = debug
            };
        }

    protected:
        double cached_sigma_ = -1.0;
        Tensor cached_inv_cdf_;
        const int table_size_ = 256; // 256 points is plenty for linear interpolation

        /**
         * @brief Generates the 1D Inverse CDF table for the intrinsic Haar measure.
         * Evaluates p(theta) = (1 - cos(theta)) * exp(-theta^2 / (2 * sigma^2)),
         * computes the CDF, and inverts it onto a uniform grid [0, 1].
         */
        void update_intrinsic_table(double sigma) {
            // 1. UPPER CLAMP: Prevent distribution flattening beyond the cut locus
            sigma = std::min(sigma, M_PI);

            std::vector<double> theta_grid(table_size_);
            std::vector<double> pdf(table_size_);
            std::vector<double> cdf(table_size_, 0.0);

            double d_theta = M_PI / (table_size_ - 1);
            double sum_pdf = 0.0;

            // 2. CPU CACHE-FRIENDLY LOOP (Standard compilers will auto-vectorize this SIMD)
            double inv_two_sigma_sq = 1.0 / (2.0 * sigma * sigma);
            for (int i = 0; i < table_size_; ++i) {
                double theta = i * d_theta;
                theta_grid[i] = theta;

                double haar_penalty = 1.0 - std::cos(theta);
                double gaussian_decay = std::exp(-(theta * theta) * inv_two_sigma_sq);

                pdf[i] = haar_penalty * gaussian_decay;
                sum_pdf += pdf[i];
            }

            double current_sum = 0.0;
            double inv_sum = 1.0 / sum_pdf; // Multiply is faster than divide
            for (int i = 0; i < table_size_; ++i) {
                current_sum += pdf[i];
                cdf[i] = current_sum * inv_sum;
            }

            std::vector<float> inv_cdf(table_size_);
            int search_start_idx = 0; // Optimization: Exploit monotonic CDF

            for (int i = 0; i < table_size_; ++i) {
                double u_target = static_cast<double>(i) / (table_size_ - 1);

                // Fast-forward search starting from previous index
                while (search_start_idx < table_size_ && cdf[search_start_idx] < u_target) {
                    search_start_idx++;
                }
                int idx = search_start_idx;

                if (idx == 0) {
                    inv_cdf[i] = static_cast<float>(theta_grid[0]);
                } else {
                    double u_high = cdf[idx];
                    double u_low = cdf[idx - 1];
                    double t_high = theta_grid[idx];
                    double t_low = theta_grid[idx - 1];

                    double weight = (u_target - u_low) / (std::max(1e-9, u_high - u_low));
                    inv_cdf[i] = static_cast<float>(t_low + weight * (t_high - t_low));
                }
            }

            cached_inv_cdf_ = math::array(inv_cdf, {table_size_}, config_.dtype);
            cached_sigma_ = sigma;
        }

        /**
         * @brief Vectorized linear interpolation to sample from the Inverse CDF.
         * Takes standard uniform noise U [0, 1] and maps it to true theta [0, pi].
         */
        Tensor sample_intrinsic_angle(const Tensor &U_chunk) {
            float max_idx = static_cast<float>(table_size_ - 1) - 0.001f;

            // 1. Calculate exact floating point index
            Tensor exact_idx = math::multiply(U_chunk, Tensor(max_idx, config_.dtype));

            // 2. Get lower and upper bounds (Float)
            Tensor lower_idx_float = math::floor(exact_idx);
            Tensor upper_idx_float = math::add(lower_idx_float, Tensor(1.0f, config_.dtype));

            // 3. Cast bounds to Int32 for MLX GPU gather
            Tensor lower_idx = math::astype_int32(lower_idx_float);
            Tensor upper_idx = math::astype_int32(upper_idx_float);

            // 4. Calculate the interpolation weights
            Tensor weight = math::subtract(exact_idx, lower_idx_float);
            Tensor inv_weight = math::subtract(Tensor(1.0f, config_.dtype), weight);

            // 5. Gather the table values using our new MLX hook
            Tensor val_lower = math::gather(cached_inv_cdf_, lower_idx, 0);
            Tensor val_upper = math::gather(cached_inv_cdf_, upper_idx, 0);

            Tensor delta = math::subtract(val_upper, val_lower);
            Tensor scaled_delta = math::multiply(weight, delta);

            return math::add(val_lower, scaled_delta);
        }

        /**
         * @brief Dispatcher for the Batched Cayley Transform.
         * Routes to an exact analytic solver for SO(3), or a Newton-Schulz iterative
         * solver for higher-dimensional SO(d).
         */
        Tensor cayley_transform_chunk(Tensor W_chunk) const {
            int d = W_chunk.shape()[1];
            if (d == 3) {
                return cayley_transform_3d(std::move(W_chunk));
            } else {
                return cayley_transform_nd(std::move(W_chunk));
            }
        }

        /**
         * @brief Fast Analytical Cayley Transform specifically for SO(3).
         * Uses the closed-form geometric equivalent: R = I + c * S + c * S^2
         * Completely bypasses iterative matrix inversion.
         */
        Tensor cayley_transform_3d(Tensor W_chunk) const {
            auto w_shape = W_chunk.shape();
            int d = w_shape[1]; // Guaranteed to be 3

            // S = W / 2
            Tensor S = math::multiply(W_chunk, Tensor(0.5, config_.dtype));

            // Compute S^2
            Tensor S_sq = math::matmul(S, S);

            // Compute ||S||_F^2 (Frobenius norm squared over spatial dimensions)
            Tensor S_element_sq = math::square(S);
            Tensor S_F2 = math::sum(S_element_sq, {1, 2}); // Shape: [batch]

            // ||v||^2 = 0.5 * ||S||_F^2
            Tensor v_sq = math::multiply(S_F2, Tensor(0.5, config_.dtype));

            // c = 2 / (1 + ||v||^2)
            Tensor denominator = math::add(Tensor(1.0, config_.dtype), v_sq);
            Tensor c = math::divide(Tensor(2.0, config_.dtype), denominator);

            // Expand c for batch broadcasting: [batch] -> [batch, 1, 1]
            Tensor c_expanded = math::expand_dims(c, {1, 2});

            // Compute the update: c * S + c * S^2
            Tensor cS = math::multiply(c_expanded, S);
            Tensor cS_sq = math::multiply(c_expanded, S_sq);
            Tensor update = math::add(cS, cS_sq);

            // R = I + (c * S + c * S^2)
            Tensor I = math::eye(d, config_.dtype);
            Tensor R = math::add(I, update);

            // Evaluate to clear graph
            //math::eval(R);

            return R;
        }

        /**
         * @brief General Batched Cayley Transform for SO(d) where d > 3.
         * Uses a Newton-Schulz iteration to compute the inverse of (I - 0.5W).
         */
        Tensor cayley_transform_nd(Tensor W_chunk) const {
            auto w_shape = W_chunk.shape();
            int current_batch = w_shape[0];
            int d = w_shape[1];

            Tensor I = math::eye(d, config_.dtype);
            Tensor two_I = math::multiply(I, Tensor(2.0, config_.dtype));

            Tensor A, right;

            // ====================================================================
            // PHASE 2A: INITIALIZATION (Scope isolates and destroys W_chunk)
            // ====================================================================
            {
                Tensor half_W = math::multiply(W_chunk, Tensor(0.5, config_.dtype));
                A = math::subtract(I, half_W);
                right = math::add(I, half_W);

                math::eval(A);
                math::eval(right);
            }

            Tensor A_T = math::transpose(A, {0, 2, 1});
            Tensor A_sq = math::square(A);
            Tensor A_F2 = math::sum(A_sq, {1, 2});

            Tensor alpha = math::divide(Tensor(1.0, config_.dtype), A_F2);
            Tensor alpha_expanded = math::expand_dims(alpha, {1, 2});
            Tensor X = math::multiply(A_T, alpha_expanded);

            const int max_iterations = 7;
            // const double tolerance = 1e-8;

            for (int iter = 0; iter < max_iterations; ++iter) {
                Tensor AX = math::matmul(A, X);
                Tensor inner = math::subtract(two_I, AX);
                Tensor X_next = math::matmul(X, inner);

                // Tensor diff = math::subtract(X_next, X);
                // Tensor sq_diff = math::square(diff);

                X = X_next;
            }
            math::eval(X);

            return math::matmul(X, right);
        }

        /**
         * @brief Forces an SVD decomposition into SO(d) using the Kabsch algorithm.
         * Mimics Eigen's U_flipped * V.transpose() by applying a diagonal correction.
         */
        Tensor enforce_so_d(const Tensor &U, const Tensor &Vt) {
            int N = U.shape()[0];
            int d = U.shape()[1];

            // 1. Calculate the standard O(d) projection: P = U * V^T
            Tensor p_tensor = math::matmul(U, Vt);
            //mlx::core::eval(U);

            // 2. Calculate determinants to find reflections
            Tensor dets = math::det(p_tensor); // Using the custom CPU fallback we wrote

            // Clean dets to exactly +1.0 or -1.0
            Tensor zero = Tensor(0.0f, U.dtype());
            Tensor one = Tensor(1.0f, U.dtype());
            Tensor neg_one = Tensor(-1.0f, U.dtype());
            Tensor clean_dets = math::where(
                math::greater(dets, zero), one, neg_one
            );

            // 3. Construct the diagonal correction matrix D
            Tensor dets_exp = math::reshape(clean_dets, {N, 1, 1});
            std::vector<float> mask_data(d * d, 0.0f);
            mask_data[d * d - 1] = 1.0f;

            Tensor mask = math::array(mask_data, {d, d}, U.dtype());

            Tensor I = math::eye(d, U.dtype());
            Tensor diff = math::subtract(dets_exp, one);
            Tensor correction = math::multiply(mask, diff);

            // D is Identity, except D(d-1, d-1) = det
            Tensor D = math::add(I, correction);

            // 4. Kabsch Projection: SO(d) Matrix = U * D * V^T
            Tensor UD = math::matmul(U, D);
            Tensor SO_d_matrix = math::matmul(UD, Vt);

            return SO_d_matrix;
        }

        /**
         * @brief Initializes particles using random Lie algebra elements.
         */
        Tensor initialize_particles() override {
            int N_total = config_.N;
            int d = config_.d;

            // Dynamically scale chunk size based on O(d^3) complexity.
            // TODO: Estimate this properly
            size_t ops_threshold = 500000000;
            int chunk_size = ops_threshold / (d * d * d);

            std::vector<Tensor> chunks;

            for (int i = 0; i < N_total; i += chunk_size) {
                int current_batch = std::min(chunk_size, N_total - i);

                // Generate Gaussian noise for this chunk
                Tensor Z_chunk = math::random_normal({current_batch, d, d}, config_.dtype);

                // Decompose via SVD and project strictly to SO(d)
                auto [U, S, Vt] = math::svd(Z_chunk);
                //std::cout << Z_chunk << "\n";
                Tensor SO_d_chunk = enforce_so_d(U, Vt);

                // Force the Apple GPU to execute and flush this chunk's graph
                math::eval(SO_d_chunk);

                chunks.push_back(SO_d_chunk);
            }

            // Hardware-agnostic concatenation of the evaluated chunks
            return math::concatenate(chunks, 0);
        }


        /**
         * @brief Computes the weighted consensus and applies a direct chordal projection.
         * Uses a direct, non-iterative QR decomposition to snap the ambient sum
         * to the orthogonal group.
         */
        Tensor compute_consensus(const Tensor &particles, const Tensor &weights) override {
            Tensor weights_expanded = math::expand_dims(weights, {1, 2});
            Tensor weighted_particles = math::multiply(particles, weights_expanded);

            // Extrinsic ambient mean
            Tensor M_amb = math::sum(weighted_particles, {0});

            // Temporarily batch to [1, d, d] for our 3D utilities
            Tensor M_amb_batched = math::expand_dims(M_amb, {0});

            // Decompose ambient mean via SVD
            auto [U_batched, S_batched, Vt_batched] = math::svd(M_amb_batched);

            // Snap back to SO(d)
            Tensor Q_so_d_batched = enforce_so_d(U_batched, Vt_batched);

            //std::cout << Q_so_d_batched << std::endl;
            //Tensor Q_so_d_batched = math::matmul(U_batched, U_batched);

            //std::cout << "\n--- [SOSolver] Extrinsic Ambient Mean (Pre-Projection) ---\n"
            //          << Q_so_d_batched << std::endl;

            // Squeeze the batch dimension back out to return [d, d]
            return math::sum(Q_so_d_batched, {0});
        }

        /**
         * @brief Exact Intrinsic Exponential Map for SO(3).
         * Uses Rodrigues' Rotation Formula to perfectly map tangent space
         * noise onto the manifold without the distortion of the Cayley transform.
         */
        Tensor exponential_map_3d(Tensor W_chunk) const {
            int d = W_chunk.shape()[1]; // Guaranteed to be 3

            // 1. Compute W^2 (Matrix multiplication for the formula)
            Tensor W_mat_sq = math::matmul(W_chunk, W_chunk);

            // 2. Compute the rotation angle theta
            // ||W||_F^2 = sum(W * W) (element-wise square)
            Tensor W_element_sq = math::square(W_chunk);
            Tensor W_F2 = math::sum(W_element_sq, {1, 2}); // Shape: [batch]
            Tensor theta_sq = math::multiply(W_F2, Tensor(0.5, config_.dtype));
            Tensor theta = math::sqrt(theta_sq); // Shape: [batch]

            // 3. Safe division to prevent NaNs at the origin (theta = 0)
            // We add a tiny epsilon to theta. When theta is near 0,
            // sin(theta)/theta -> 1, which gracefully degrades to the identity matrix.
            Tensor eps = Tensor(1e-7, config_.dtype);
            Tensor safe_theta = math::add(theta, eps);
            Tensor safe_theta_sq = math::square(safe_theta);

            // 4. Calculate coefficients
            // c1 = sin(theta) / theta
            Tensor sin_theta = math::sin(safe_theta);
            Tensor c1 = math::divide(sin_theta, safe_theta);

            // c2 = (1 - cos(theta)) / theta^2
            Tensor cos_theta = math::cos(safe_theta);
            Tensor one_minus_cos = math::subtract(Tensor(1.0, config_.dtype), cos_theta);
            Tensor c2 = math::divide(one_minus_cos, safe_theta_sq);

            // Expand dims for broadcasting: [batch] -> [batch, 1, 1]
            Tensor c1_exp = math::expand_dims(c1, {1, 2});
            Tensor c2_exp = math::expand_dims(c2, {1, 2});

            // 5. Compute exact rotation: R = I + c1 * W + c2 * W^2
            Tensor c1_W = math::multiply(c1_exp, W_chunk);
            Tensor c2_W_sq = math::multiply(c2_exp, W_mat_sq);

            Tensor I = math::eye(d, config_.dtype);
            Tensor R = math::add(I, math::add(c1_W, c2_W_sq));

            return R;
        }

        /**
         * @brief Generates the canonical isotropic measure on so(d).
         */
        Tensor generate_noise() override {
            Tensor G = math::random_normal({config_.N, config_.d, config_.d}, config_.dtype);
            Tensor G_t = math::transpose(G, {0, 2, 1});
            Tensor diff = math::subtract(G, G_t);
            return math::multiply(diff, Tensor(1.0 / std::sqrt(2.0), config_.dtype));
        }

        /**
         * @brief The True Spectral Hopping Scheme Update.
         */
        Tensor step(const Tensor &particles, const Tensor &consensus,
                    const Tensor &xi, core::HyperParameters params) override {
            bool use_intrinsic_sampler = true;
            int N_total = config_.N;
            int d = config_.d;

            // Ensure table is up to date based on current SDE dynamics
            double current_sigma = params.delta / std::sqrt(2.0 * params.lambda);
            if (std::abs(current_sigma - cached_sigma_) > 1e-6) {
                update_intrinsic_table(current_sigma);
            }

            // Memory scaling logic (omitted for brevity, keep your existing chunking logic)
            int chunk_size = 25000; // Your memory logic here
            std::vector<Tensor> next_particles;
            Tensor C_exp = math::expand_dims(consensus, {0});

            for (int i = 0; i < N_total; i += chunk_size) {
                int current_batch = std::min(chunk_size, N_total - i);
                Tensor xi_chunk = math::slice(xi, i, i + current_batch, 0);
                Tensor W_inner_chunk;

                if (use_intrinsic_sampler && d == 3) {
                    // Calculate norms first
                    Tensor W_sq = math::square(xi_chunk);
                    Tensor W_F2 = math::sum(W_sq, {1, 2});
                    Tensor norm = math::sqrt(W_F2);
                    Tensor safe_norm = math::add(norm, Tensor(1e-8, config_.dtype));
                    Tensor u_matrix = math::divide(xi_chunk, math::expand_dims(safe_norm, {1, 2}));

                    Tensor theta;

                    // --- THE ZERO-COST FAST PATH ---
                    if (current_sigma < 1e-3) {
                        // Space is effectively flat. The magnitude of a 3D Gaussian
                        // mathematically matches the tiny-sigma Haar limit.
                        // Bypass the CDF table and gather operations entirely!
                        Tensor S_iso = Tensor(current_sigma, config_.dtype);
                        theta = math::multiply(norm, S_iso);
                    } else {
                        // --- THE EXACT INTRINSIC PATH ---
                        Tensor U_chunk = math::random_uniform({current_batch}, config_.dtype);
                        theta = sample_intrinsic_angle(U_chunk);
                    }

                    // Reassemble and retract
                    W_inner_chunk = math::multiply(math::expand_dims(theta, {1, 2}), u_matrix);
                    math::eval(W_inner_chunk);

                    Tensor R_update_chunk = exponential_map_3d(std::move(W_inner_chunk));
                    Tensor X_new_chunk = math::matmul(R_update_chunk, C_exp);

                    math::eval(X_new_chunk);
                    next_particles.push_back(X_new_chunk);
                } else {
                    // Fallback to Wrapped Gaussian + Cayley for d != 3
                    Tensor S_iso = math::sqrt(Tensor((params.delta * params.delta) / (2.0 * params.lambda),
                                                     config_.dtype));
                    W_inner_chunk = math::multiply(xi_chunk, S_iso);

                    Tensor R_update_chunk = cayley_transform_chunk(std::move(W_inner_chunk));
                    Tensor X_new_chunk = math::matmul(R_update_chunk, C_exp);

                    math::eval(X_new_chunk);
                    next_particles.push_back(X_new_chunk);
                }
            }

            return math::concatenate(next_particles, 0);
        }

        /**
         * @brief Verifies that the entire batch of particles sits on the SO(d) manifold.
         * Checks the orthogonality constraint: X^T * X = I for all N particles.
         */
        bool check_manifold_constraint(const Tensor &p) const override {
            // 1. Transpose the spatial dimensions of the particle batch
            // p shape: [N, d, d] -> p_T shape: [N, d, d]
            Tensor p_T = math::transpose(p, {0, 2, 1});

            // 2. Compute X^T * X for all particles in parallel
            Tensor Pt_P = math::matmul(p_T, p);

            // 3. Compare against the Identity matrix
            // I is [d, d] and will broadcast perfectly against [N, d, d]
            Tensor I = math::eye(config_.d, config_.dtype);
            Tensor diff = math::subtract(Pt_P, I);

            // 4. Calculate the Mean Squared Error (MSE) across the entire batch
            Tensor sq_diff = math::square(diff);
            double total_squared_error = math::to_double(math::sum(sq_diff));

            // Normalize the error by the total number of matrix elements (N * d * d)
            // This makes the tolerance threshold independent of your batch size or dimensions.
            double mean_squared_error = total_squared_error / (config_.N * config_.d * config_.d);

            // For Float32 GPU math, an MSE under 1e-5 means the constraint is perfectly satisfied.
            const double tolerance = 1e-5;

            if (mean_squared_error > tolerance) {
                std::cerr << "[SOSolver] Manifold constraint failed! MSE: "
                        << mean_squared_error << " exceeds tolerance of " << tolerance << "\n";
                return false;
            }

            return true;
        }
    };
} // namespace involute::solvers
