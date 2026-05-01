#pragma once

/**
 * @file so_objectives.hpp
 * @brief Shared objective function constructors for SO(d) test suites.
 *
 * All functions return a FuncObj (or helper struct) centred at the identity
 * matrix I ∈ SO(d), unless a target is supplied explicitly.
 */

#include "involute/core/objective.hpp"
#include "involute/core/math.hpp"
#include <cmath>

namespace involute::test {

using namespace involute::core;

// =============================================================================
// Ackley — multimodal, global min f(I) = 0
// =============================================================================

/**
 * @brief Ackley function on SO(d), centred at the identity matrix.
 *
 * f(X) = -a·exp(-b·√(‖X−I‖²_F / d²))
 *        − exp(mean cos(c·(X−I)))
 *        + a + e
 *
 * Global minimum: f(I) = 0.
 */
inline FuncObj make_ackley(int d, DType dtype) {
    return FuncObj([d, dtype](const Tensor& X) {
        const float a = 20.0f;
        const float b = 0.2f;
        const float c = static_cast<float>(2.0 * 3.14159265358979323846);
        const float n = static_cast<float>(d * d);

        Tensor I    = math::eye(d, dtype);
        Tensor diff = math::subtract(X, I);

        Tensor sum_sq   = math::sum(math::square(diff), {1, 2});
        Tensor mean_sq  = math::divide(sum_sq, Tensor(n, dtype));
        Tensor t1_inner = math::multiply(Tensor(-b, dtype), math::sqrt(mean_sq));
        Tensor term1    = math::multiply(Tensor(-a, dtype), math::exp(t1_inner));

        Tensor cos_diff = math::cos(math::multiply(diff, Tensor(c, dtype)));
        Tensor mean_cos = math::divide(math::sum(cos_diff, {1, 2}), Tensor(n, dtype));
        Tensor term2    = math::multiply(Tensor(-1.0f, dtype), math::exp(mean_cos));

        float constant = a + static_cast<float>(std::exp(1.0));
        return math::add(math::add(term1, term2), Tensor(constant, dtype));
    });
}

// =============================================================================
// Rastrigin — highly multimodal, global min f(I) = 0
// =============================================================================

/**
 * @brief Rastrigin function on SO(d), centred at the identity matrix.
 *
 * f(X) = Σᵢⱼ [ (Xᵢⱼ − Iᵢⱼ)² + A(1 − cos(2π(Xᵢⱼ − Iᵢⱼ))) ]
 *
 * Global minimum: f(I) = 0.
 */
inline FuncObj make_rastrigin(int d, DType dtype) {
    Tensor X_target   = math::expand_dims(math::eye(d, dtype), {0});
    Tensor tensor_two_pi = Tensor(static_cast<float>(2.0 * 3.14159265358979323846), dtype);
    Tensor tensor_A   = Tensor(10.0f, dtype);

    return FuncObj([X_target, tensor_two_pi, tensor_A](const Tensor& X) {
        Tensor diff          = math::subtract(X, X_target);
        Tensor sq_diff       = math::square(diff);
        Tensor two_pi_diff   = math::multiply(diff, tensor_two_pi);
        Tensor cos_term      = math::cos(two_pi_diff);
        Tensor A_cos         = math::multiply(cos_term, tensor_A);
        Tensor A_minus_A_cos = math::subtract(tensor_A, A_cos);
        Tensor elem          = math::add(sq_diff, A_minus_A_cos);
        return math::sum(elem, {1, 2});
    });
}

// =============================================================================
// Schwefel — single-basin in the rotated frame, global min f(I) = 0
// =============================================================================

/**
 * @brief Schwefel function on SO(d), centred at the identity matrix.
 *
 * The standard Euclidean Schwefel is shifted/scaled so that the optimum
 * corresponds to the identity matrix: diff = X − I is mapped to
 * z = 250·diff + 420.968746, then f = A·d² − Σ z·sin(√|z|).
 *
 * Global minimum: f(I) ≈ 0.
 */
inline FuncObj make_schwefel(int d, DType dtype) {
    return FuncObj([d, dtype](const Tensor& X) {
        const double optimal_val = 420.968746;
        const double A = 418.9829;
        const double n = static_cast<double>(d * d);

        Tensor I        = math::eye(d, dtype);
        Tensor diff     = math::subtract(X, I);
        Tensor D_scaled = math::multiply(diff, Tensor(250.0, dtype));
        Tensor Z        = math::add(D_scaled, Tensor(optimal_val, dtype));

        Tensor abs_Z      = math::abs(Z);
        Tensor sqrt_abs_Z = math::sqrt(abs_Z);
        Tensor sin_term   = math::sin(sqrt_abs_Z);
        Tensor Z_sin      = math::multiply(Z, sin_term);
        Tensor sum_Z_sin  = math::sum(Z_sin, {1, 2});

        return math::subtract(Tensor(A * n, dtype), sum_Z_sin);
    });
}

// =============================================================================
// Brockett — smooth trace cost, known theoretical minimum
// =============================================================================

/**
 * @brief Return type for make_brockett, bundling the FuncObj and theoretical minimum.
 */
struct BrockettObjective {
    FuncObj func;
    double  theoretical_min;
};

/**
 * @brief Brockett cost function f(X) = tr(X^T A X S) on SO(d).
 *
 * A = diag(d, d−1, …, 1), S = diag(1, 2, …, d).
 * Theoretical minimum = Σᵢ aᵢ·sᵢ (achieved at X = I for the paired ordering).
 */
inline BrockettObjective make_brockett(int d, DType dtype) {
    std::vector<float> A_data(d * d, 0.0f), S_data(d * d, 0.0f);
    double tmin = 0.0;
    for (int i = 0; i < d; i++) {
        float av = static_cast<float>(d - i);
        float sv = static_cast<float>(i + 1);
        A_data[i * d + i] = av;
        S_data[i * d + i] = sv;
        tmin += av * sv;
    }
    Tensor A = math::array(A_data, {1, d, d}, dtype);
    Tensor S = math::array(S_data, {1, d, d}, dtype);

    FuncObj f([d, dtype, A, S](const Tensor& X) {
        Tensor XT    = math::transpose(X, {0, 2, 1});
        Tensor XTAXS = math::matmul(math::matmul(XT, math::matmul(A, X)), S);
        Tensor I1    = math::reshape(math::eye(d, dtype), {1, d, d});
        return math::sum(math::multiply(XTAXS, I1), {1, 2});
    });

    return {std::move(f), tmin};
}

// =============================================================================
// Geodesic — distance squared to a fixed target R* ∈ SO(d)
// =============================================================================

/**
 * @brief Geodesic distance squared to a fixed target R_star ∈ SO(d).
 *
 * f(X) = ½ ‖Log_{R*}(X)‖²_F  where  Log_{R*}(X) = skew(log(R*ᵀ X)).
 *
 * Global minimum: f(R*) = 0.
 */
inline FuncObj make_geodesic(int /*d*/, DType dtype, const Tensor& R_star) {
    Tensor Rst_T = math::expand_dims(math::transpose(R_star, {1, 0}), {0});
    return FuncObj([dtype, Rst_T](const Tensor& X) {
        Tensor R     = math::matmul(Rst_T, X);
        Tensor logR  = math::matrix_log(R);
        Tensor logRT = math::transpose(logR, {0, 2, 1});
        Tensor V     = math::multiply(math::subtract(logR, logRT), Tensor(0.5f, dtype));
        return math::multiply(math::sum(math::square(V), {1, 2}), Tensor(0.5f, dtype));
    });
}

// =============================================================================
// Utility — Haar-uniform random rotation from SO(d)
// =============================================================================

/**
 * @brief Draw a single Haar-uniform random rotation matrix from SO(d).
 */
inline Tensor haar_rotation(int d, DType dtype) {
    Tensor Z  = math::random_normal({d, d}, dtype);
    Tensor Zb = math::expand_dims(Z, {0});
    auto [U, S, Vt] = math::svd(Zb);
    Tensor UV   = math::matmul(U, Vt);
    Tensor dets = math::det(UV);
    math::eval(dets);
    float dv = math::to_float_vector(dets)[0];
    Tensor m = math::squeeze(UV, {0});
    if (dv < 0.0f) {
        std::vector<float> buf = math::to_float_vector(m);
        for (int r = 0; r < d; r++) buf[r * d + (d - 1)] *= -1.0f;
        return math::array(buf, {d, d}, dtype);
    }
    return m;
}

} // namespace involute::test
