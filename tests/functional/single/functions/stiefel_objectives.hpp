#pragma once

/**
 * @file stiefel_objectives.hpp
 * @brief Shared objective function constructors for Stiefel V(n,k) test suites.
 *
 * All functions return a FuncObj centred at the leading-k-columns identity
 * slice I_{n,k} ∈ V(n,k), unless a target is supplied explicitly.
 */

#include "involute/core/objective.hpp"
#include "involute/core/math.hpp"
#include <cmath>

namespace involute::test {

using namespace involute::core;

// =============================================================================
// Ackley — multimodal, global min f(I_{n,k}) = 0
// =============================================================================

/**
 * @brief Ackley function on V(n,k), centred at I_{n,k} = eye(n)[:, :k].
 *
 * f(X) = -a·exp(-b·√(‖X−I_{n,k}‖²_F / (n·k)))
 *        − exp(mean cos(c·(X−I_{n,k})))
 *        + a + e
 *
 * Global minimum: f(I_{n,k}) = 0.
 *
 * @param n  Ambient dimension (rows of X)
 * @param k  Frame size (columns of X)
 * @param dtype  Floating-point type
 */
inline FuncObj make_ackley(int n, int k, DType dtype) {
    Tensor I_n  = math::eye(n, dtype);
    Tensor I_nk = math::slice(I_n, 0, k, 1);  // [n, k]

    return FuncObj([n, k, dtype, I_nk](const Tensor& X) {
        const float a   = 20.0f;
        const float b   = 0.2f;
        const float c   = static_cast<float>(2.0 * 3.14159265358979323846);
        const float num = static_cast<float>(n * k);

        Tensor diff    = math::subtract(X, I_nk);
        Tensor sq_diff = math::square(diff);
        Tensor sum_sq  = math::sum(sq_diff, {1, 2});
        Tensor mean_sq = math::divide(sum_sq, Tensor(num, dtype));
        Tensor t1_inner = math::multiply(Tensor(-b, dtype), math::sqrt(mean_sq));
        Tensor term1    = math::multiply(Tensor(-a, dtype), math::exp(t1_inner));

        Tensor c_diff   = math::multiply(diff, Tensor(c, dtype));
        Tensor cos_diff = math::cos(c_diff);
        Tensor sum_cos  = math::sum(cos_diff, {1, 2});
        Tensor mean_cos = math::divide(sum_cos, Tensor(num, dtype));
        Tensor term2    = math::multiply(Tensor(-1.0f, dtype), math::exp(mean_cos));

        return math::add(math::add(term1, term2),
                         Tensor(a + static_cast<float>(std::exp(1.0)), dtype));
    });
}

} // namespace involute::test
