/**
 * @file test_math_basic.cpp
 * @brief Pure C++ execution entry point for testing basic MLX math operations.
 * Zero external testing frameworks used.
 */

#include "involute/core/math.hpp"
#include <iostream>
#include <cmath>

using namespace involute;

// Helper macros for raw C++ assertions
#define CHECK_NOT_NAN(val, name) \
    do { \
        if (std::isnan(val)) { \
            std::cerr << "[FAIL] " << name << " produced NaN!\n"; \
            return 1; \
        } \
    } while(0)

#define CHECK_APPROX(actual, expected, tol, name) \
    do { \
        CHECK_NOT_NAN(actual, name); \
        if (std::abs((actual) - (expected)) > (tol)) { \
            std::cerr << "[FAIL] " << name << " | Expected: " << (expected) \
                      << ", Got: " << (actual) << "\n"; \
            return 1; \
        } else { \
            std::cout << "[PASS] " << name << "\n"; \
        } \
    } while(0)

int main(int argc, char* argv[]) {
    std::cout << "--- Initializing Raw MLX Math Tests ---\n";

        // ------------------------------------------------------------------
        // 1. Scalar Arithmetic
        // ------------------------------------------------------------------
        Tensor a(2.0);
        Tensor b(3.0);

        Tensor add_res = math::add(a, b);
        Tensor sub_res = math::subtract(a, b);
        Tensor mul_res = math::multiply(a, b);
        Tensor div_res = math::divide(a, b);

        CHECK_APPROX(math::to_double(add_res), 5.0, 1e-5, "Scalar Addition");
        CHECK_APPROX(math::to_double(sub_res), -1.0, 1e-5, "Scalar Subtraction");
        CHECK_APPROX(math::to_double(mul_res), 6.0, 1e-5, "Scalar Multiplication");
        CHECK_APPROX(math::to_double(div_res), 2.0 / 3.0, 1e-5, "Scalar Division");

        // ------------------------------------------------------------------
        // 2. Matrix Generation & Reductions
        // ------------------------------------------------------------------
        int d = 4;
        Tensor I = math::eye(d);

        // Summing an Identity matrix should exactly equal its dimension 'd'
        Tensor sum_I = math::sum(I);
        CHECK_APPROX(math::to_double(sum_I), static_cast<double>(d), 1e-5, "math::eye and math::sum");

        // ------------------------------------------------------------------
        // 3. Element-wise Matrix Arithmetic & Broadcasting
        // ------------------------------------------------------------------
        Tensor scaled_I = math::multiply(I, Tensor(5.0));
        Tensor sum_scaled = math::sum(scaled_I);

        // Sum of a 4x4 identity scaled by 5 should be 20
        CHECK_APPROX(math::to_double(sum_scaled), 20.0, 1e-5, "Matrix Scalar Broadcasting");

        // ------------------------------------------------------------------
        // 4. Non-Linearities & NaN Stress Testing
        // ------------------------------------------------------------------
        Tensor neg_tensor(-4.0);
        Tensor abs_tensor = math::abs(neg_tensor);
        Tensor sqrt_tensor = math::sqrt(abs_tensor);
        CHECK_APPROX(math::to_double(sqrt_tensor), 2.0, 1e-5, "Absolute Value and Square Root");

        // Stress Test: Ensure division by a very small number doesn't NaN out
        Tensor tiny_val(1e-12);
        Tensor safe_div = math::divide(Tensor(1.0), tiny_val);
        double safe_div_val = math::to_double(safe_div);
        CHECK_NOT_NAN(safe_div_val, "Division by epsilon");

        // Stress Test: Ensure sqrt of 0.0 evaluates to 0.0 without generating NaN
        Tensor zero_tensor(0.0);
        Tensor sqrt_zero = math::sqrt(zero_tensor);
        CHECK_APPROX(math::to_double(sqrt_zero), 0.0, 1e-7, "Square Root of Zero");


    std::cout << "--- All MLX Math Tests Completed Successfully ---\n";
    return 0;
}