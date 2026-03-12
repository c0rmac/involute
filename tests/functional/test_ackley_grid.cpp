/**
 * @file grid_search_so_d.cpp
 * @brief Standalone Grid Search for the Ackley function on SO(d).
 */

#include <iostream>
#include <vector>
#include <cmath>
#include <iomanip>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif
#ifndef M_E
#define M_E 2.71828182845904523536
#endif

using Matrix = std::vector<std::vector<double>>;

// ==============================================================================
// 1. BASIC MATRIX MATH (Adapted for pure C++)
// ==============================================================================
namespace Math {
    Matrix zeros(int rows, int cols) {
        return Matrix(rows, std::vector<double>(cols, 0.0));
    }

    Matrix eye(int d) {
        Matrix I = zeros(d, d);
        for (int i = 0; i < d; ++i) I[i][i] = 1.0;
        return I;
    }

    Matrix add(const Matrix& A, const Matrix& B) {
        int d = A.size();
        Matrix C = zeros(d, d);
        for (int i = 0; i < d; ++i)
            for (int j = 0; j < d; ++j)
                C[i][j] = A[i][j] + B[i][j];
        return C;
    }

    Matrix subtract(const Matrix& A, const Matrix& B) {
        int d = A.size();
        Matrix C = zeros(d, d);
        for (int i = 0; i < d; ++i)
            for (int j = 0; j < d; ++j)
                C[i][j] = A[i][j] - B[i][j];
        return C;
    }

    Matrix multiply(const Matrix& A, double scalar) {
        int d = A.size();
        Matrix C = zeros(d, d);
        for (int i = 0; i < d; ++i)
            for (int j = 0; j < d; ++j)
                C[i][j] = A[i][j] * scalar;
        return C;
    }

    Matrix matmul(const Matrix& A, const Matrix& B) {
        int d = A.size();
        Matrix C = zeros(d, d);
        for (int i = 0; i < d; ++i)
            for (int k = 0; k < d; ++k)
                for (int j = 0; j < d; ++j)
                    C[i][j] += A[i][k] * B[k][j];
        return C;
    }

    Matrix transpose(const Matrix& A) {
        int d = A.size();
        Matrix C = zeros(d, d);
        for (int i = 0; i < d; ++i)
            for (int j = 0; j < d; ++j)
                C[i][j] = A[j][i];
        return C;
    }

    void print(const Matrix& A) {
        int d = A.size();
        for (int i = 0; i < d; ++i) {
            for (int j = 0; j < d; ++j) {
                std::cout << std::setw(10) << std::setprecision(5) << A[i][j] << " ";
            }
            std::cout << "\n";
        }
    }
}

// ==============================================================================
// 2. CAYLEY TRANSFORM (Iterative Newton-Schulz approach)
// ==============================================================================
Matrix cayley_transform(const Matrix& W) {
    int d = W.size();
    Matrix I = Math::eye(d);
    Matrix two_I = Math::multiply(I, 2.0);

    // Initialization phase: A = I - 0.5W, right = I + 0.5W
    Matrix half_W = Math::multiply(W, 0.5);
    Matrix A = Math::subtract(I, half_W);
    Matrix right = Math::add(I, half_W);

    Matrix A_T = Math::transpose(A);

    // Frobenius norm squared calculation (sum of squares)
    double A_F2 = 0.0;
    for (int i = 0; i < d; ++i) {
        for (int j = 0; j < d; ++j) {
            A_F2 += A[i][j] * A[i][j];
        }
    }

    double alpha = 1.0 / A_F2;
    Matrix X = Math::multiply(A_T, alpha);

    const int max_iterations = 7;

    // Newton-Schulz Loop
    for (int iter = 0; iter < max_iterations; ++iter) {
        Matrix AX = Math::matmul(A, X);
        Matrix inner = Math::subtract(two_I, AX);
        X = Math::matmul(X, inner);
    }

    // Return mapped orthogonal matrix: X * right
    return Math::matmul(X, right);
}

// ==============================================================================
// 3. ACKLEY FUNCTION
// ==============================================================================
double ackley_function(const Matrix& R) {
    int d = R.size();
    int n = d * d;

    double sum_sq = 0.0;
    double sum_cos = 0.0;

    for (int i = 0; i < d; ++i) {
        for (int j = 0; j < d; ++j) {
            double val = R[i][j];
            sum_sq += val * val;
            sum_cos += std::cos(2.0 * M_PI * val);
        }
    }

    double term1 = -20.0 * std::exp(-0.2 * std::sqrt(sum_sq / n));
    double term2 = -std::exp(sum_cos / n);

    return term1 + term2 + 20.0 + M_E;
}

// ==============================================================================
// 4. RECURSIVE GRID SEARCH ENGINE
// ==============================================================================
void search_grid_recursive(int d, const std::vector<double>& grid_1d,
                           std::vector<double>& current_combination,
                           int depth, int max_depth,
                           Matrix& best_R, double& best_score, int& eval_count) {
    if (depth == max_depth) {
        // 1. Construct the skew-symmetric matrix W
        Matrix W = Math::zeros(d, d);
        int idx = 0;
        for (int i = 0; i < d; ++i) {
            for (int j = i + 1; j < d; ++j) {
                W[i][j] = current_combination[idx];
                W[j][i] = -current_combination[idx];
                idx++;
            }
        }

        // 2. Map W to the SO(d) manifold
        Matrix R = cayley_transform(W);

        // 3. Evaluate the Ackley objective
        double score = ackley_function(R);
        eval_count++;

        if (score < best_score) {
            best_score = score;
            best_R = R;
        }
        return;
    }

    // Branch to test every point in the grid for the current dimension
    for (double val : grid_1d) {
        current_combination[depth] = val;
        search_grid_recursive(d, grid_1d, current_combination, depth + 1, max_depth, best_R, best_score, eval_count);
    }
}

// ==============================================================================
// MAIN RUNNER
// ==============================================================================
int main() {
    int d = 8; // Dimension of the SO(d) group.

    // Degrees of freedom for skew-symmetric matrices (e.g., d=3 has 3 parameters)
    int k = (d * (d - 1)) / 2;

    // Define a basic parameter grid around 0 in the Lie algebra.
    std::vector<double> grid_1d = {-1.5, -0.5, 0.0, 0.5, 1.5};

    std::cout << "--- Starting Grid Search on SO(" << d << ") ---\n";
    std::cout << "Degrees of freedom (Lie Algebra parameters): " << k << "\n";
    std::cout << "Grid points per parameter: " << grid_1d.size() << "\n";
    std::cout << "Total evaluations planned: " << std::pow(grid_1d.size(), k) << "\n\n";

    std::vector<double> current_combination(k, 0.0);
    Matrix best_R = Math::eye(d);
    double best_score = std::numeric_limits<double>::infinity();
    int eval_count = 0;

    // Trigger the search
    search_grid_recursive(d, grid_1d, current_combination, 0, k, best_R, best_score, eval_count);

    std::cout << "--- Search Complete ---\n";
    std::cout << "Evaluations Performed: " << eval_count << "\n";
    std::cout << "Best Ackley Score: " << best_score << "\n";
    std::cout << "Optimal Matrix configuration in SO(" << d << "):\n";
    Math::print(best_R);

    return 0;
}