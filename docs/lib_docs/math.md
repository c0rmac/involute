# 🧮 `involute::core::math`

This namespace provides a Domain Specific Language (DSL) for batched tensor operations. These functions act as a routing layer: they take opaque `involute::Tensor` objects and translate the operation into highly optimized native calls for the active backend (MLX on Apple, oneMKL/SYCL on PC).

### Usage Example

To initialize tensors and perform basic arithmetic using the Involute Math API, you can utilize the `math::array` and `math::full` functions for vector and scalar-based initialization. The following example demonstrates creating a 3-element vector tensor, a scalar tensor, and performing a fused operation with broadcasting:

```cpp
#include <involute/core/math.hpp>
#include <vector>

using namespace involute::core;

int main() {
    // 1. Initialize a tensor from a C++ vector (shape [3])
    std::vector<float> vec_data = {1.0f, 2.0f, 3.0f};
    Tensor vec_tensor = math::array(vec_data, {3}, DType::Float32);

    // 2. Initialize a tensor from a scalar value (shape [1])
    Tensor scalar_tensor = math::full({1}, 5.0f, DType::Float32);

    // 3. Perform basic operations: (vec_tensor * scalar_tensor) + vec_tensor
    // Note: math::multiply and math::add support broadcasting
    Tensor product = math::multiply(vec_tensor, scalar_tensor);
    Tensor result = math::add(product, vec_tensor);

    // Execute the computation and pull a value back to CPU
    math::eval(result);
    double first_val = math::to_double(math::slice(result, 0, 1)); 
}
```

---

## 1. Element-Wise Arithmetic

| Function Signature | Description |
| :--- | :--- |
| `Tensor add(const Tensor &a, const Tensor &b)` | Element-wise addition of two tensors; supports broadcasting. |
| `Tensor subtract(const Tensor &a, const Tensor &b)` | Element-wise subtraction ($a - b$); supports broadcasting. |
| `Tensor multiply(const Tensor &a, const Tensor &b)` | Element-wise multiplication; used for scaling by beta, lambda, etc. |
| `Tensor divide(const Tensor &a, const Tensor &b)` | Element-wise division ($a / b$); supports broadcasting. |
| `Tensor floor(const Tensor &a)` | Element-wise floor function. |
| `Tensor mean(const Tensor &a)` | Computes the mean of the tensor elements. |
| `Tensor minimum(const Tensor &a, const Tensor &b)` | Element-wise minimum of two tensors; supports broadcasting. |
| `Tensor ceil(const Tensor &a)` | Element-wise ceiling: Returns the smallest integer greater than or equal to each element. |
| `Tensor round(const Tensor &a)` | Element-wise rounding: Rounds elements to the nearest integer. |
| `Tensor clamp(const Tensor &a, float min, float max)` | Constraints tensor values to the range [min, max]; essential for numerical stability and preventing NaN/Inf propagation. |
| `Tensor abs(const Tensor &a)` | Element-wise absolute value: $|a|$; used as a numerical safety net before `sqrt`. |

---

## 2. Core Matrix Operations

| Function Signature | Description |
| :--- | :--- |
| `Tensor matmul(const Tensor &a, const Tensor &b)` | Batched matrix multiplication. Evaluates $A \times B$. If `a` is `[N, d, d]` and `b` is `[N, d, d]`, performs $N$ matmuls. |
| `Tensor transpose(const Tensor &a, const std::vector<int> &axes)` | Swaps the last two dimensions of a tensor. Essential for creating skew-symmetric Lie algebra matrices: $0.5 \cdot (W - W^T)$. |
| `Tensor eye(int d, DType dtype)` | Generates a 2D Identity matrix of size `[d, d]`. Can be broadcasted against batched tensors of shape `[N, d, d]`. |
| `Tensor expand_dims(const Tensor &a, const std::vector<int> &axes)` | Expands the dimensions of a tensor. Essential for broadcasting (e.g., expanding a `[N]` tensor of norms to `[N, 1, 1]` to scale a batch of `[N, d, d]` matrices). |
| `Tensor reshape(const Tensor &a, const std::vector<int> &shape)` | Reshapes a tensor to a new shape. |
| `Tensor broadcast_to(const Tensor &a, const std::vector<int> &shape)` | Broadcasts a tensor to a new target shape. |
| `Tensor array(const std::vector<float> &data, const std::vector<int> &shape, DType dtype)` | Creates a tensor from a raw C++ vector of floats. |
| `Tensor astype_int32(const Tensor &a)` | Casts a tensor to Int32 for index lookups on the GPU. |
| `Tensor gather(const Tensor &a, const Tensor &indices, int axis = 0)` | Gathers values along an axis based on an array of indices. |
| `Tensor stack(const std::vector<Tensor> &tensors, int axis = 0)` | Joins a sequence of tensors along a NEW axis. Contrast with `concatenate`, which joins along an existing axis. |
| `Tensor squeeze(const Tensor &a, const std::vector<int> &axes = {})` | Removes dimensions of size 1 from the tensor shape. |
| `Tensor full(const std::vector<int> &shape, float value, DType dtype)` | Creates a tensor of given shape filled with a constant value. |

---

## 3. Logical Operations

| Function Signature | Description |
| :--- | :--- |
| `Tensor where(const Tensor &condition, const Tensor &x, const Tensor &y)` | Element-wise conditional selection: `(condition) ? x : y`. Vital for keeping branching logic on the GPU without a CPU synchronization event. |
| `Tensor equal(const Tensor &a, const Tensor &b)` | Element-wise equality comparison. |
| `Tensor not_equal(const Tensor &a, const Tensor &b)` | Element-wise inequality comparison. |
| `Tensor greater(const Tensor &a, const Tensor &b)` | Element-wise 'greater than' comparison. |
| `Tensor less(const Tensor &a, const Tensor &b)` | Element-wise 'less than' comparison. |
| `Tensor logical_and(const Tensor &a, const Tensor &b)` | Element-wise logical AND. |
| `Tensor logical_or(const Tensor &a, const Tensor &b)` | Element-wise logical OR. |

---

## 4. Reductions & Non-Linearities

| Function Signature | Description |
| :--- | :--- |
| `Tensor sum(const Tensor &a, const std::vector<int> &axes = {})` | Sums elements of a tensor along specified axes. If axes is empty, sums all elements into a scalar tensor. |
| `Tensor min(const Tensor &a)` | Returns the minimum element of the entire tensor as a scalar tensor. |
| `Tensor exp(const Tensor &a)` | Element-wise exponential: $\exp(a)$. |
| `Tensor log(const Tensor &a)` | Element-wise natural logarithm: $\log(a)$. |
| `Tensor square(const Tensor &a)` | Element-wise square: $a^2$. Used for calculating Frobenius norms. |
| `Tensor sqrt(const Tensor &a)` | Element-wise square root: $\sqrt{a}$. |
| `Tensor sin(const Tensor &a)` | Element-wise sine: $\sin(a)$. Required for the geometric generator. |
| `Tensor cos(const Tensor &a)` | Element-wise cosine: $\cos(a)$. Required for the geometric generator and Ackley. |
| `Tensor asin(const Tensor &a)` | Element-wise inverse sine: $\arcsin(a)$. |
| `Tensor acos(const Tensor &a)` | Element-wise inverse cosine: $\arccos(a)$. |
| `Tensor atan(const Tensor &a)` | Element-wise inverse tangent: $\arctan(a)$. |
| `Tensor argmax(const Tensor &a, int axis = 0)` | Returns the index of the maximum element along an axis. |
| `Tensor max(const Tensor &a)` | Returns the maximum element of the entire tensor as a scalar tensor. |
| `Tensor prod(const Tensor &a, const std::vector<int> &axes = {})` | Computes the product of elements along specified axes. |
| `Tensor all(const Tensor &a, const std::vector<int> &axes = {})` | Returns true if all elements along specified axes are non-zero. |
| `Tensor any(const Tensor &a, const std::vector<int> &axes = {})` | Returns true if any element along specified axes is non-zero. |
| `Tensor pow(const Tensor &a, float exponent)` | Element-wise power function: $a^{\text{exponent}}$. |
| `Tensor tan(const Tensor &a)` | Element-wise tangent function. |
| `Tensor atan2(const Tensor &y, const Tensor &x)` | Element-wise four-quadrant inverse tangent. Essential for robustly recovering angles in geometric and manifold generators. |

---

## 5. Heavy Linear Algebra

| Function Signature | Description |
| :--- | :--- |
| `Tensor solve(const Tensor &a, const Tensor &b)` | Solves the linear system $AX = B$ for $X$. Heavily utilized in the Cayley Transform to avoid matrix inversion. |
| `std::tuple<Tensor, Tensor, Tensor> svd(const Tensor &a)` | Batched Singular Value Decomposition (SVD). Returns a tuple of 3 tensors: `{U, S, V_transpose}`. Required for projecting the ambient Fréchet mean back onto the $SO(d)$ manifold. |
| `std::tuple<Tensor, Tensor> qr(const Tensor &a)` | Batched QR Decomposition. Returns a tuple of 2 tensors: `{Q, R}`. The Q matrix represents the direct chordal projection of an ambient matrix onto the orthogonal group in a single, non-iterative pass. |
| `Tensor det(const Tensor &a)` | Computes the determinant of a square matrix or a batch of square matrices. Returns a tensor containing the determinant(s). |
| `Tensor inv(const Tensor &a)` | Computes the explicit matrix inverse. Prefer `solve` for systems of equations. |
| `Tensor trace(const Tensor &a)` | Computes the sum of diagonal elements (the trace) of a matrix. |

---

## 6. Stochastic Generation

| Function Signature | Description |
| :--- | :--- |
| `Tensor random_normal(const std::vector<int> &shape, DType dtype)` | Generates a tensor filled with random normal (Gaussian) values. Used for initializing particles and synthesizing anisotropic noise. |
| `Tensor random_uniform(const std::vector<int> &shape, DType dtype)` | Generates a tensor filled with uniform random values in $[0, 1)$. |

---

## 7. CPU-GPU Bridge

> **Warning:** Operations that move data to the CPU inherently block the execution pipeline and trigger synchronization events.

| Function Signature | Description |
| :--- | :--- |
| `double to_double(const Tensor &a)` | Pulls a scalar 0D or 1D tensor back to the CPU as a standard C++ double. This triggers a synchronization event. Use sparingly (e.g., once per step for checking convergence). |
| `std::vector<float> to_float_vector(const Tensor &a)` | Converts a tensor to a raw C++ `std::vector<float>`. |
| `void eval(const Tensor &a)` | Explicitly executes the pending computation graph for this tensor. |
| `Tensor concatenate(const std::vector<Tensor> &tensors, int axis = 0)` | Concatenates a vector of tensors along a specified axis. |
| `Tensor slice(const Tensor &a, int start, int end, int axis = 0)` | Slices a tensor along a specified axis from start to end indices. |
| `int to_int(const Tensor &a)` | Pulls a scalar integer tensor back to the CPU as a standard C++ int. Essential for index-based operations like selecting the global best particle. |