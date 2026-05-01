#pragma once

#include <isomorphism/tensor.hpp>

// Forward all tensor types from isomorphism into the involute namespace.
namespace involute {
    using Tensor    = isomorphism::Tensor;
    using DType     = isomorphism::DType;
    using TensorImpl = isomorphism::TensorImpl;
} // namespace involute
