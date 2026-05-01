#pragma once

#include <isomorphism/math.hpp>
#include "involute/core/tensor.hpp"

// Expose isomorphism::math as both involute::math and involute::core::math
// so that all existing call sites continue to compile unchanged regardless
// of which using-directive they rely on.
namespace involute {
    namespace math = isomorphism::math;
} // namespace involute

namespace involute::core {
    namespace math = isomorphism::math;
} // namespace involute::core
