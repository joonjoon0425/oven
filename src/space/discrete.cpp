#include <c10/util/ArrayRef.h>
#include <oven/space/discrete.hpp>
#include <oven/utils/assert.hpp>

namespace oven::environment {
DiscreteSpace::DiscreteSpace(int64_t n)
: n_(n) {}

c10::IntArrayRef DiscreteSpace::shape() const {
    return {};
}

torch::Tensor DiscreteSpace::sample() const {

}

bool DiscreteSpace::contains(const torch::Tensor& x) const {
    OVEN_ASSERT(x.dim() == 0, "Must be a scalar.");
    OVEN_ASSERT(x.scalar_type() == torch::kInt64, "Must be torch::kInt64 type.");

    int64_t item = x.item<int64_t>();
    return (item >= 0 && item < n_);
}

int64_t DiscreteSpace::n() const {
    return n_;
}

}