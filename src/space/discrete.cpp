#include <c10/core/TensorOptions.h>
#include <c10/util/ArrayRef.h>
#include <oven/space/discrete.hpp>
#include <oven/utils/assert.hpp>
#include <random>

namespace oven::environment {
DiscreteSpace::DiscreteSpace(int64_t n, int64_t offset)
: n_(n), offset_(offset) {}

c10::IntArrayRef DiscreteSpace::shape() const {
    return {};
}

torch::Tensor DiscreteSpace::sample() const {
    // TODO add device
    return torch::randint(offset_, offset_ + n_, {}, torch::dtype(torch::kInt64));
}

bool DiscreteSpace::contains(const torch::Tensor& x) const {
    OVEN_ASSERT(x.dim() == 0, "Must be a scalar.");
    OVEN_ASSERT(x.scalar_type() == torch::kInt64, "Must be torch::kInt64 type.");

    int64_t item = x.item<int64_t>();
    return (item >= offset_ && item < offset_ + n_);
}

int64_t DiscreteSpace::n() const {
    return n_;
}

int64_t DiscreteSpace::offset() const {
    return offset_;
}

}