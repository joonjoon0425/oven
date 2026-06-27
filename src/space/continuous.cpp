#include <c10/util/ArrayRef.h>
#include <oven/space/continuous.hpp>
#include <oven/utils/assert.hpp>

namespace oven::environment {

ContinuousSpace::ContinuousSpace(torch::Tensor lower_bound, torch::Tensor upper_bound)
: lower_bound_(lower_bound),
  upper_bound_(upper_bound) {}

c10::IntArrayRef ContinuousSpace::shape() const {
    return lower_bound_.sizes();
}

torch::Tensor ContinuousSpace::sample() const {

}

bool ContinuousSpace::contains(const torch::Tensor& x) const {
    OVEN_ASSERT(lower_bound_.device() == x.device(), "Must be on the same device.");
    OVEN_ASSERT(lower_bound_.dtype() == x.dtype(), "Must have same tensor type.");

    return (lower_bound_ <= x).all().item<bool>() && (upper_bound_ >= x).all().item<bool>();
}

const torch::Tensor& ContinuousSpace::lower_bound() const {
    return lower_bound_;
}

const torch::Tensor& ContinuousSpace::upper_bound() const {
    return upper_bound_;
}

}