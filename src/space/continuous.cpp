#include <ATen/Functions.h>
#include <c10/core/TensorOptions.h>
#include <c10/util/ArrayRef.h>
#include <oven/space/continuous.hpp>
#include <oven/utils/assert.hpp>

namespace oven::environment {

ContinuousSpace::ContinuousSpace(torch::Tensor lower_bound, torch::Tensor upper_bound)
: lower_bound_(lower_bound),
  upper_bound_(upper_bound)
{
    OVEN_ASSERT(lower_bound.sizes() == upper_bound.sizes(), "Upper bound and lower bound have different sizes");
    OVEN_ASSERT(lower_bound.device() == upper_bound.device(), "Upper bound and lower bound located in different devices.");
    OVEN_ASSERT(torch::is_floating_point(lower_bound), "Lower bound is not floating type.");
    OVEN_ASSERT(torch::is_floating_point(upper_bound), "Upper bound is not floating type.");
    OVEN_ASSERT((lower_bound < upper_bound).all().item<bool>(), "Upper bound is lower than lower bound. What the...");
}

c10::IntArrayRef ContinuousSpace::shape() const {
    return lower_bound_.sizes();
}

torch::Tensor ContinuousSpace::sample() const {
    auto r = torch::rand(lower_bound_.sizes(), torch::TensorOptions().dtype(lower_bound_.dtype()).device(lower_bound_.device()));
    return (upper_bound_ - lower_bound_) * r + lower_bound_;
}

bool ContinuousSpace::contains(const torch::Tensor& x) const {
    OVEN_TENSOR_BINOP_CHECKLIST_STRICT(lower_bound_, x);

    return (lower_bound_ <= x).all().item<bool>() && (upper_bound_ >= x).all().item<bool>();
}

const torch::Tensor& ContinuousSpace::lower_bound() const {
    return lower_bound_;
}

const torch::Tensor& ContinuousSpace::upper_bound() const {
    return upper_bound_;
}

}