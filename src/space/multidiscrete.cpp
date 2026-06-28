#include <ATen/core/TensorBody.h>
#include <c10/core/TensorOptions.h>
#include <limits>
#include <oven/utils/assert.hpp>
#include <oven/space/multidiscrete.hpp>

namespace oven::environment {

MultiDiscreteSpace::MultiDiscreteSpace(torch::Tensor lower_bound, torch::Tensor upper_bound)
: lower_bound_(lower_bound),
  upper_bound_(upper_bound)
{
    OVEN_ASSERT(lower_bound.sizes() == upper_bound.sizes(), "Upper bound and lower bound have different sizes");
    OVEN_ASSERT(lower_bound.device() == upper_bound.device(), "Upper bound and lower bound located in different devices.");
    OVEN_ASSERT(lower_bound.dtype() == torch::kInt64, "Lower bound type must be torch::kInt64");
    OVEN_ASSERT(upper_bound.dtype() == torch::kInt64, "Upper bound type must be torch::kInt64");
    OVEN_ASSERT((lower_bound < upper_bound).sum().item<bool>() == 0, "Upper bound is lower than lower bound. What the...");
}

c10::IntArrayRef MultiDiscreteSpace::shape() const {
    return upper_bound_.sizes();
}

torch::Tensor MultiDiscreteSpace::sample() const {
    torch::Tensor random = torch::rand(lower_bound_.sizes(), torch::TensorOptions().dtype(torch::kFloat64).device(lower_bound_.device()));
    auto diff = upper_bound_ - lower_bound_;
    return (random * diff.to(torch::kFloat64)).to(torch::kInt64) + lower_bound_;
}

bool MultiDiscreteSpace::contains(const torch::Tensor& x) const {
    OVEN_TENSOR_BINOP_CHECKLIST_STRICT(lower_bound_, x);

    return (lower_bound_ <= x).all().item<bool>() && (x < upper_bound_).all().item<bool>();
}

const torch::Tensor& MultiDiscreteSpace::lower_bound() const {
    return lower_bound_;
}

const torch::Tensor& MultiDiscreteSpace::upper_bound() const {
    return upper_bound_;
}
}