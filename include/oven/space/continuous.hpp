#pragma once
#include <ATen/core/ivalue.h>
#include <c10/util/ArrayRef.h>
#include <oven/space/space.hpp>

namespace oven::environment {

class ContinuousSpace : public Space {
private:
    torch::Tensor lower_bound_;
    torch::Tensor upper_bound_;
    
public:
    ContinuousSpace(torch::Tensor lower_bound, torch::Tensor upper_bound);

    c10::IntArrayRef shape() const override;
    torch::Tensor sample() const override;
    bool contains(const torch::Tensor& x) const override;

    const torch::Tensor& lower_bound() const;
    const torch::Tensor& upper_bound() const;

};

}