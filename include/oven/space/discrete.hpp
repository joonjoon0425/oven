#pragma once
#include <c10/util/ArrayRef.h>
#include <oven/space/space.hpp>

namespace oven::environment {

class DiscreteSpace : public Space {
private:
    int64_t n_;

public:
    DiscreteSpace(int64_t n);

    c10::IntArrayRef shape() const override;
    torch::Tensor sample() const override;
    bool contains(const torch::Tensor& x) const override;

    int64_t n() const;
};

}