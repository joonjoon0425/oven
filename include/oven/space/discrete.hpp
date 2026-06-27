#pragma once
#include <c10/util/ArrayRef.h>
#include <cstdint>
#include <oven/space/space.hpp>
// #include <random>

namespace oven::environment {

class DiscreteSpace : public Space {
private:
    int64_t n_;
    int64_t offset_;
    // std::mt19937 gen_;
    // std::uniform_int_distribution<int64_t> dist_;

public:
    DiscreteSpace(int64_t n, int64_t offset = 0);

    c10::IntArrayRef shape() const override;
    torch::Tensor sample() const override;
    bool contains(const torch::Tensor& x) const override;

    int64_t n() const;
    int64_t offset() const;
};

}