#pragma once
#include <c10/util/ArrayRef.h>
#include <torch/torch.h>
// Basic interface of all state spaces and action spaces.
// All state spaces and action spaces must inherit oven::environment::Space

using Shape = std::vector<int64_t>;

namespace oven::environment{
class Space {
public:
    virtual ~Space() = default;

    // returns the shape of the space
    // It will be used when constructing Q-network or any related operation
    virtual c10::IntArrayRef shape() const = 0;

    // returns the random sample of a space
    virtual torch::Tensor sample() const = 0;

    // checks if the given tensor is contained in a space
    virtual bool contains(const torch::Tensor& x) const = 0;
};

}