#pragma once

#include <vector>
#include <cstdint>
namespace oven::environment {

class Space {
public:
    virtual ~Space() = default;
    virtual std::vector<int64_t> shape() const = 0;
    virtual bool contains() const = 0;
};

}