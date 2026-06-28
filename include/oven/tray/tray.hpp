#pragma once

#include <memory>
#include <vector>
#include <cstdint>
#include <oven/tray/types.hpp>

namespace oven {
using SmallVector = std::vector<int64_t>;

namespace detail {
    // helper function for computing stride from given shape
    SmallVector compute_stride(const SmallVector& shape);
}

class TrayImpl;

// A simple numpy-like object
class Tray {
private:
    std::shared_ptr<TrayImpl> impl_;
    
public:
    Tray(std::shared_ptr<TrayImpl> impl) : impl_(impl) {}
    Tray() : impl_(nullptr) {}
    const SmallVector& shape() const;
    const SmallVector& stride() const;
    const int64_t ndim() const;
    const DType& dtype() const;
    const Device& device() const;
    const std::shared_ptr<void> data() const;
}; // class Tray

Tray zeros(SmallVector shape, DType dtype, Device device = Device::CPU);

} // namespace oven