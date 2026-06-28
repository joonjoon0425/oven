#pragma once

#include <memory>
#include <cstdint>
#include <oven/tray/types.hpp>
#include <oven/tray/trayimpl.hpp>
#include <oven/utils/intrusive_ptr.hpp>

namespace oven {

namespace detail {
    // helper function for computing stride from given shape
    SmallVector compute_stride(const SmallVector& shape);
}

class TrayImpl;

// A simple numpy-like object
class Tray {
private:
    intrusive_ptr<TrayImpl> impl_;
    
public:
    Tray(intrusive_ptr<TrayImpl> impl);
    const SmallVector& shape() const;
    const SmallVector& stride() const;
    const int64_t ndim() const;
    const DType& dtype() const;
    const Device& device() const;
    const std::shared_ptr<void> data() const;
}; // class Tray

Tray zeros(SmallVector shape, DType dtype, Device device = Device::CPU);

} // namespace oven