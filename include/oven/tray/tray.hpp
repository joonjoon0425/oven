#pragma once

#include <memory>
#include <cstdint>
#include <oven/tray/types.hpp>
#include <oven/tray/trayimpl.hpp>
#include <oven/utils/intrusive_ptr.hpp>
#include <oven/tray/operations.hpp>
#include <torch/csrc/autograd/generated/variable_factories.h>

namespace oven {

class TrayImpl;

// A simple numpy-like object <--- It's not simple anymore....
class Tray {
private:
    intrusive_ptr<TrayImpl> impl_;
    
public:
    Tray(){ impl_ = nullptr;};
    Tray(intrusive_ptr<TrayImpl> impl);
    const SmallVector& shape() const;
    const SmallVector& stride() const;
    const int64_t ndim() const;
    const DType& dtype() const;
    const Device& device() const;
    const std::shared_ptr<void> data() const;
    const int64_t numel() const;

    // arithmetic operators
    Tray operator+(const Tray& other) const;
    Tray operator-(const Tray& other) const;
    Tray operator*(const Tray& other) const;
    Tray operator/(const Tray& other) const;
    // compare operators
    Tray operator<(const Tray& other) const;
    Tray operator<=(const Tray& other) const;
    Tray operator>(const Tray& other) const;
    Tray operator>=(const Tray& other) const;
    Tray operator==(const Tray& other) const;
    Tray operator!=(const Tray& other) const;
    
}; // class Tray

// Tray Constructors
Tray zeros(const SmallVector& shape, DType dtype = oven::kFloat32, Device device = Device::CPU);
Tray ones(const SmallVector& shape, DType dtype = oven::kFloat32, Device device = Device::CPU);
Tray full(const SmallVector& shape, Scalar val, DType dtype = oven::DType::Undefined, Device device = Device::CPU);
// random
Tray rand(const SmallVector& shape, Scalar low = 0., Scalar high = 1., DType dtype = oven::kFloat32, Device device = Device::CPU);
Tray randint(const SmallVector& shape, Scalar low, Scalar high, DType dtype = oven::DType::kInt64, Device device = Device::CPU);
Tray randn(const SmallVector& shape, Scalar mean = 0., Scalar std = 1., DType dtype = oven::DType::kFloat32, Device device = Device::CPU);
} // namespace oven