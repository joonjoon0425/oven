#pragma once

#include <memory>
#include <oven/tray/types.hpp>
#include <oven/tray/tray.hpp>

namespace oven {

class TrayImpl {
private:
    SmallVector shape_;
    SmallVector stride_;
    DType dtype_;
    Device device_;

    std::shared_ptr<void> data_;

public:
    friend Tray;

    TrayImpl(const SmallVector& shape, const SmallVector& stride, const DType& dtype, const Device& device, std::shared_ptr<void> data)
    : shape_(shape),
      stride_(stride),
      dtype_(dtype),
      device_(device),
      data_(data) {}

    TrayImpl(const TrayImpl& other) = default;
    TrayImpl& operator=(const TrayImpl& other) = default;
    
    TrayImpl(TrayImpl&&) = default;
    TrayImpl& operator=(TrayImpl&&) = default;
};
}