#pragma once

#include <memory>
#include <vector>
#include <oven/tray/types.hpp>
#include <oven/utils/intrusive_ptr.hpp>

namespace oven {
class Tray;

class TrayImpl : public RefCountable {
private:
    SmallVector shape_;
    SmallVector stride_;
    DType dtype_;
    Device device_;

    std::shared_ptr<void> data_;

public:
    friend Tray;

    TrayImpl(const SmallVector& shape, const SmallVector& stride, const DType& dtype, const Device& device, std::shared_ptr<void> data);
    TrayImpl(const TrayImpl& other);
    TrayImpl(TrayImpl&& other) noexcept;

    TrayImpl& operator=(TrayImpl) = delete;
};
}