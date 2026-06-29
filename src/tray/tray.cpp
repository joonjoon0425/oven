#include "oven/tray/dispatcher.hpp"
#include "oven/tray/types.hpp"
#include "oven/tray/utils.hpp"
#include <memory>
#include <optional>
#include <oven/utils/assert.hpp>
#include <oven/tray/tray.hpp>
#include <oven/tray/trayimpl.hpp>

namespace oven {

Tray::Tray(intrusive_ptr<TrayImpl> impl) : impl_(impl) {}

const SmallVector& Tray::shape() const {
    return impl_->shape_;
}

const SmallVector& Tray::stride() const {
    return impl_->stride_;
}

const int64_t Tray::ndim() const {
    return impl_->shape_.size();
}

const DType& Tray::dtype() const {
    return impl_->dtype_;
}

const Device& Tray::device() const {
    return impl_->device_;
}

const std::shared_ptr<void> Tray::data() const {
    return impl_->data_;
}

Tray Tray::operator+(const Tray& other) const {
    std::optional<SmallVector> shape = detail::broadcastable(this->shape(), other.shape());
    OVEN_ASSERT(shape != std::nullopt, "Not broadcastable for add operation.");
    // later, make SmallerVector print function for debugging...
    // checking logics for binary operation goes here...
    OVEN_TRAY_BINOP_CHECKLIST((*this), other);
    // get broadcasted strides
    auto a_stride = detail::get_broadcasted_stride(this->shape(), *shape);
    auto b_stride = detail::get_broadcasted_stride(other.shape(), *shape);
    return detail::Dispatcher::get_instance()
        .dispatch<oven::ElementWiseKernelType>
            ({detail::OpCode::add, Device::CPU},
                this->data().get(), other.data().get(), a_stride, b_stride, *shape, this->dtype());
}


Tray Tray::operator-(const Tray& other) const {
    std::optional<SmallVector> shape = detail::broadcastable(this->shape(), other.shape());
    OVEN_ASSERT(shape != std::nullopt, "Not broadcastable for add operation.");
    // later, make SmallerVector print function for debugging...
    // checking logics for binary operation goes here...
    OVEN_TRAY_BINOP_CHECKLIST((*this), other);
    // get broadcasted strides
    auto a_stride = detail::get_broadcasted_stride(this->shape(), *shape);
    auto b_stride = detail::get_broadcasted_stride(other.shape(), *shape);
    return detail::Dispatcher::get_instance()
        .dispatch<oven::ElementWiseKernelType>
            ({detail::OpCode::sub, Device::CPU},
                this->data().get(), other.data().get(), a_stride, b_stride, *shape, this->dtype());
}

Tray Tray::operator*(const Tray& other) const {
    std::optional<SmallVector> shape = detail::broadcastable(this->shape(), other.shape());
    OVEN_ASSERT(shape != std::nullopt, "Not broadcastable for add operation.");
    // later, make SmallerVector print function for debugging...
    // checking logics for binary operation goes here...
    OVEN_TRAY_BINOP_CHECKLIST((*this), other);
    // get broadcasted strides
    auto a_stride = detail::get_broadcasted_stride(this->shape(), *shape);
    auto b_stride = detail::get_broadcasted_stride(other.shape(), *shape);
    return detail::Dispatcher::get_instance()
        .dispatch<oven::ElementWiseKernelType>
            ({detail::OpCode::mul, Device::CPU},
                this->data().get(), other.data().get(), a_stride, b_stride, *shape, this->dtype());
}
Tray Tray::operator/(const Tray& other) const {
    std::optional<SmallVector> shape = detail::broadcastable(this->shape(), other.shape());
    OVEN_ASSERT(shape != std::nullopt, "Not broadcastable for add operation.");
    // later, make SmallerVector print function for debugging...
    // checking logics for binary operation goes here...
    OVEN_TRAY_BINOP_CHECKLIST((*this), other);
    // get broadcasted strides
    auto a_stride = detail::get_broadcasted_stride(this->shape(), *shape);
    auto b_stride = detail::get_broadcasted_stride(other.shape(), *shape);
    return detail::Dispatcher::get_instance()
        .dispatch<oven::ElementWiseKernelType>
            ({detail::OpCode::div, Device::CPU},
                this->data().get(), other.data().get(), a_stride, b_stride, *shape, this->dtype());
}

}// namespace oven