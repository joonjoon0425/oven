#include <memory>
#include <oven/tray/tray.hpp>
#include <oven/tray/trayimpl.hpp>

namespace oven {

namespace detail {
    // helper function for computing stride from given shape
    SmallVector compute_stride(const SmallVector& shape) {
        SmallVector ret(shape.size(), 1);
        for (int64_t i = shape.size() - 1; i >= 1; i--) {
            ret[i - 1] = ret[i] * shape[i];
        }

        return ret;
    }    
}// namespace oven::detail

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

}// namespace oven