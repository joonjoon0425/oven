#include "oven/tray/types.hpp"
#include <memory>
#include <oven/utils/assert.hpp>
#include <oven/tray/tray.hpp>
#include <oven/tray/operations.hpp>

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

const int64_t Tray::numel() const {
    return impl_->numel_;
}

const std::shared_ptr<void> Tray::data() const {
    return impl_->data_;
}

Tray Tray::operator+(const Tray& other) const {return add((*this), other);}
Tray Tray::operator-(const Tray& other) const {return sub((*this), other);}
Tray Tray::operator*(const Tray& other) const {return mul((*this), other);}
Tray Tray::operator/(const Tray& other) const {return div((*this), other);}

Tray Tray::operator<(const Tray& other) const {return le((*this), other);}
Tray Tray::operator<=(const Tray& other) const {return leq((*this), other);}
Tray Tray::operator>(const Tray& other) const {return ge((*this), other);}
Tray Tray::operator>=(const Tray& other) const {return geq((*this), other);}
Tray Tray::operator==(const Tray& other) const{ return eq((*this), other);}
Tray Tray::operator!=(const Tray& other) const {return neq((*this), other);}


}// namespace oven