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

Tray Tray::operator+(const Scalar& scalar) const {return add((*this), scalar);}
Tray Tray::operator-(const Scalar& scalar) const {return sub((*this), scalar);}
Tray Tray::operator*(const Scalar& scalar) const {return mul((*this), scalar);}
Tray Tray::operator/(const Scalar& scalar) const {return div((*this), scalar);}

Tray operator+(const Scalar& scalar, const Tray& tray) {return add(scalar, tray);}
Tray operator-(const Scalar& scalar, const Tray& tray) {return sub(scalar, tray);}
Tray operator*(const Scalar &scalar, const Tray &tray) {return mul(scalar, tray);}
Tray operator/(const Scalar &scalar, const Tray &tray) {return div(scalar, tray);}

Tray Tray::operator<(const Tray& other) const {return le((*this), other);}
Tray Tray::operator<=(const Tray& other) const {return leq((*this), other);}
Tray Tray::operator>(const Tray& other) const {return ge((*this), other);}
Tray Tray::operator>=(const Tray& other) const {return geq((*this), other);}
Tray Tray::operator==(const Tray& other) const {return eq((*this), other);}
Tray Tray::operator!=(const Tray& other) const {return neq((*this), other);}

Tray Tray::operator<(const Scalar& scalar) const {return le((*this), scalar);}
Tray Tray::operator<=(const Scalar& scalar) const {return leq((*this), scalar);}
Tray Tray::operator>(const Scalar& scalar) const {return ge((*this), scalar);}
Tray Tray::operator>=(const Scalar& scalar) const {return geq((*this), scalar);}
Tray Tray::operator==(const Scalar& scalar) const {return eq((*this), scalar);}
Tray Tray::operator!=(const Scalar& scalar) const {return neq((*this), scalar);}

Tray operator<(const Scalar& scalar, const Tray& tray) {return le(scalar, tray);}
Tray operator<=(const Scalar& scalar, const Tray& tray) {return leq(scalar, tray);}
Tray operator>(const Scalar& scalar, const Tray& tray) {return ge(scalar, tray);}
Tray operator>=(const Scalar& scalar, const Tray& tray) {return geq(scalar, tray);}
Tray operator==(const Scalar& scalar, const Tray& tray) {return eq(scalar, tray);}
Tray operator!=(const Scalar& scalar, const Tray& tray) {return neq(scalar, tray);}

Tray Tray::operator-() const {return neg(*this);}
}// namespace oven











