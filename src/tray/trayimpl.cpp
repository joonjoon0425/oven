#include "oven/utils/intrusive_ptr.hpp"
#include <oven/tray/trayimpl.hpp>

namespace oven {

TrayImpl::TrayImpl(const SmallVector& shape, const SmallVector& stride, const DType& dtype, const Device& device, std::shared_ptr<void> data)
: RefCountable(),
  shape_(shape),
  stride_(stride),
  dtype_(dtype),
  device_(device),
  data_(data) {}

TrayImpl::TrayImpl(const TrayImpl& other)
: RefCountable(), // Note that we are making a new TrayImpl object. That is, if we copy the RefCount, two TrayImpl objects will have same lifetime.
  shape_(other.shape_),
  stride_(other.stride_),
  dtype_(other.dtype_),
  device_(other.device_),
  data_(other.data_)
{

}

TrayImpl::TrayImpl(TrayImpl&& other) noexcept
: RefCountable(),
  shape_(std::move(other.shape_)),
  stride_(std::move(other.stride_)),
  dtype_(std::move(other.dtype_)),
  device_(std::move(other.device_)),
  data_(std::move(other.data_))
{

}

}