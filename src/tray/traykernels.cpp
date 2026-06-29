#include "oven/utils/intrusive_ptr.hpp"
#include <cstdint>
#include <memory>
#include <optional>
#include <oven/tray/tray.hpp>
#include <oven/tray/utils.hpp>
#include <oven/tray/dispatcher.hpp>

namespace oven {

template <typename T, typename BinOp>
void __cpu_binary_elementwise_kernel_template(T* a, T* b, T* c, const SmallVector& a_stride, const SmallVector& b_stride, const SmallVector& shape, BinOp&& op) {
    // I wish I could use openMP or any parallel techniques here!
    SmallVector c_stride = detail::compute_stride(shape);
    SmallVector coord(c_stride.size(), 0);
    int64_t total_size = c_stride[0] * shape[0];
    
    int64_t a_index = 0;
    int64_t b_index = 0;

    for(int64_t i = 0; i < total_size; i++) {
        // reset coord vector
        coord.assign(coord.size(), 0);
        detail::compute_coordinate(i, total_size, c_stride, coord);
        a_index = detail::compute_index(coord, a_stride);
        b_index = detail::compute_index(coord, b_stride);
        c[i] = op(a[a_index], b[b_index]);
    }
}

Tray __cpu_add_kernel(void* a, void* b, const SmallVector& a_stride, const SmallVector& b_stride, const SmallVector& shape, DType dtype) {
    int64_t total_size = 1;
    for (int64_t i = 0; i < shape.size(); i++) total_size *= shape[i];
    return TRAY_DISPATCH_NONBOOL_TYPES(dtype, "add", [&]{
        using T = scalar_t;
        std::shared_ptr<scalar_t> data(new scalar_t[total_size]);

        __cpu_binary_elementwise_kernel_template(static_cast<T*>(a), static_cast<T*>(b), data.get(), a_stride, b_stride, shape, [](T a, T b) {return a + b;});
        return Tray(make_intrusive<TrayImpl>(shape, detail::compute_stride(shape), dtype, Device::CPU, data));
    });
}

Tray __cpu_sub_kernel(void* a, void* b, const SmallVector& a_stride, const SmallVector& b_stride, const SmallVector& shape, DType dtype) {
    int64_t total_size = 1;
    for (int64_t i = 0; i < shape.size(); i++) total_size *= shape[i];
    return TRAY_DISPATCH_NONBOOL_TYPES(dtype, "sub", [&]{
        using T = scalar_t;
        std::shared_ptr<scalar_t> data(new scalar_t[total_size]);
        
        __cpu_binary_elementwise_kernel_template(static_cast<T*>(a), static_cast<T*>(b), data.get(), a_stride, b_stride, shape, [](T a, T b) {return a - b;});
        return Tray(make_intrusive<TrayImpl>(shape, detail::compute_stride(shape), dtype, Device::CPU, data));
    });
}

Tray __cpu_div_kernel(void* a, void* b, const SmallVector& a_stride, const SmallVector& b_stride, const SmallVector& shape, DType dtype) {
    int64_t total_size = 1;
    for (int64_t i = 0; i < shape.size(); i++) total_size *= shape[i];
    return TRAY_DISPATCH_NONBOOL_TYPES(dtype, "div", [&]{
        using T = scalar_t;
        std::shared_ptr<scalar_t> data(new scalar_t[total_size]);
        
        __cpu_binary_elementwise_kernel_template(static_cast<T*>(a), static_cast<T*>(b), data.get(), a_stride, b_stride, shape, [](T a, T b) {return a / b;});
        return Tray(make_intrusive<TrayImpl>(shape, detail::compute_stride(shape), dtype, Device::CPU, data));
    });
}

Tray __cpu_mul_kernel(void* a, void* b, const SmallVector& a_stride, const SmallVector& b_stride, const SmallVector& shape, DType dtype) {
    int64_t total_size = 1;
    for (int64_t i = 0; i < shape.size(); i++) total_size *= shape[i];
    return TRAY_DISPATCH_ALL_TYPES(dtype, "mul", [&]{
        using T = scalar_t;
        std::shared_ptr<scalar_t> data(new scalar_t[total_size]);
        
        __cpu_binary_elementwise_kernel_template(static_cast<T*>(a), static_cast<T*>(b), data.get(), a_stride, b_stride, shape, [](T a, T b) {return a * b;});
        return Tray(make_intrusive<TrayImpl>(shape, detail::compute_stride(shape), dtype, Device::CPU, data));
    });
}

}// namespace oven

// register kernels here
TRAY_REGISTER(add, CPU, oven::__cpu_add_kernel);
TRAY_REGISTER(sub, CPU, oven::__cpu_sub_kernel);
TRAY_REGISTER(div, CPU, oven::__cpu_div_kernel);
TRAY_REGISTER(mul, CPU, oven::__cpu_mul_kernel);
