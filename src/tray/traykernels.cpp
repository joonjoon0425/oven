#include "oven/tray/types.hpp"
#include "oven/utils/intrusive_ptr.hpp"
#include <concepts>
#include <cstdint>
#include <memory>
#include <optional>
#include <oven/tray/tray.hpp>
#include <oven/tray/utils.hpp>
#include <oven/tray/dispatcher.hpp>
#include <type_traits>

namespace oven {

template <typename T, typename BinOp>
void __cpu_binary_elementwise_kernel_template(T* c, const SmallVector& c_shape, const Tray& a, const Tray& b, BinOp&& op) {
    // I wish I could use openMP or any parallel techniques here!
    SmallVector c_stride = detail::compute_stride(c_shape);
    SmallVector coord(c_stride.size(), 0);
    int64_t total_size = c_stride[0] * c_shape[0];
    SmallVector a_stride = detail::get_broadcasted_stride(a.shape(), c_shape);
    SmallVector b_stride = detail::get_broadcasted_stride(b.shape(), c_shape);
    
    int64_t a_index = 0;
    int64_t b_index = 0;

    T* a_data = static_cast<T*>(a.data().get());
    T* b_data = static_cast<T*>(b.data().get());

    for(int64_t i = 0; i < total_size; i++) {
        // reset coord vector
        coord.assign(coord.size(), 0);
        detail::compute_coordinate(i, c_stride, coord);
        a_index = detail::compute_index(coord, a_stride);
        b_index = detail::compute_index(coord, b_stride);
        c[i] = op(a_data[a_index], b_data[b_index]);
    }
}

template <typename T, typename CompOp>
requires std::predicate<CompOp, T, T>
void __cpu_binary_compare_kernel_template(T* a, T* b, bool* c, const SmallVector& a_stride, const SmallVector& b_stride, const SmallVector& shape, CompOp&& op) {
    // I wish I could use openMP or any parallel techniques here!
    SmallVector c_stride = detail::compute_stride(shape);
    SmallVector coord(c_stride.size(), 0);
    int64_t total_size = c_stride[0] * shape[0];
    
    int64_t a_index = 0;
    int64_t b_index = 0;

    for(int64_t i = 0; i < total_size; i++) {
        // reset coord vector
        coord.assign(coord.size(), 0);
        detail::compute_coordinate(i, c_stride, coord);
        a_index = detail::compute_index(coord, a_stride);
        b_index = detail::compute_index(coord, b_stride);
        c[i] = op(a[a_index], b[b_index]);
    }
}

template <typename T>
void __cpu_ternery_kernel_template(bool* c, T* a, T* b, T* result, const SmallVector& a_stride, const SmallVector& b_stride, const SmallVector& c_stride, const SmallVector& shape) {
    SmallVector result_stride = detail::compute_stride(shape);
    SmallVector coord(result_stride.size(), 0);
    int64_t total_size = result_stride[0] * shape[0];

    int64_t a_index = 0;
    int64_t b_index = 0;
    int64_t c_index = 0;

    for(int64_t i = 0; i < total_size; i++) {
        coord.assign(coord.size(), 0);
        detail::compute_coordinate(i, result_stride, coord);
        a_index = detail::compute_index(coord, a_stride);
        b_index = detail::compute_index(coord, b_stride);
        c_index = detail::compute_index(coord, c_stride);
        result[i] = c[c_index] ? a[a_index] : b[b_index];
    }
}

template <typename T>
void __cpu_gather_kernel_template(const Tray& self, int64_t dim, const Tray& index, T* target) {
    TRAY_DISPATCH_INT_TYPES(index.dtype(), "__cpu_gather_kernel_template", [&] {
        const SmallVector& index_stride = index.stride();
        const SmallVector& self_stride = self.stride();
        SmallVector coord(index_stride.size(), 0);
        int64_t numel = index.numel();
        int64_t self_index = 0;
        for (int64_t i = 0; i < numel; i++) {
            coord.assign(coord.size(), 0);
            detail::compute_coordinate(i, index_stride, coord);
            coord[dim] = static_cast<scalar_t*>(index.data().get())[i];
            self_index = detail::compute_index(coord, self_stride);
            target[i] = static_cast<T*>(self.data().get())[self_index];
        }
    });
    
}

Tray __cpu_add_kernel(const Tray& a, const Tray& b) {
    auto broadcast_shape = CHECK_BINARY_BROADCAST(a, b, "add");
    return TRAY_DISPATCH_NONBOOL_TYPES(a.dtype(), "add", [&]{
        using T = scalar_t;
        std::shared_ptr<scalar_t> data(new scalar_t[detail::compute_numel(broadcast_shape)]);

        __cpu_binary_elementwise_kernel_template(data.get(), broadcast_shape, a, b, [](T a, T b) {return a + b;});
        return Tray(make_intrusive<TrayImpl>(broadcast_shape, detail::compute_stride(broadcast_shape), a.dtype(), Device::CPU, data));
    });
}

Tray __cpu_sub_kernel(const Tray& a, const Tray& b) {
    auto broadcast_shape = CHECK_BINARY_BROADCAST(a, b, "sub");
    return TRAY_DISPATCH_NONBOOL_TYPES(a.dtype(), "sub", [&]{
        using T = scalar_t;
        std::shared_ptr<scalar_t> data(new scalar_t[detail::compute_numel(broadcast_shape)]);

        __cpu_binary_elementwise_kernel_template(data.get(), broadcast_shape, a, b, [](T a, T b) {return a - b;});
        return Tray(make_intrusive<TrayImpl>(broadcast_shape, detail::compute_stride(broadcast_shape), a.dtype(), Device::CPU, data));
    });
}

Tray __cpu_div_kernel(const Tray& a, const Tray& b) {
    auto broadcast_shape = CHECK_BINARY_BROADCAST(a, b, "div");
    return TRAY_DISPATCH_NONBOOL_TYPES(a.dtype(), "div", [&]{
        using T = scalar_t;
        std::shared_ptr<scalar_t> data(new scalar_t[detail::compute_numel(broadcast_shape)]);

        __cpu_binary_elementwise_kernel_template(data.get(), broadcast_shape, a, b, [](T a, T b) {return a / b;});
        return Tray(make_intrusive<TrayImpl>(broadcast_shape, detail::compute_stride(broadcast_shape), a.dtype(), Device::CPU, data));
    });
}

Tray __cpu_mul_kernel(const Tray& a, const Tray& b) {
    auto broadcast_shape = CHECK_BINARY_BROADCAST(a, b, "mul");
    return TRAY_DISPATCH_NONBOOL_TYPES(a.dtype(), "mul", [&]{
        using T = scalar_t;
        std::shared_ptr<scalar_t> data(new scalar_t[detail::compute_numel(broadcast_shape)]);

        __cpu_binary_elementwise_kernel_template(data.get(), broadcast_shape, a, b, [](T a, T b) {return a * b;});
        return Tray(make_intrusive<TrayImpl>(broadcast_shape, detail::compute_stride(broadcast_shape), a.dtype(), Device::CPU, data));
    });
}

Tray __cpu_le_kernel(void* a, void* b, const SmallVector& a_stride, const SmallVector& b_stride, const SmallVector& shape, DType dtype) {
    int64_t total_size = 1;
    for (int64_t i = 0; i < shape.size(); i++) total_size *= shape[i];
    return TRAY_DISPATCH_NONBOOL_TYPES(dtype, "le", [&] {
        std::shared_ptr<bool> data(new bool[total_size]);

        __cpu_binary_compare_kernel_template(static_cast<scalar_t*>(a), static_cast<scalar_t*>(b), data.get(), a_stride, b_stride, shape, [](scalar_t a, scalar_t b) {return a < b;});
        return Tray(make_intrusive<TrayImpl>(shape, detail::compute_stride(shape), oven::kBool, Device::CPU, data));
    });
}

Tray __cpu_leq_kernel(void* a, void* b, const SmallVector& a_stride, const SmallVector& b_stride, const SmallVector& shape, DType dtype) {
    int64_t total_size = 1;
    for (int64_t i = 0; i < shape.size(); i++) total_size *= shape[i];
    return TRAY_DISPATCH_NONBOOL_TYPES(dtype, "le", [&] {
        std::shared_ptr<bool> data(new bool[total_size]);

        __cpu_binary_compare_kernel_template(static_cast<scalar_t*>(a), static_cast<scalar_t*>(b), data.get(), a_stride, b_stride, shape, [](scalar_t a, scalar_t b) {return a <= b;});
        return Tray(make_intrusive<TrayImpl>(shape, detail::compute_stride(shape), oven::kBool, Device::CPU, data));
    });
}

Tray __cpu_ge_kernel(void* a, void* b, const SmallVector& a_stride, const SmallVector& b_stride, const SmallVector& shape, DType dtype) {
    int64_t total_size = 1;
    for (int64_t i = 0; i < shape.size(); i++) total_size *= shape[i];
    return TRAY_DISPATCH_NONBOOL_TYPES(dtype, "le", [&] {
        std::shared_ptr<bool> data(new bool[total_size]);

        __cpu_binary_compare_kernel_template(static_cast<scalar_t*>(a), static_cast<scalar_t*>(b), data.get(), a_stride, b_stride, shape, [](scalar_t a, scalar_t b) {return a > b;});
        return Tray(make_intrusive<TrayImpl>(shape, detail::compute_stride(shape), oven::kBool, Device::CPU, data));
    });
}

Tray __cpu_geq_kernel(void* a, void* b, const SmallVector& a_stride, const SmallVector& b_stride, const SmallVector& shape, DType dtype) {
    int64_t total_size = 1;
    for (int64_t i = 0; i < shape.size(); i++) total_size *= shape[i];
    return TRAY_DISPATCH_NONBOOL_TYPES(dtype, "le", [&] {
        std::shared_ptr<bool> data(new bool[total_size]);

        __cpu_binary_compare_kernel_template(static_cast<scalar_t*>(a), static_cast<scalar_t*>(b), data.get(), a_stride, b_stride, shape, [](scalar_t a, scalar_t b) {return a >= b;});
        return Tray(make_intrusive<TrayImpl>(shape, detail::compute_stride(shape), oven::kBool, Device::CPU, data));
    });
}

Tray __cpu_eq_kernel(void* a, void* b, const SmallVector& a_stride, const SmallVector& b_stride, const SmallVector& shape, DType dtype) {
    int64_t total_size = 1;
    for (int64_t i = 0; i < shape.size(); i++) total_size *= shape[i];
    return TRAY_DISPATCH_NONBOOL_TYPES(dtype, "le", [&] {
        std::shared_ptr<bool> data(new bool[total_size]);

        __cpu_binary_compare_kernel_template(static_cast<scalar_t*>(a), static_cast<scalar_t*>(b), data.get(), a_stride, b_stride, shape, [](scalar_t a, scalar_t b) {return a == b;});
        return Tray(make_intrusive<TrayImpl>(shape, detail::compute_stride(shape), oven::kBool, Device::CPU, data));
    });
}

Tray __cpu_neq_kernel(void* a, void* b, const SmallVector& a_stride, const SmallVector& b_stride, const SmallVector& shape, DType dtype) {
    int64_t total_size = 1;
    for (int64_t i = 0; i < shape.size(); i++) total_size *= shape[i];
    return TRAY_DISPATCH_NONBOOL_TYPES(dtype, "le", [&] {
        std::shared_ptr<bool> data(new bool[total_size]);

        __cpu_binary_compare_kernel_template(static_cast<scalar_t*>(a), static_cast<scalar_t*>(b), data.get(), a_stride, b_stride, shape, [](scalar_t a, scalar_t b) {return a != b;});
        return Tray(make_intrusive<TrayImpl>(shape, detail::compute_stride(shape), oven::kBool, Device::CPU, data));
    });
}

Tray __cpu_ternery_kernel(bool* c, void* a, void* b, const SmallVector& a_stride, const SmallVector& b_stride, const SmallVector& c_stride, const SmallVector& shape, DType dtype) {
    int64_t total_size = 1;
    for (int64_t i = 0; i < shape.size(); i++) total_size *= shape[i];
    return TRAY_DISPATCH_ALL_TYPES(dtype, "ternery", [&] {
        std::shared_ptr<scalar_t> data(new scalar_t[total_size]);

        __cpu_ternery_kernel_template(c, static_cast<scalar_t*>(a), static_cast<scalar_t*>(b), data.get(), a_stride, b_stride, c_stride, shape);
        return Tray(make_intrusive<TrayImpl>(shape, detail::compute_stride(shape), dtype, Device::CPU, data));
    });
}

Tray __cpu_gather_kernel(const Tray& self, int64_t dim, const Tray& index) {
    return TRAY_DISPATCH_ALL_TYPES(self.dtype(), "gather", [&] {
        std::shared_ptr<scalar_t> data(new scalar_t[index.numel()]);
        __cpu_gather_kernel_template(self, dim, index, static_cast<scalar_t*>(data.get()));
        return Tray(make_intrusive<TrayImpl>(index.shape(), index.stride(), self.dtype(), Device::CPU, data));
    });
}

}// namespace oven

// register kernels here
TRAY_REGISTER(add, CPU, oven::__cpu_add_kernel);
TRAY_REGISTER(sub, CPU, oven::__cpu_sub_kernel);
TRAY_REGISTER(div, CPU, oven::__cpu_div_kernel);
TRAY_REGISTER(mul, CPU, oven::__cpu_mul_kernel);

TRAY_REGISTER(le, CPU, oven::__cpu_le_kernel);
TRAY_REGISTER(leq, CPU, oven::__cpu_leq_kernel);
TRAY_REGISTER(ge, CPU, oven::__cpu_ge_kernel);
TRAY_REGISTER(geq, CPU, oven::__cpu_geq_kernel);
TRAY_REGISTER(eq, CPU, oven::__cpu_eq_kernel);
TRAY_REGISTER(neq, CPU, oven::__cpu_neq_kernel);

TRAY_REGISTER(ternery, CPU, oven::__cpu_ternery_kernel);

TRAY_REGISTER(gather, CPU, oven::__cpu_gather_kernel);