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
    SmallVector a_stride = detail::get_right_aligned_broadcasted_stride(a.shape(), a.stride(), c_shape);
    SmallVector b_stride = detail::get_right_aligned_broadcasted_stride(b.shape(), b.stride(), c_shape);
    
    int64_t a_index = 0;
    int64_t b_index = 0;

    T* a_data = static_cast<T*>(a.data().get());
    T* b_data = static_cast<T*>(b.data().get());

    for(int64_t i = 0; i < total_size; i++) {
        detail::compute_coordinate(i, c_stride, coord);
        a_index = detail::compute_index(coord, a_stride);
        b_index = detail::compute_index(coord, b_stride);
        c[i] = op(a_data[a_index], b_data[b_index]);
    }
}

template <typename T, typename CompOp>
requires std::predicate<CompOp, T, T>
void __cpu_binary_compare_kernel_template(bool* c, const Tray& a, const Tray& b, const SmallVector& c_shape, CompOp&& op) {
    // I wish I could use openMP or any parallel techniques here!
    SmallVector c_stride = detail::compute_stride(c_shape);
    SmallVector coord(c_stride.size(), 0);
    int64_t total_size = c_stride[0] * c_shape[0];
    SmallVector a_stride = detail::get_right_aligned_broadcasted_stride(a.shape(), a.stride(), c_shape);
    SmallVector b_stride = detail::get_right_aligned_broadcasted_stride(b.shape(), b.stride(), c_shape);
    
    int64_t a_index = 0;
    int64_t b_index = 0;

    T* a_data = static_cast<T*>(a.data().get());
    T* b_data = static_cast<T*>(b.data().get());

    for(int64_t i = 0; i < total_size; i++) {
        detail::compute_coordinate(i, c_stride, coord);
        a_index = detail::compute_index(coord, a_stride);
        b_index = detail::compute_index(coord, b_stride);
        c[i] = op(a_data[a_index], b_data[b_index]);
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
        detail::compute_coordinate(i, result_stride, coord);
        a_index = detail::compute_index(coord, a_stride);
        b_index = detail::compute_index(coord, b_stride);
        c_index = detail::compute_index(coord, c_stride);
        result[i] = c[c_index] ? a[a_index] : b[b_index];
    }
}

template <typename T>
void __cpu_gather_kernel_template(const Tray& self, int64_t dim, const Tray& index, const SmallVector& broadcasted_self, const SmallVector& broadcasted_index, T* target) {
    TRAY_DISPATCH_INT_TYPES(index.dtype(), "__cpu_gather_kernel_template", [&] {
        const SmallVector& index_stride = detail::get_left_aligned_broadcasted_stride(index.shape(), index.stride(), broadcasted_index);
        const SmallVector& self_stride = detail::get_left_aligned_broadcasted_stride(self.shape(), self.stride(), broadcasted_self);
        SmallVector result_stride = detail::compute_stride(broadcasted_index);
        SmallVector coord(index_stride.size(), 0);
        int64_t numel = result_stride[0] * index.shape()[0];
        int64_t self_index = 0;
        int64_t index_index = 0;
        scalar_t* index_data = static_cast<scalar_t*>(index.data().get());
        T* self_data = static_cast<T*>(self.data().get());
        for (int64_t i = 0; i < numel; i++) {
            detail::compute_coordinate(i, result_stride, coord);
            index_index = detail::compute_index(coord, index_stride);
            coord[dim] = index_data[index_index];
            OVEN_ASSERT(index_data[index_index] >= 0 && index_data[index_index] < self.shape()[dim], "index value out of range.");
            self_index = detail::compute_index(coord, self_stride);
            
            target[i] = self_data[self_index];
        }
    });
}

template <typename T>
void __cpu_scatter_inplace_kernel_template(const Tray& self, int64_t dim, const Tray& index, const Tray& src, const SmallVector& broadcasted_index) {
    TRAY_DISPATCH_INT_TYPES(index.dtype(), "__cpu_scatter_inplace_kernel_template", [&]{
        // broadcasted strides
        // note that the index and src has same shape, when calling scatter_ operation.
        // but let's save this for later implementation of more flexible broadcast...
        // const SmallVector src_stride = detail::get_right_aligned_broadcasted_stride(src.shape(), src.stride(), broadcasted_index);
        const SmallVector index_stride = detail::get_left_aligned_broadcasted_stride(index.shape(), index.stride(), broadcasted_index);
       
        // numel of index
        const SmallVector stride = detail::compute_stride(broadcasted_index);
        const int64_t numel = detail::compute_numel(broadcasted_index);

        // raw data ptr
        T* self_data = static_cast<T*>(self.data().get());
        scalar_t* index_data = static_cast<scalar_t*>(index.data().get());
        T* src_data = static_cast<T*>(src.data().get());
         
        // indices
        int64_t self_index = 0;
        int64_t index_index = 0;
        int64_t src_index = 0;
        // coord for computation of index
        SmallVector coord(self.shape().size(), 0);
        // keep in mind that src and self are now in the same shape, except the dim
        for (int64_t i = 0; i < numel; i++) {
            // calculate current coordinate of 
            detail::compute_coordinate(i, stride, coord);
            
            index_index = detail::compute_index(coord, index_stride);
            src_index = detail::compute_index(coord, index_stride/*src_stride*/);
            
            coord[dim] = index_data[index_index];

            OVEN_ASSERT(index_data[index_index] >= 0 && index_data[index_index] < self.shape()[dim], "index value out of range.");

            self_index = detail::compute_index(coord, self.stride());
            self_data[self_index] = src_data[src_index];
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

Tray __cpu_le_kernel(const Tray& a, const Tray& b) {
    auto broadcast_shape = CHECK_BINARY_BROADCAST(a, b, "le");
    return TRAY_DISPATCH_NONBOOL_TYPES(a.dtype(), "le", [&] {
        std::shared_ptr<bool> data(new bool[detail::compute_numel(broadcast_shape)]);

        __cpu_binary_compare_kernel_template<scalar_t>(data.get(), a, b, broadcast_shape, [](scalar_t a, scalar_t b) {return a < b;});
        return Tray(make_intrusive<TrayImpl>(broadcast_shape, detail::compute_stride(broadcast_shape), oven::kBool, Device::CPU, data));
    });
}

Tray __cpu_leq_kernel(const Tray& a, const Tray& b) {
    auto broadcast_shape = CHECK_BINARY_BROADCAST(a, b, "leq");
    return TRAY_DISPATCH_NONBOOL_TYPES(a.dtype(), "leq", [&] {
        std::shared_ptr<bool> data(new bool[detail::compute_numel(broadcast_shape)]);

        __cpu_binary_compare_kernel_template<scalar_t>(data.get(), a, b, broadcast_shape, [](scalar_t a, scalar_t b) {return a <= b;});
        return Tray(make_intrusive<TrayImpl>(broadcast_shape, detail::compute_stride(broadcast_shape), oven::kBool, Device::CPU, data));
    });
}

Tray __cpu_ge_kernel(const Tray& a, const Tray& b) {
    auto broadcast_shape = CHECK_BINARY_BROADCAST(a, b, "ge");
    return TRAY_DISPATCH_NONBOOL_TYPES(a.dtype(), "ge", [&] {
        std::shared_ptr<bool> data(new bool[detail::compute_numel(broadcast_shape)]);

        __cpu_binary_compare_kernel_template<scalar_t>(data.get(), a, b, broadcast_shape, [](scalar_t a, scalar_t b) {return a > b;});
        return Tray(make_intrusive<TrayImpl>(broadcast_shape, detail::compute_stride(broadcast_shape), oven::kBool, Device::CPU, data));
    });
}

Tray __cpu_geq_kernel(const Tray& a, const Tray& b) {
    auto broadcast_shape = CHECK_BINARY_BROADCAST(a, b, "geq");
    return TRAY_DISPATCH_NONBOOL_TYPES(a.dtype(), "geq", [&] {
        std::shared_ptr<bool> data(new bool[detail::compute_numel(broadcast_shape)]);

        __cpu_binary_compare_kernel_template<scalar_t>(data.get(), a, b, broadcast_shape, [](scalar_t a, scalar_t b) {return a >= b;});
        return Tray(make_intrusive<TrayImpl>(broadcast_shape, detail::compute_stride(broadcast_shape), oven::kBool, Device::CPU, data));
    });
}

Tray __cpu_eq_kernel(const Tray& a, const Tray& b) {
    auto broadcast_shape = CHECK_BINARY_BROADCAST(a, b, "eq");
    return TRAY_DISPATCH_NONBOOL_TYPES(a.dtype(), "eq", [&] {
        std::shared_ptr<bool> data(new bool[detail::compute_numel(broadcast_shape)]);

        __cpu_binary_compare_kernel_template<scalar_t>(data.get(), a, b, broadcast_shape, [](scalar_t a, scalar_t b) {return a == b;});
        return Tray(make_intrusive<TrayImpl>(broadcast_shape, detail::compute_stride(broadcast_shape), oven::kBool, Device::CPU, data));
    });
}

Tray __cpu_neq_kernel(const Tray& a, const Tray& b) {
    auto broadcast_shape = CHECK_BINARY_BROADCAST(a, b, "neq");
    return TRAY_DISPATCH_NONBOOL_TYPES(a.dtype(), "neq", [&] {
        std::shared_ptr<bool> data(new bool[detail::compute_numel(broadcast_shape)]);

        __cpu_binary_compare_kernel_template<scalar_t>(data.get(), a, b, broadcast_shape, [](scalar_t a, scalar_t b) {return a != b;});
        return Tray(make_intrusive<TrayImpl>(broadcast_shape, detail::compute_stride(broadcast_shape), oven::kBool, Device::CPU, data));
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
    auto [broadcasted_self, broadcasted_index] = CHECK_GATHER_BROADCAST(self, dim, index);
    return TRAY_DISPATCH_ALL_TYPES(self.dtype(), "gather", [&] {
        std::shared_ptr<scalar_t> data(new scalar_t[detail::compute_numel(broadcasted_index)]);
        __cpu_gather_kernel_template(self, dim, index, broadcasted_self, broadcasted_index, static_cast<scalar_t*>(data.get()));
        return Tray(make_intrusive<TrayImpl>(broadcasted_index, detail::compute_stride(broadcasted_index), self.dtype(), Device::CPU, data));
    });
}

void __cpu_scatter_inplace_kernel(const Tray& self, int64_t dim, const Tray& index, const Tray& src) {
    auto broadcasted_index = CHECK_SCATTER_BROADCAST(self, dim, index, src);
    return TRAY_DISPATCH_ALL_TYPES(self.dtype(), "scatter_", [&]{
        __cpu_scatter_inplace_kernel_template<scalar_t>(self, dim, index, src, broadcasted_index);
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
TRAY_REGISTER(scatter_, CPU, oven::__cpu_scatter_inplace_kernel);