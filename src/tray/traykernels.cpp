#include "oven/tray/types.hpp"
#include "oven/utils/intrusive_ptr.hpp"
#include <concepts>
#include <cstdint>
#include <memory>
#include <oven/tray/operatorstruct.hpp>
#include <oven/tray/tray.hpp>
#include <oven/tray/utils.hpp>
#include <oven/tray/dispatcher.hpp>
#include <type_traits>

// All template kernels assume that the input Trays are all-checked for its validity.
// Currently, the gather and scatter kernels breaking the assumption, since the reduction operations are not implemented yet.
// Make an issue for this.

namespace oven {

template <typename T, typename UnOp>
requires requires (T a, UnOp op) {
    {op(a)} -> std::same_as<std::invoke_result_t<UnOp, T>>;
}
void __cpu_unary_elementwise_kernel_template(std::invoke_result_t<UnOp, T>* c, const Tray& a) {
    SmallVector c_stride = detail::compute_stride(a.shape());
    int64_t numel = a.numel();
    SmallVector coord(a.shape().size(), 0);

    int64_t a_idx = 0;
    T* a_data = static_cast<T*>(a.data().get());

    for (int64_t i = 0; i < numel; i++) {
        detail::compute_coordinate(i, c_stride, coord);
        a_idx = detail::compute_index(coord, a.stride());
        c[i] = UnOp::get_instance()(a_data[a_idx]);        
    }
}

template <typename T, typename BinOp>
requires requires (T a, T b, BinOp op) {
    {op(a, b)} -> std::same_as<std::invoke_result_t<BinOp, T, T>>;
}
void __cpu_binary_elementwise_kernel_template(std::invoke_result_t<BinOp, T, T>* c, const SmallVector& c_shape, const Tray& a, const Tray& b) {
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
        c[i] = BinOp::get_instance()(a_data[a_index], b_data[b_index]);
    }
}

template <typename T, typename BinOp>
requires requires (T a, T b, BinOp op)
{
    {op(a, b)} -> std::same_as<std::invoke_result_t<BinOp, T, T>>;
}
void __cpu_binary_scalar_tensor_kernel_template(std::invoke_result_t<BinOp, T, T>* dst, T scalar, const Tray& a) {
    SmallVector coord(a.shape().size(), 0);
    const SmallVector& a_stride = a.stride();
    const SmallVector& dst_stride = detail::compute_stride(a.shape());
    T* a_data = static_cast<T*>(a.data().get());
    int64_t a_idx = 0;

    for (int64_t i = 0; i < a.numel(); i++) {
        detail::compute_coordinate(i, dst_stride, coord);
        a_idx = detail::compute_index(coord, a_stride);
        dst[i] = BinOp::get_instance()(scalar, a_data[a_idx]);
    }
}

template <typename T, typename BinOp>
requires requires (T a, T b, BinOp op)
{
    {op(a, b)} -> std::same_as<std::invoke_result_t<BinOp, T, T>>;
}
void __cpu_binary_tensor_scalar_kernel_template(std::invoke_result_t<BinOp, T, T>* dst, const Tray& a, T scalar) {
    SmallVector coord(a.shape().size(), 0);
    const SmallVector& a_stride = a.stride();
    const SmallVector& dst_stride = detail::compute_stride(a.shape());
    T* a_data = static_cast<T*>(a.data().get());
    int64_t a_idx = 0;

    for (int64_t i = 0; i < a.numel(); i++) {
        detail::compute_coordinate(i, dst_stride, coord);
        a_idx = detail::compute_index(coord, a_stride);
        dst[i] = BinOp::get_instance()(a_data[a_idx], scalar);
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

template <typename UnOp>
Tray __cpu_unary_elementwise_kernel(const Tray& a, UnOp op) {
    return TRAY_DISPATCH_ALL_TYPES(a.dtype(), UnOp::name, [&] {
        using result_t = std::invoke_result_t<UnOp, scalar_t>;
        std::shared_ptr<result_t> data(new result_t[a.numel()]);
        __cpu_unary_elementwise_kernel_template<scalar_t, UnOp>(data.get(), a);

        return Tray(make_intrusive<TrayImpl>(a.shape(), detail::compute_stride(a.shape()), detail::CppTypeToDType_v<result_t>, Device::CPU, data));
    });
}

template <typename BinOp>
Tray __cpu_binary_elementwise_kernel(const Tray& a, const Tray& b) {
    auto broadcasted_shape = CHECK_BINARY_BROADCAST(a, b, BinOp::name);
    return TRAY_DISPATCH_ALL_TYPES(a.dtype(), BinOp::name, [&] {
        using result_t = std::invoke_result_t<BinOp, scalar_t, scalar_t>;
        std::shared_ptr<result_t> data(new result_t[detail::compute_numel(broadcasted_shape)]);
        __cpu_binary_elementwise_kernel_template<scalar_t, BinOp>(data.get(), broadcasted_shape, a, b);

        return Tray(make_intrusive<TrayImpl>(broadcasted_shape, detail::compute_stride(broadcasted_shape), detail::CppTypeToDType_v<result_t>, Device::CPU, data));
    });
}

template <typename BinOp>
Tray __cpu_binary_scalar_tensor_kernel(const Scalar& scalar, const Tray& self) {
    return TRAY_DISPATCH_ALL_TYPES(self.dtype(), BinOp::name, [&] {
        return std::visit([&](const auto& v) {
            using T = std::decay_t<decltype(v)>;
            OVEN_ASSERT((std::is_same_v<T, scalar_t>), "DType does not matches in scalar-tensor operation " + BinOp::name);
            using result_t = std::invoke_result_t<BinOp, scalar_t, scalar_t>;
            std::shared_ptr<result_t> data(new result_t[self.numel()]);
            __cpu_binary_scalar_tensor_kernel_template<scalar_t, BinOp>(data.get(), v, self);
            return Tray(make_intrusive<TrayImpl>(self.shape(), detail::compute_stride(self.shape()), detail::CppTypeToDType_v<result_t>, Device::CPU, data));
        }, scalar);
    });
}

template <typename BinOp>
Tray __cpu_binary_tensor_scalar_kernel(const Tray& self, const Scalar& scalar) {
    return TRAY_DISPATCH_ALL_TYPES(self.dtype(), BinOp::name, [&] {
        return std::visit([&](const auto& v) {
            using T = std::decay_t<decltype(v)>;
            OVEN_ASSERT((std::is_same_v<T, scalar_t>), "DType does not matches in tensor-scalar operation " + BinOp::name);
            using result_t = std::invoke_result_t<BinOp, scalar_t, scalar_t>;
            std::shared_ptr<result_t> data(new result_t[self.numel()]);
            __cpu_binary_tensor_scalar_kernel_template<scalar_t, BinOp>(data.get(), self, v);
            return Tray(make_intrusive<TrayImpl>(self.shape(), detail::compute_stride(self.shape()), detail::CppTypeToDType_v<result_t>, Device::CPU, data));
        }, scalar);
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
#define TRAY_REGISTER_BINOP_CPU(binop, possible_types, OpStruct)\
TRAY_REGISTER(binop, CPU, possible_types, oven::__cpu_binary_elementwise_kernel<oven::detail::OpStruct>);\
TRAY_REGISTER(binop##_ts, CPU, possible_types, oven::__cpu_binary_tensor_scalar_kernel<oven::detail::OpStruct>);\
TRAY_REGISTER(binop##_st, CPU, possible_types, oven::__cpu_binary_scalar_tensor_kernel<oven::detail::OpStruct>);\

TRAY_REGISTER_BINOP_CPU(add, oven::all_types, AddOp);
TRAY_REGISTER_BINOP_CPU(sub, oven::all_types, SubOp);
TRAY_REGISTER_BINOP_CPU(mul, oven::all_types, MulOp);
TRAY_REGISTER_BINOP_CPU(div, oven::all_types, DivOp);
TRAY_REGISTER_BINOP_CPU(le, oven::all_types, LeOp);
TRAY_REGISTER_BINOP_CPU(leq, oven::all_types, LeqOp);
TRAY_REGISTER_BINOP_CPU(ge, oven::all_types, GeOp);
TRAY_REGISTER_BINOP_CPU(geq, oven::all_types, GeqOp);
TRAY_REGISTER_BINOP_CPU(eq, oven::all_types, EqOp);
TRAY_REGISTER_BINOP_CPU(neq, oven::all_types, NeqOp);

TRAY_REGISTER(ternery, CPU, oven::all_types,oven::__cpu_ternery_kernel);

TRAY_REGISTER(gather, CPU, oven::all_types,oven::__cpu_gather_kernel);
TRAY_REGISTER(scatter_, CPU, oven::all_types,oven::__cpu_scatter_inplace_kernel);

TRAY_REGISTER(neg, CPU, oven::all_types, oven::__cpu_unary_elementwise_kernel<oven::detail::NegOp>);
TRAY_REGISTER(recip, CPU, oven::all_types, oven::__cpu_unary_elementwise_kernel<oven::detail::RecipOp>);
TRAY_REGISTER(log, CPU, oven::detail::LogOp::possible_types, oven::__cpu_unary_elementwise_kernel<oven::detail::LogOp>);
TRAY_REGISTER(exp, CPU, oven::detail::ExpOp::possible_types, oven::__cpu_unary_elementwise_kernel<oven::detail::ExpOp>);