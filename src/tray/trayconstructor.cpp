#include "oven/tray/types.hpp"
#include "oven/utils/intrusive_ptr.hpp"
#include <algorithm>
#include <cstring>
#include <memory>
#include <oven/tray/tray.hpp>
#include <oven/tray/utils.hpp>
#include <oven/tray/dispatcher.hpp>
#include <random>
#include <variant>

namespace oven {

Tray __gpu_zeros() {
    // TODO
}

void __cpu_zeros(void* data, size_t memory_size) {
    std::memset(data, 0, memory_size);
}

void __cpu_full(void* data, const void* val, size_t n, DType dtype) {
    TRAY_DISPATCH_ALL_TYPES(dtype, "cpu_full", [&]{
        std::fill_n(static_cast<scalar_t*>(data), n, *(static_cast<const scalar_t*>(val))); 
    });
}

template <typename T, typename RandomDist, typename... Args>
void __cpu_rand_kernel_template(T* data, int64_t size, Args&&... args) {
    static thread_local std::random_device rd;
    static thread_local std::mt19937 gen(rd());
    RandomDist dist(std::forward<Args>(args)...);

    for (int64_t i = 0; i < size; i++) data[i] = dist(gen);
}

Tray __cpu_rand(const SmallVector& shape, Scalar low, Scalar high, DType dtype) {
    return TRAY_DISPATCH_FLOAT_TYPES(dtype, "rand", [&]{
        auto n = detail::compute_numel(shape);
        std::shared_ptr<scalar_t> data(new scalar_t[n]);
        std::visit([&](const auto& v){
            using T = std::decay_t<decltype(v)>;
            
            scalar_t l = static_cast<scalar_t>(std::get<T>(low));
            scalar_t h = static_cast<scalar_t>(std::get<T>(high));

            __cpu_rand_kernel_template<scalar_t, std::uniform_real_distribution<scalar_t>>(data.get(), n, l, h);
        }, low);
        return Tray(make_intrusive<TrayImpl>(shape, detail::compute_stride(shape), dtype, Device::CPU, data));
    });
}

Tray __cpu_randn(const SmallVector& shape, Scalar mean, Scalar std, DType dtype) {
    return TRAY_DISPATCH_FLOAT_TYPES(dtype, "randn", [&]{
        auto n = detail::compute_numel(shape);
        std::shared_ptr<scalar_t> data(new scalar_t[n]);
        std::visit([&](const auto& v){
            using T = std::decay_t<decltype(v)>;
            
            scalar_t m = static_cast<scalar_t>(std::get<T>(mean));
            scalar_t s = static_cast<scalar_t>(std::get<T>(std));

            __cpu_rand_kernel_template<scalar_t, std::normal_distribution<scalar_t>>(data.get(), n, m, s);
        }, mean);
        return Tray(make_intrusive<TrayImpl>(shape, detail::compute_stride(shape), dtype, Device::CPU, data));
    });
}

Tray __cpu_randint(const SmallVector&shape, Scalar low, Scalar high, DType dtype) {
    return TRAY_DISPATCH_INT_TYPES(dtype, "randint", [&]{
        auto n = detail::compute_numel(shape);
        std::shared_ptr<scalar_t> data(new scalar_t[n]);
        std::visit([&](const auto& v){
            using T = std::decay_t<decltype(v)>;
            
            scalar_t l = static_cast<scalar_t>(std::get<T>(low));
            scalar_t h = static_cast<scalar_t>(std::get<T>(high));

            __cpu_rand_kernel_template<scalar_t, std::uniform_int_distribution<scalar_t>>(data.get(), n, l, h);
        }, low);

        return Tray(make_intrusive<TrayImpl>(shape, detail::compute_stride(shape), dtype, Device::CPU, data));
    });
}

Tray zeros(const SmallVector& shape, DType dtype, Device device) {
    return TRAY_DISPATCH_ALL_TYPES(dtype, "zeros", [&]{
        int64_t total_size = 1;
        for (int i = 0; i < shape.size(); i++) total_size *= shape[i];
        
        std::shared_ptr<scalar_t> data(new scalar_t[total_size]);
        
        detail::Dispatcher::get_instance()
        .dispatch<void(void*, size_t)>
            ({detail::OpCode::zeros, dtype, device}, data.get(), total_size * sizeof(scalar_t));

        return Tray(make_intrusive<TrayImpl>(shape, detail::compute_stride(shape), dtype, device, data));    
    }); 
}


Tray full(const SmallVector& shape, Scalar val, DType dtype, Device device) {
    if (dtype == oven::DType::Undefined) {
        std::visit([&](const auto& v) {
            using T = std::decay_t<decltype(v)>;
            dtype = detail::CppTypeToDType_v<T>;
        }, val);
    }

    return TRAY_DISPATCH_ALL_TYPES(dtype, "full",[&]{
        int64_t total_size = 1;
        for (int i = 0; i < shape.size(); i++) total_size *= shape[i];

        std::shared_ptr<scalar_t> data(new scalar_t[total_size]);
        
        auto ptr = std::get_if<scalar_t>(&val);
        if(ptr == nullptr) {
            std::visit([&](const auto& v){
                using T = std::decay_t<decltype(v)>;
                
                scalar_t s = static_cast<scalar_t>(std::get<T>(val));

                detail::Dispatcher::get_instance()
                .dispatch<decltype(__cpu_full)>
                ({detail::OpCode::full, dtype, device}, data.get(), &s, total_size, dtype);
            }, val);
        } else {
            detail::Dispatcher::get_instance()
            .dispatch<decltype(__cpu_full)>
                ({detail::OpCode::full, dtype, device}, data.get(), ptr, total_size, dtype);
        }
        
        return Tray(make_intrusive<TrayImpl>(shape, detail::compute_stride(shape), dtype, device, data));
    });
}

Tray rand(const SmallVector &shape, Scalar low, Scalar high, DType dtype, Device device) {
    return oven::detail::Dispatcher::get_instance()
        .dispatch<Tray(const SmallVector&, Scalar, Scalar, DType)>({oven::detail::OpCode::rand, dtype, device}, shape, low, high, dtype);
}
Tray randint(const SmallVector& shape, Scalar low, Scalar high, DType dtype, Device device) {
    return oven::detail::Dispatcher::get_instance()
        .dispatch<Tray(const SmallVector&, Scalar, Scalar, DType)>({oven::detail::OpCode::randint, dtype, device}, shape, low, high, dtype);
}
Tray randn(const SmallVector& shape, Scalar mean, Scalar std, DType dtype, Device device) {
    return oven::detail::Dispatcher::get_instance()
        .dispatch<Tray(const SmallVector&, Scalar, Scalar, DType)>({oven::detail::OpCode::randn, dtype, device}, shape, mean, std, dtype);
}
Tray ones(const SmallVector& shape, DType dtype, Device device) {
    return full(shape, 1, dtype, device);
}

}// namespace oven

TRAY_REGISTER(zeros, CPU, oven::all_types, oven::__cpu_zeros);
TRAY_REGISTER(full, CPU, oven::all_types, oven::__cpu_full);
TRAY_REGISTER(rand, CPU, oven::all_types, oven::__cpu_rand);
TRAY_REGISTER(randint, CPU, oven::all_types, oven::__cpu_randint);
TRAY_REGISTER(randn, CPU, oven::all_types, oven::__cpu_randn);