#include "oven/tray/types.hpp"
#include "oven/utils/intrusive_ptr.hpp"
#include <algorithm>
#include <cstring>
#include <memory>
#include <oven/tray/tray.hpp>
#include <oven/tray/dispatcher.hpp>
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

Tray zeros(SmallVector shape, DType dtype, Device device) {
    return TRAY_DISPATCH_ALL_TYPES(dtype, "zeros", [&]{
        int64_t total_size = 1;
        for (int i = 0; i < shape.size(); i++) total_size *= shape[i];

        // we used shared ptr as our dynamic storage
        std::shared_ptr<void> data(
                new scalar_t[total_size],
                [](void* ptr){
                    delete[] static_cast<scalar_t*>(ptr);
                }
            );
        // it is important to make a custom deleter.
        // now we set all bytes to 0, using the dispatcher
        detail::Dispatcher::get_instance()
        .dispatch<void(void*, size_t)>
            ({detail::OpCode::zeros, device}, data.get(), total_size * sizeof(scalar_t));

        return Tray(make_intrusive<TrayImpl>(shape, detail::compute_stride(shape), dtype, device, data));    
    }); 
}


Tray full(SmallVector shape, Scalar val, DType dtype, Device device) {
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
                ({detail::OpCode::full, device}, data.get(), &s, total_size, dtype);
            }, val);
        } else {
            detail::Dispatcher::get_instance()
            .dispatch<decltype(__cpu_full)>
                ({detail::OpCode::full, device}, data.get(), ptr, total_size, dtype);
        }
        return Tray(make_intrusive<TrayImpl>(shape, detail::compute_stride(shape), dtype, device, data));
    });
}

Tray ones(SmallVector shape, DType dtype, Device device) {
    return full(shape, 1, dtype, device);
}

}// namespace oven

TRAY_REGISTER(zeros, CPU, oven::__cpu_zeros);
TRAY_REGISTER(full, CPU, oven::__cpu_full);