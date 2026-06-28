#include "oven/tray/types.hpp"
#include <cstring>
#include <memory>
#include <oven/tray/trayimpl.hpp>
#include <oven/tray/dispatcher.hpp>

namespace oven {

Tray __gpu_zeros() {
    // TODO
}

Tray __cpu_zeros(SmallVector shape, DType dtype) {
    return TRAY_DISPATCH_ALL_TYPES(dtype, "zeros", [&]
    {
        scalar_t scalar;
        int64_t total_size = 1;
        for (int i = 0; i < shape.size(); i++) total_size *= shape[i];

        // we used shared ptr as our dynamic storage
        std::shared_ptr<void> data(
                new scalar_t[total_size],
                [=](void* ptr){
                    delete[] static_cast<scalar_t*>(ptr);
                }
            );
        // it is important to make a custom deleter.
        // now we set all bytes to 0
        std::memset(data.get(), 0, total_size * sizeof(scalar_t));
        return Tray(std::make_shared<TrayImpl>(shape, detail::compute_stride(shape), dtype, Device::CPU, data));
    }
    );
}

Tray zeros(SmallVector shape, DType dtype, Device device) {
    return detail::Dispatcher::get_instance()
        .dispatch<decltype(__cpu_zeros)>
            ({detail::OpCode::zeros, device}, shape, dtype);
}


}// namespace oven

TRAY_REGISTER(zeros, CPU, oven::__cpu_zeros);