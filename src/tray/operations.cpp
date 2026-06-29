#include <oven/tray/tray.hpp>
#include "oven/tray/dispatcher.hpp"
#include "oven/tray/types.hpp"
#include "oven/tray/utils.hpp"
#include <memory>
#include <optional>
#include <oven/utils/assert.hpp>
#include <oven/tray/operations.hpp>

namespace oven {

Tray add(const Tray& self, const Tray& other) {
    return detail::Dispatcher::get_instance()
        .dispatch<decltype(add)>
            ({detail::OpCode::add, self.device()},
                self, other);
}

Tray sub(const Tray& self, const Tray& other) {
    return detail::Dispatcher::get_instance()
        .dispatch<decltype(sub)>
            ({detail::OpCode::sub, self.device()},
                self, other);
}

Tray mul(const Tray& self, const Tray& other) {
    return detail::Dispatcher::get_instance()
        .dispatch<decltype(mul)>
            ({detail::OpCode::mul, self.device()},
                self, other);
}

Tray div(const Tray& self, const Tray& other) {
    return detail::Dispatcher::get_instance()
        .dispatch<decltype(div)>
            ({detail::OpCode::div, self.device()},
                self, other);
}

Tray le(const Tray& self, const Tray& other) {
    return detail::Dispatcher::get_instance()
        .dispatch<decltype(le)>
            ({detail::OpCode::le, self.device()},
                self, other);
}

Tray leq(const Tray& self, const Tray& other) {
    return detail::Dispatcher::get_instance()
        .dispatch<decltype(leq)>
            ({detail::OpCode::leq, self.device()},
                self, other);
}

Tray ge(const Tray& self, const Tray& other) {
    return detail::Dispatcher::get_instance()
        .dispatch<decltype(ge)>
            ({detail::OpCode::ge, self.device()},
                self, other);
}

Tray geq(const Tray& self, const Tray& other) {
    return detail::Dispatcher::get_instance()
        .dispatch<decltype(geq)>
            ({detail::OpCode::geq, self.device()},
                self, other);
}

Tray eq(const Tray& self, const Tray& other) {
    return detail::Dispatcher::get_instance()
        .dispatch<decltype(eq)>
            ({detail::OpCode::eq, self.device()},
                self, other);
}

Tray neq(const Tray& self, const Tray& other) {
    return detail::Dispatcher::get_instance()
        .dispatch<decltype(le)>
            ({detail::OpCode::neq, self.device()},
                self, other);
}

Tray where(const Tray& predicate, const Tray& self, const Tray& other) {
    std::optional<SmallVector> shape = detail::broadcastable(self.shape(), other.shape());
    OVEN_ASSERT(shape != std::nullopt, "Not broadcastable for ternery comparison operation.");
    shape = detail::broadcastable(predicate.shape(), *shape);
    OVEN_ASSERT(shape != std::nullopt, "Not broadcastable for != operation.");
    auto a_stride = detail::get_broadcasted_stride(self.shape(), *shape);
    auto b_stride = detail::get_broadcasted_stride(other.shape(), *shape);
    auto predicate_stride = detail::get_broadcasted_stride(predicate.shape(), *shape);

    return detail::Dispatcher::get_instance()
        .dispatch<oven::TerneryKernelType>({detail::OpCode::ternery, self.device()},
        static_cast<bool*>(predicate.data().get()), self.data().get(), other.data().get(), a_stride, b_stride, predicate_stride, *shape, self.dtype()
    );
}

Tray gather(const Tray& self, int64_t dim, const Tray& index) {
    OVEN_ASSERT(index.dtype() == DType::kInt32 || index.dtype() == DType::kInt64, "index tensor is not integral type");
    OVEN_ASSERT(index.ndim() == self.ndim(), "index tensor has invalid dimension");
    // check if index tensor is valid
    for (int64_t i = 0; i < index.ndim(); i++) {
        if (i != dim) {
            OVEN_ASSERT(self.shape()[i] == index.shape()[i], "given tensor and index tensor must have same dimension along axis other then given dim");
        } else {
            OVEN_ASSERT(index.shape()[i] < self.shape()[i], "index tensor's dimension of given axis must be smaller than that of given tensor");
        }
    }
    return detail::Dispatcher::get_instance()
        .dispatch<decltype(gather)>({detail::OpCode::gather, self.device()},
        self, dim, index
    );
}
}// namespace oven