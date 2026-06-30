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
            ({detail::OpCode::add, self.dtype(), self.device()},
                self, other);
}

Tray sub(const Tray& self, const Tray& other) {
    return detail::Dispatcher::get_instance()
        .dispatch<decltype(sub)>
            ({detail::OpCode::sub, self.dtype(), self.device()},
                self, other);
}

Tray mul(const Tray& self, const Tray& other) {
    return detail::Dispatcher::get_instance()
        .dispatch<decltype(mul)>
            ({detail::OpCode::mul, self.dtype(), self.device()},
                self, other);
}

Tray div(const Tray& self, const Tray& other) {
    return detail::Dispatcher::get_instance()
        .dispatch<decltype(div)>
            ({detail::OpCode::div, self.dtype(), self.device()},
                self, other);
}

Tray le(const Tray& self, const Tray& other) {
    return detail::Dispatcher::get_instance()
        .dispatch<decltype(le)>
            ({detail::OpCode::le, self.dtype(), self.device()},
                self, other);
}

Tray leq(const Tray& self, const Tray& other) {
    return detail::Dispatcher::get_instance()
        .dispatch<decltype(leq)>
            ({detail::OpCode::leq, self.dtype(), self.device()},
                self, other);
}

Tray ge(const Tray& self, const Tray& other) {
    return detail::Dispatcher::get_instance()
        .dispatch<decltype(ge)>
            ({detail::OpCode::ge, self.dtype(), self.device()},
                self, other);
}

Tray geq(const Tray& self, const Tray& other) {
    return detail::Dispatcher::get_instance()
        .dispatch<decltype(geq)>
            ({detail::OpCode::geq, self.dtype(), self.device()},
                self, other);
}

Tray eq(const Tray& self, const Tray& other) {
    return detail::Dispatcher::get_instance()
        .dispatch<decltype(eq)>
            ({detail::OpCode::eq, self.dtype(), self.device()},
                self, other);
}

Tray neq(const Tray& self, const Tray& other) {
    return detail::Dispatcher::get_instance()
        .dispatch<decltype(le)>
            ({detail::OpCode::neq, self.dtype(), self.device()},
                self, other);
}

Tray where(const Tray& predicate, const Tray& self, const Tray& other) {
    std::optional<SmallVector> shape = detail::binop_broadcastable(self.shape(), other.shape());
    OVEN_ASSERT(shape != std::nullopt, "Not broadcastable for ternery comparison operation.");
    shape = detail::binop_broadcastable(predicate.shape(), *shape);
    OVEN_ASSERT(shape != std::nullopt, "Not broadcastable for != operation.");
    auto a_stride = detail::get_right_aligned_broadcasted_stride(self.shape(), self.stride(), *shape);
    auto b_stride = detail::get_right_aligned_broadcasted_stride(other.shape(), other.stride(), *shape);
    auto predicate_stride = detail::get_right_aligned_broadcasted_stride(predicate.shape(), predicate.stride(), *shape);

    return detail::Dispatcher::get_instance()
        .dispatch<oven::TerneryKernelType>({detail::OpCode::ternery, self.dtype(), self.device()},
        static_cast<bool*>(predicate.data().get()), self.data().get(), other.data().get(), a_stride, b_stride, predicate_stride, *shape, self.dtype()
    );
}

Tray gather(const Tray& self, int64_t dim, const Tray& index) {
    // The gather operation gets the value of 'dim'-dimension from self, according to the index.
    OVEN_ASSERT(0 <= dim && dim < index.ndim(), "dim should be smaller than ndim of index.");
    return detail::Dispatcher::get_instance()
        .dispatch<decltype(gather)>({detail::OpCode::gather, self.dtype(), self.device()},
        self, dim, index
    );
}

void scatter_(const Tray& self, int64_t dim, const Tray& index, const Tray& src) {
    // The scatter operation gets index from index tray and "scatter" the values of src to target, according to the index.
    OVEN_ASSERT(index.shape() == src.shape(), "The shape of index and src must be same for scatter_ operation.");
    detail::Dispatcher::get_instance()
        .dispatch<decltype(scatter_)>({detail::OpCode::scatter_, self.dtype(), self.device()}, 
        self, dim, index, src
    );
}

// unary operations
Tray exp(Tray& self) {
    return detail::Dispatcher::get_instance()
        .dispatch<decltype(exp)>({detail::OpCode::exp, self.dtype(), self.device()}, self);
}

}// namespace oven