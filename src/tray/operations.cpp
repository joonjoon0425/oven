#include <oven/tray/tray.hpp>
#include "oven/tray/dispatcher.hpp"
#include "oven/tray/types.hpp"
#include "oven/tray/utils.hpp"
#include <memory>
#include <optional>
#include <oven/utils/assert.hpp>
#include <oven/tray/operations.hpp>

// rules for using TRAY_DEFINE_BINOP
// 1. broadcasting equals that of arithmetic operations.
// 2. operator kernel enum name must equal op, op_ts, op_st.

#define TRAY_DEFINE_BINOP(opname)\
Tray opname(const Tray& self, const Tray& other) {\
    return detail::Dispatcher::get_instance()\
        .dispatch<Tray(const Tray&, const Tray&)>\
        ({detail::OpCode::opname, self.dtype(), self.device()},\
            self, other);\
}\
Tray opname(const Tray& self, const Scalar& scalar) {\
    return detail::Dispatcher::get_instance()\
        .dispatch<Tray(const Tray&, const Scalar&)>\
        ({detail::OpCode::opname##_ts, self.dtype(), self.device()},\
            self, scalar);\
}\
Tray opname(const Scalar& scalar, const Tray& self) {\
    return detail::Dispatcher::get_instance()\
        .dispatch<Tray(const Scalar&, const Tray&)>\
        ({detail::OpCode::opname##_st, self.dtype(), self.device()},\
            scalar, self);\
}

namespace oven {
TRAY_DEFINE_BINOP(add);
TRAY_DEFINE_BINOP(sub);
TRAY_DEFINE_BINOP(mul);
TRAY_DEFINE_BINOP(div);
TRAY_DEFINE_BINOP(le);
TRAY_DEFINE_BINOP(leq);
TRAY_DEFINE_BINOP(ge);
TRAY_DEFINE_BINOP(geq);
TRAY_DEFINE_BINOP(eq);
TRAY_DEFINE_BINOP(neq);

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
Tray exp(const Tray& self) {
    return detail::Dispatcher::get_instance()
        .dispatch<decltype(exp)>({detail::OpCode::exp, self.dtype(), self.device()}, self);
}

Tray neg(const Tray& self) {
    return detail::Dispatcher::get_instance()
        .dispatch<decltype(neg)>({detail::OpCode::neg, self.dtype(), self.device()}, self);
}

Tray recip(const Tray& self) {
    return detail::Dispatcher::get_instance()
        .dispatch<decltype(recip)>({detail::OpCode::recip, self.dtype(), self.device()}, self);
}

Tray log(const Tray& self) {
    return detail::Dispatcher::get_instance()
        .dispatch<decltype(log)>({detail::OpCode::log, self.dtype(), self.device()}, self);
}

}// namespace oven