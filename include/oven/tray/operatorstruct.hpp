#pragma once

#include "oven/tray/types.hpp"
#include <array>
#include <cstdint>
#include <cmath>
#include <span>

namespace oven::detail {

struct NegOp {
    static constexpr const char* name = "neg";

    template <typename T>
    T operator()(T a) {return -a;}
    
    static NegOp get_instance() {static NegOp op; return op;}
};

struct RecipOp {
    static constexpr const char* name = "reciprocal";

    template <typename T>
    T operator()(T a) {return 1 / a;}

    static RecipOp get_instance() {static RecipOp op; return op;}
};

struct ExpOp {
    static constexpr const char* name = "exp";
    static std::array<DType, 2> possible_types;
    template <typename T>
    T operator()(T a) {return std::exp(a);}

    static ExpOp get_instance() {static ExpOp op; return op;}
};

struct LogOp {
    static constexpr const char* name = "log";

    template <typename T>
    T operator()(T a) {return std::log(a);}

    static ExpOp get_instance() {static ExpOp op; return op;}
};



}// namespace oven::detail