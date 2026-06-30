# pragma once

#include <cmath>
#include <type_traits>
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