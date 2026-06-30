#pragma once

#include "oven/tray/types.hpp"
#include <array>
#include <cstdint>
#include <cmath>
#include <span>

namespace oven::detail {

struct AddOp {
    static constexpr const char* name = "add";

    template <typename T>
    T operator()(T a, T b) {return a + b;}

    static AddOp get_instance() {static AddOp op; return op;}
};


struct SubOp {
    static constexpr const char* name = "sub";

    template <typename T>
    T operator()(T a, T b) {return a - b;}

    static SubOp get_instance() {static SubOp op; return op;}
};


struct MulOp {
    static constexpr const char* name = "mul";

    template <typename T>
    T operator()(T a, T b) {return a * b;}

    static MulOp get_instance() {static MulOp op; return op;}
};


struct DivOp {
    static constexpr const char* name = "div";

    template <typename T>
    T operator()(T a, T b) {return a / b;}

    static DivOp get_instance() {static DivOp op; return op;}
};


struct LeOp {
    static constexpr const char* name = "le";

    template <typename T>
    bool operator()(T a, T b) {return a < b;}

    static LeOp get_instance() {static LeOp op; return op;}
};

struct LeqOp {
    static constexpr const char* name = "leq";

    template <typename T>
    bool operator()(T a, T b) {return a <= b;}

    static LeqOp get_instance() {static LeqOp op; return op;}
};

struct GeOp {
    static constexpr const char* name = "ge";

    template <typename T>
    bool operator()(T a, T b) {return a > b;}

    static GeOp get_instance() {static GeOp op; return op;}
};

struct GeqOp {
    static constexpr const char* name = "geq";

    template <typename T>
    bool operator()(T a, T b) {return a >= b;}

    static GeqOp get_instance() {static GeqOp op; return op;}
};

struct EqOp {
    static constexpr const char* name = "eq";

    template <typename T>
    bool operator()(T a, T b) {return a == b;}

    static EqOp get_instance() {static EqOp op; return op;}
};

struct NeqOp {
    static constexpr const char* name = "neq";

    template <typename T>
    bool operator()(T a, T b) {return a != b;}

    static NeqOp get_instance() {static NeqOp op; return op;}
};

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
    static constexpr std::array<DType, 2> possible_types = std::array<DType, 2>{DType::kFloat32, DType::kFloat64};
    template <typename T>
    T operator()(T a) {return std::exp(a);}

    static ExpOp get_instance() {static ExpOp op; return op;}
};

struct LogOp {
    static constexpr const char* name = "log";
    static constexpr std::array<DType, 2> possible_types = std::array<DType, 2>{DType::kFloat32, DType::kFloat64};
    template <typename T>
    T operator()(T a) {return std::log(a);}

    static ExpOp get_instance() {static ExpOp op; return op;}
};



}// namespace oven::detail